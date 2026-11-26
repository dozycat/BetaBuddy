#!/bin/bash
# Try the automatic edit on a clip of your own, on a connected device or a running emulator, with
# the bundled people detector. Nothing is added to the repository and nothing leaves your machine.
#
#   ./android/scripts/edit-backtest.sh <video> <plan.json> <output directory>
#
# <plan.json> is an edit plan for that video, as the analysis page of a gym pack hands it over
# (the `edit` field of its result). The script installs the debug build and its tests, puts the two
# files into the app's own cache, runs the optional test EditBacktest, and brings back:
#
#   full.mp4           every switch on
#   plain.mp4          every switch off
#   removal-only.mp4   passer-by removal alone
#   report.txt         who was found, who was taken out or left and why, plate, times
#   sheet.jpg          the source next to removal-only.mp4 at eight moments (needs ffmpeg)
#
# Needs JAVA_HOME (JDK 17), the Android SDK (ANDROID_HOME, or sdk.dir in android/local.properties)
# and python3. Keep clips and what comes out of them outside the repository.
set -euo pipefail
if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <video> <plan.json> <output directory>" >&2
  exit 64
fi
VIDEO="$1"; PLAN="$2"; OUT="$3"
[[ -f "$VIDEO" ]] || { echo "No such video: $VIDEO" >&2; exit 1; }
[[ -f "$PLAN" ]] || { echo "No such plan: $PLAN" >&2; exit 1; }
ANDROID_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SDK="${ANDROID_HOME:-}"
if [[ -z "$SDK" && -f "$ANDROID_ROOT/local.properties" ]]; then
  SDK="$(sed -n 's/^sdk\.dir=//p' "$ANDROID_ROOT/local.properties")"
fi
ADB="${SDK:+$SDK/platform-tools/}adb"
APP=com.betabuddy.android
WORK=cache/edit-backtest
STAGE=/data/local/tmp/betabuddy-edit-backtest

# The plan as the app stores it. A file that wraps it ({"edit": {...}}, as a saved analysis message
# does) is unwrapped; one that says no plan was made ({"edit": null}) ends the run here: for such a
# clip the app shows no edit at all.
BARE="$(mktemp)"
trap 'rm -f "$BARE"' EXIT
python3 - "$PLAN" "$BARE" <<'PY' || exit 2
import json, sys
data = json.load(open(sys.argv[1]))
if isinstance(data, dict) and "v" not in data and "edit" in data:
    if not isinstance(data["edit"], dict):
        sys.stderr.write("This file holds no plan (edit: %s, why: %s). The app offers no edit for such a clip.\n" % (json.dumps(data["edit"]), data.get("why")))
        sys.exit(1)
    data = data["edit"]
json.dump(data, open(sys.argv[2], "w"))
PY
PLAN="$BARE"

"$ADB" get-state >/dev/null 2>&1 || { echo "No device or emulator is connected." >&2; exit 1; }
(cd "$ANDROID_ROOT" && ./gradlew -q :app:installDebug :app:installDebugAndroidTest)

# Into the app's own cache, by way of the debug build's run-as: no storage permission is involved.
"$ADB" shell "rm -rf $STAGE && mkdir -p $STAGE"
"$ADB" push "$VIDEO" "$STAGE/clip.mp4" >/dev/null
"$ADB" push "$PLAN" "$STAGE/plan.json" >/dev/null
"$ADB" shell "chmod 755 $STAGE && chmod 644 $STAGE/*"
"$ADB" shell "run-as $APP sh -c 'rm -rf $WORK && mkdir -p $WORK && cp $STAGE/clip.mp4 $STAGE/plan.json $WORK/'"
"$ADB" shell "rm -rf $STAGE"

"$ADB" shell am instrument -w -e class "$APP.edit.EditBacktest" -e editClip clip.mp4 -e editPlan plan.json \
  "$APP.test/androidx.test.runner.AndroidJUnitRunner" | tee /dev/stderr | grep -q "^OK (1 test)" \
  || { echo "The backtest did not finish; see above." >&2; exit 1; }

mkdir -p "$OUT"
for name in full.mp4 plain.mp4 removal-only.mp4 report.txt; do
  "$ADB" exec-out run-as "$APP" cat "$WORK/out/$name" > "$OUT/$name"
done
"$ADB" shell "run-as $APP rm -rf $WORK"
cat "$OUT/report.txt"

# The source next to the removal-only render at eight moments of the attempt. That render plays the
# attempt alone at normal speed and full frame, so its time is the source's minus the attempt's start.
if command -v ffmpeg >/dev/null 2>&1; then
  TILES="$(mktemp -d)"
  trap 'rm -rf "$TILES" "$BARE"' EXIT
  python3 - "$PLAN" > "$TILES/times" <<'PY'
import json, sys
plan = json.load(open(sys.argv[1]))
start, end = plan["trim"]
for index in range(8):
    moment = start + (end - start) * (index + 0.5) / 8
    print("%.3f %.3f" % (moment, moment - start))
PY
  index=0
  while read -r source rendered; do
    ffmpeg -v error -y -ss "$source" -i "$VIDEO" -frames:v 1 -vf "scale=-2:640" "$TILES/a$index.png"
    ffmpeg -v error -y -ss "$rendered" -i "$OUT/removal-only.mp4" -frames:v 1 -vf "scale=-2:640" "$TILES/b$index.png"
    ffmpeg -v error -y -i "$TILES/a$index.png" -i "$TILES/b$index.png" \
      -filter_complex "[0][1]hstack=inputs=2,drawtext=text='${source}s  source | removal-only':x=8:y=8:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.6,pad=iw+12:ih+12:6:6:white" \
      "$TILES/pair$index.png" 2>/dev/null \
      || ffmpeg -v error -y -i "$TILES/a$index.png" -i "$TILES/b$index.png" -filter_complex "[0][1]hstack=inputs=2,pad=iw+12:ih+12:6:6:white" "$TILES/pair$index.png"
    index=$((index + 1))
  done < "$TILES/times"
  ffmpeg -v error -y -i "$TILES/pair%d.png" -filter_complex "tile=4x2" -frames:v 1 -q:v 3 "$OUT/sheet.jpg"
  echo "Contact sheet: $OUT/sheet.jpg (each pair: source on the left, removal-only on the right)"
else
  echo "ffmpeg is not installed: no contact sheet was made."
fi

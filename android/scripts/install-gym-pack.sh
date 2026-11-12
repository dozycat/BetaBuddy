#!/bin/bash
# Install a gym pack into android/app/src/main/assets/gympack/ so the next build bundles it.
#
#   ./android/scripts/install-gym-pack.sh /path/to/GymPack    verify, then install
#   ./android/scripts/install-gym-pack.sh --remove            back to a build without a pack
#
# The pack (venue model, routes, solver, web page) is produced outside this repository and
# must never be committed; the assets directory is ignored by Git. Interface: docs/ios/gym-design.md.
# Same checks as ios/scripts/install-gym-pack.sh, plus one for Android: the build tools leave
# some file names out of an APK's assets, so a pack that uses them is refused here instead of
# turning up incomplete on the phone.
set -euo pipefail
ANDROID_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ASSETS="$ANDROID_ROOT/app/src/main/assets"
TARGET="$ASSETS/gympack"

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <gym pack directory> | --remove" >&2
  exit 64
fi
if [[ "$1" == "--remove" ]]; then
  rm -rf "$TARGET"
  echo "Removed $TARGET. Builds now ship without a gym pack."
  exit 0
fi

SOURCE="$(cd "$1" && pwd)"
[[ -f "$SOURCE/manifest.json" ]] || { echo "No manifest.json in $SOURCE" >&2; exit 1; }
STAGE="$(mktemp -d "$ANDROID_ROOT/.gympack.XXXXXX")"
trap 'rm -rf "$STAGE"' EXIT

# Verify every listed file against its SHA-256 and copy only what the manifest lists.
# Mirrors GymManifest in app/src/main/java/com/betabuddy/android/gym/GymManifest.kt.
/usr/bin/python3 -I - "$SOURCE" "$STAGE" <<'PY'
import hashlib, json, os, shutil, sys

source, stage = sys.argv[1], sys.argv[2]
def fail(message):
    sys.exit("Gym pack rejected: " + message)
def first(data, *names):
    for name in names:
        if data.get(name) is not None:
            return data[name]
    return None
def safe(path):
    return (isinstance(path, str) and path and not path.startswith("/") and "\\" not in path
            and all(part not in ("", ".", "..") for part in path.split("/")))

def packaged(path):
    # aapt's default ignore pattern for assets: dot files, directories starting with "_",
    # editor backups and a few version-control leftovers.
    parts = path.split("/")
    return not any(part.startswith(".") or part.endswith("~") or part.lower().endswith(".scc")
                   or part.lower() in ("cvs", "thumbs.db", "picasa.ini") for part in parts) \
        and not any(part.startswith("_") for part in parts[:-1])

try:
    with open(os.path.join(source, "manifest.json"), encoding="utf-8") as handle:
        manifest = json.load(handle)
except (OSError, ValueError) as error:
    fail("manifest.json is unreadable (%s)" % error)
if not isinstance(manifest, dict):
    fail("manifest.json is not an object")

schema = first(manifest, "schema", "schemaVersion", "schema_version") or 1
if schema != 1:
    fail("unsupported schema %r" % schema)
gym = manifest.get("gym") if isinstance(manifest.get("gym"), dict) else {}
fields = {
    "packVersion": first(manifest, "packVersion", "pack_version", "version"),
    "generatedAt": first(manifest, "generatedAt", "generated_at"),
    "gymName": first(manifest, "gymName", "gym_name") or gym.get("name"),
    "routeCount": first(manifest, "routeCount", "route_count"),
}
for name, value in fields.items():
    if value is None:
        fail("manifest is missing " + name)
if not isinstance(fields["routeCount"], int):
    fail("routeCount is not an integer")
entry = manifest.get("entry") or "index.html"

listed = manifest.get("files")
if isinstance(listed, dict):
    files = {path: (value.get("sha256") if isinstance(value, dict) else value) for path, value in listed.items()}
elif isinstance(listed, list):
    files = {item.get("path"): item.get("sha256") for item in listed if isinstance(item, dict)}
    if len(files) != len(listed):
        fail("files has malformed or duplicate entries")
else:
    fail("manifest has no files list")
if entry not in files:
    fail("entry %s is not in the files list" % entry)

for path, expected in sorted(files.items()):
    if not safe(path):
        fail("unsafe path %r" % path)
    if not packaged(path):
        fail("%s would be left out of the APK (the Android build skips dot files and directories starting with _)" % path)
    origin = os.path.join(source, path)
    if os.path.islink(origin) or not os.path.isfile(origin):
        fail("missing file " + path)
    digest = hashlib.sha256()
    with open(origin, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    if not isinstance(expected, str) or digest.hexdigest() != expected.lower():
        fail("SHA-256 mismatch for " + path)
    target = os.path.join(stage, path)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    shutil.copyfile(origin, target)

# Anything the manifest does not vouch for stays out of the app.
for folder, _, names in os.walk(source):
    for name in names:
        relative = os.path.relpath(os.path.join(folder, name), source)
        if relative != "manifest.json" and relative not in files and not name.startswith("."):
            fail("file not listed in manifest: " + relative)
shutil.copyfile(os.path.join(source, "manifest.json"), os.path.join(stage, "manifest.json"))
size = sum(os.path.getsize(os.path.join(stage, path)) for path in files)
print("Verified %d files, %.1f MB: %s, %d routes, pack %s" % (
    len(files), size / 1e6, fields["gymName"], fields["routeCount"], fields["packVersion"]))
PY

mkdir -p "$ASSETS"
rm -rf "$TARGET"
mv "$STAGE" "$TARGET"
trap - EXIT
chmod -R u+rwX,go+rX "$TARGET"
echo "Installed to $TARGET (ignored by Git). Rebuild the app to bundle it."

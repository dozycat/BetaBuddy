#!/bin/bash
set -euo pipefail
IOS_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENDOR_ROOT="$IOS_ROOT/Vendor"
command -v xcodegen >/dev/null || { echo 'Install XcodeGen first: https://github.com/yonaskolb/XcodeGen'; exit 1; }
mkdir -p "$VENDOR_ROOT"
TASK_TMP="$(mktemp -d)"
trap 'rm -rf "$TASK_TMP"' EXIT

fetch_sdk() {
  local sdk_name="$1" sdk_url="$2" sdk_hash="$3"
  if [[ ! -d "$VENDOR_ROOT/$sdk_name/frameworks/$sdk_name.xcframework" ]]; then
    curl --fail --location --retry 3 "$sdk_url" -o "$TASK_TMP/$sdk_name.tar.gz"
    echo "$sdk_hash  $TASK_TMP/$sdk_name.tar.gz" | shasum -a 256 -c -
    mkdir -p "$TASK_TMP/$sdk_name"
    tar -xzf "$TASK_TMP/$sdk_name.tar.gz" -C "$TASK_TMP/$sdk_name"
    mv "$TASK_TMP/$sdk_name" "$VENDOR_ROOT/$sdk_name"
  fi
}

# Pinned official MediaPipe 1.0.0 SDK distribution, matching the published podspecs.
fetch_sdk MediaPipeTasksVision https://dl.google.com/cpdc/20260727-225049/MediaPipeTasksVision-1.0.0.tar.gz bd386a43caa40ae957e5f42907076c885992383e642188d64ed96dad5bbdcb76
fetch_sdk MediaPipeTasksCommon https://dl.google.com/cpdc/20260727-225047/MediaPipeTasksCommon-1.0.0.tar.gz 952403622f5f331ccdc65a1931f5fc6a4c798bd129579fb8accc94c16aed8369
MODEL_PATH="$IOS_ROOT/BetaBuddy/Resources/pose_landmarker_full.task"
if [[ ! -f "$MODEL_PATH" ]]; then
  curl --fail --location --retry 3 https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task -o "$TASK_TMP/model.task"
  echo "5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1  $TASK_TMP/model.task" | shasum -a 256 -c -
  mv "$TASK_TMP/model.task" "$MODEL_PATH"
fi
echo "5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1  $MODEL_PATH" | shasum -a 256 -c -
"$IOS_ROOT/scripts/prepare-local-graph.sh"
xcodegen generate --spec "$IOS_ROOT/project.yml" --project "$IOS_ROOT"
echo "Open $IOS_ROOT/BetaBuddy.xcodeproj and choose a signing team for your iPhone."

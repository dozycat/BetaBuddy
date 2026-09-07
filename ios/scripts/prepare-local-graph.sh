#!/bin/bash
set -euo pipefail
IOS_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOURCE="$IOS_ROOT/Vendor/MediaPipeTasksCommon/frameworks/graph_libraries"
OUTPUT="$IOS_ROOT/Vendor/LocalGraph"
mkdir -p "$OUTPUT"
# Fail closed if a future SDK no longer exposes this factory object.
# Keep official binaries unchanged, and create a local derivative at build setup.
if ! ar -t "$SOURCE/libMediaPipeTasksCommon_device_graph.a" | grep -x 'logging_factory.o' >/dev/null; then
  echo 'SDK logger factory changed. Review required before using this SDK.' >&2
  exit 1
fi
cp "$SOURCE/libMediaPipeTasksCommon_device_graph.a" "$OUTPUT/device.a"
ar -d "$OUTPUT/device.a" logging_factory.o
for SDK_ARCH in arm64 x86_64; do
  lipo "$SOURCE/libMediaPipeTasksCommon_simulator_graph.a" -thin "$SDK_ARCH" -output "$OUTPUT/simulator-$SDK_ARCH.a"
  ar -d "$OUTPUT/simulator-$SDK_ARCH.a" logging_factory.o
done
lipo -create "$OUTPUT/simulator-arm64.a" "$OUTPUT/simulator-x86_64.a" -output "$OUTPUT/simulator.a"
rm "$OUTPUT/simulator-arm64.a" "$OUTPUT/simulator-x86_64.a"

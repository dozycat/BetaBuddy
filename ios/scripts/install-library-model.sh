#!/bin/sh
# Compiles a locally exported encoder after matching it to the installed private pack policy.
set -eu
if [ "$#" -ne 1 ]; then echo 'Usage: install-library-model.sh <LibraryImageEncoder.mlpackage>' >&2; exit 2; fi
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
IOS_DIR=$(dirname "$SCRIPT_DIR")
python3 - "$1" "$IOS_DIR/GymPack/library-classifier.json" <<'PY'
import hashlib, json, sys
from pathlib import Path
root=Path(sys.argv[1]); policy=json.loads(Path(sys.argv[2]).read_text())
actual={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
assert actual == policy['modelFiles'], 'Encoder does not match installed classifier policy'
PY
mkdir -p "$IOS_DIR/Vendor"
STAGING=$(mktemp -d "$IOS_DIR/Vendor/.library-model.XXXXXX")
trap 'rm -rf "$STAGING"' EXIT
xcrun coremlcompiler compile "$1" "$STAGING"
test -d "$STAGING/LibraryImageEncoder.mlmodelc"
rm -rf "$IOS_DIR/Vendor/LibraryImageEncoder.mlmodelc"
mv "$STAGING/LibraryImageEncoder.mlmodelc" "$IOS_DIR/Vendor/LibraryImageEncoder.mlmodelc"
echo 'Installed locally compiled library image encoder.'

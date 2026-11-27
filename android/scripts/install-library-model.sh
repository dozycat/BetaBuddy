#!/bin/sh
# Installs an exported encoder only when it matches the installed private pack's policy.
set -eu
if [ "$#" -ne 1 ]; then echo 'Usage: install-library-model.sh <library-image-encoder.onnx>' >&2; exit 2; fi
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ANDROID_DIR=$(dirname "$SCRIPT_DIR")
python3 - "$1" "$ANDROID_DIR/app/src/main/assets" <<'PY'
import hashlib, json, os, shutil, sys, tempfile
from pathlib import Path
source=Path(sys.argv[1]); assets=Path(sys.argv[2])
policy=json.loads((assets/'gympack/library-classifier.json').read_text())['androidModel']
assert policy['file']=='library-image-encoder.onnx'
assert hashlib.sha256(source.read_bytes()).hexdigest()==policy['sha256'], 'Encoder does not match installed classifier policy'
handle,temporary=tempfile.mkstemp(prefix='.library-model-',dir=assets)
os.close(handle)
try:
    shutil.copyfile(source,temporary)
    os.replace(temporary, assets/policy['file'])
finally:
    if os.path.exists(temporary): os.unlink(temporary)
PY
echo 'Installed local library image encoder.'

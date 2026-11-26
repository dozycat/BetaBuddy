#!/bin/sh
# Run by XcodeGen after it writes the project (`options.postGenCommand` in project.yml).
# Switches Xcode's Thread Performance Checker off for the test action of every scheme: the checker
# crashes in its own bookkeeping while Core Image renders on several threads (the edit renderer's
# tests). XcodeGen can only set this for the run action, so the test action is patched here.
# The checker is a debugging aid and is not part of the shipped app.
set -eu
cd "$(dirname "$0")/.."
for scheme in BetaBuddy.xcodeproj/xcshareddata/xcschemes/*.xcscheme; do
  python3 - "$scheme" <<'PY'
import sys
path = sys.argv[1]
text = open(path).read()
head = '   <TestAction\n'
flag = '      disablePerformanceAntipatternChecker = "YES"\n'
start = text.find(head)
if start >= 0 and flag not in text[start:text.find('>', start)]:
    text = text[:start + len(head)] + flag + text[start + len(head):]
    open(path, 'w').write(text)
PY
done

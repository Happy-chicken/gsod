#!/usr/bin/env bash
# Update gsod/, scripts/, train.py and test.py from the GitHub branch; nothing else.
set -euo pipefail
cd "$(dirname "$0")/.."

BRANCH=claude/relaxed-cannon-qtduhi
URL="https://github.com/Happy-chicken/gsod/archive/refs/heads/${BRANCH}.zip"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

wget -q -O "$TMP/gsod.zip" "$URL"
unzip -q "$TMP/gsod.zip" -d "$TMP"
SRC=$(find "$TMP" -mindepth 1 -maxdepth 1 -type d)

rsync -a --delete "$SRC/gsod/" ./gsod/
# Trailing slash on the source: copy the contents, not a nested scripts/scripts/.
# --delete also removes a scripts/scripts/ left by older versions of this script.
rsync -a --delete "$SRC/scripts/" ./scripts/
cp "$SRC/train.py" "$SRC/test.py" ./

echo "updated gsod, scripts, train.py, test.py from $BRANCH"

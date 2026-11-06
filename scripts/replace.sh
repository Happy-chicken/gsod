#!/usr/bin/env bash
# Update gsod/, train.py and test.py from the GitHub branch; nothing else.
set -euo pipefail
cd "$(dirname "$0")/.."

BRANCH=claude/charming-lamport-iy8f83
URL="https://github.com/Happy-chicken/gsod/archive/refs/heads/${BRANCH}.zip"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

wget -q -O "$TMP/gsod.zip" "$URL"
unzip -q "$TMP/gsod.zip" -d "$TMP"
SRC=$(find "$TMP" -mindepth 1 -maxdepth 1 -type d)

rsync -a --delete "$SRC/gsod/" ./gsod/
cp "$SRC/train.py" "$SRC/test.py" ./

echo "updated gsod/, train.py, test.py from $BRANCH"

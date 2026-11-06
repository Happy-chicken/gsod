#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/cuda_env.sh
PYTHON=${PYTHON:-python}
if [[ -x .venv/bin/python ]]; then PYTHON=.venv/bin/python; fi
$PYTHON -u test.py --dataset primordia "$@"

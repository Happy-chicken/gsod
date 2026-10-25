#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/cuda_env.sh
DATA=clevr_567_train
if [[ $# -gt 0 && $1 != -* ]]; then
    DATA=$1
    shift
fi
PYTHON=${PYTHON:-python}
if [[ -x .venv/bin/python ]]; then PYTHON=.venv/bin/python; fi
$PYTHON -u train.py --dataset clevr567 --data "$DATA" "$@"

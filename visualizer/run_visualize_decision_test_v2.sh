#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" visualizer/visualize_decision.py \
  --decision-dir /nas/nfs/large-model/vince/data/xd-online-las-data/test-v2/inference/decision \
  --output-dir /nas/nfs/large-model/vince/data/xd-online-las-data/test-v2/inference/decision/vis \
  --overwrite \
  "$@"

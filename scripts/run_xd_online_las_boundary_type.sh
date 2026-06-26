#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" scripts/run_vlm_boundary_type_batch.py \
  --model /nas/nfs/large-model/vince/model/Qwen3.6-27B-PER-SFT-260529 \
  --prompt-file "${PROJECT_ROOT}/prompt-perception/prompt-classification-boundary-type.txt" \
  --image-dir /nas/nfs/large-model/vince/data/xd-online-las-data/test-v2/images \
  --center-line-dir /nas/nfs/large-model/vince/data/xd-online-las-data/test-v2/center_line_2d \
  --output-dir /nas/nfs/large-model/vince/data/xd-online-las-data/test-v2/inference \
  "$@"

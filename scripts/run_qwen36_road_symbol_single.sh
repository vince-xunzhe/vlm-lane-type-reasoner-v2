#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

MODEL_PATH="/nas/nfs/large-model/vince/model/Qwen3.6-27B-PER-SFT-260529"
IMAGE_PATH="/nas/nfs/large-model/vince/data/xd-online-las-data/test/images/xd_4403-0-00L061-250707_038-0-020117-596-000354.jpg"
PROMPT_PATH="/nas/nfs/large-model/vince/code/vlm-grounding-reasoner-release-v1/prompt-perception/prompt-box-road-symbol.txt"
OUTPUT_DIR="/nas/nfs/large-model/vince/data/xd-online-las-data/test/inference_single"

cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" scripts/run_vlm_batch_inference.py \
  --model "${MODEL_PATH}" \
  --prompt-file "${PROMPT_PATH}" \
  --image-file "${IMAGE_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --overwrite \
  "$@"

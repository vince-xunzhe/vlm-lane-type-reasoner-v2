#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" associator/associate_elements_to_lanes.py \
  --inference-dir /nas/nfs/large-model/vince/data/xd-online-las-data/test-v2/inference \
  --image-dir /nas/nfs/large-model/vince/data/xd-online-las-data/test-v2/images \
  --center-line-dir /nas/nfs/large-model/vince/data/xd-online-las-data/test-v2/center_line_2d \
  --depth-dir /nas/nfs/large-model/vince/data/xd-online-las-data/test-v2/depth \
  --sam-dir /nas/nfs/large-model/vince/data/xd-online-las-data/test-v2/sam3 \
  --output-dir /nas/nfs/large-model/vince/data/xd-online-las-data/test-v2/inference/association \
  "$@"

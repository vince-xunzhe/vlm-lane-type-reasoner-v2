#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

MODEL_PATH="/nas/nfs/large-model/vince/model/Qwen3.6-27B-PER-SFT-260529"
IMAGE_DIR="/nas/nfs/large-model/vince/data/xd-online-las-data/test-v2/images"
OUTPUT_DIR="/nas/nfs/large-model/vince/data/xd-online-las-data/test-v2/inference"
PROMPT_DIR="${PROJECT_ROOT}/prompt-perception"
USER_GPU_IDS=""
USER_NUM_GPUS=""
REMAINING_ARGS=()

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --gpu-ids)
      USER_GPU_IDS="$2"
      shift 2
      ;;
    --gpu-ids=*)
      USER_GPU_IDS="${1#*=}"
      shift
      ;;
    --num-gpus)
      USER_NUM_GPUS="$2"
      shift 2
      ;;
    --num-gpus=*)
      USER_NUM_GPUS="${1#*=}"
      shift
      ;;
    *)
      REMAINING_ARGS+=("$1")
      shift
      ;;
  esac
done

detect_gpu_ids() {
  local -n out_ref="$1"
  out_ref=()
  if [[ -n "${USER_GPU_IDS}" ]]; then
    IFS=',' read -r -a out_ref <<< "${USER_GPU_IDS}"
  elif [[ -n "${GPU_IDS:-}" ]]; then
    IFS=',' read -r -a out_ref <<< "${GPU_IDS}"
  elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "-1" ]]; then
    IFS=',' read -r -a out_ref <<< "${CUDA_VISIBLE_DEVICES}"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    mapfile -t out_ref < <(nvidia-smi --query-gpu=index --format=csv,noheader | sed 's/[[:space:]]//g' | sed '/^$/d')
  fi
  if [[ "${#out_ref[@]}" -eq 0 ]] && command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    mapfile -t out_ref < <("${PYTHON_BIN}" - <<'PY'
try:
    import torch
    for index in range(torch.cuda.device_count()):
        print(index)
except Exception:
    pass
PY
)
  fi
  local cleaned=()
  local gpu_id
  for gpu_id in "${out_ref[@]}"; do
    gpu_id="${gpu_id//[[:space:]]/}"
    if [[ -n "${gpu_id}" ]]; then
      cleaned+=("${gpu_id}")
    fi
  done
  out_ref=("${cleaned[@]}")

  local requested_num_gpus="${USER_NUM_GPUS:-${NUM_GPUS:-}}"
  if [[ -n "${requested_num_gpus}" && "${requested_num_gpus}" =~ ^[0-9]+$ && "${requested_num_gpus}" -gt 0 && "${#out_ref[@]}" -gt "${requested_num_gpus}" ]]; then
    out_ref=("${out_ref[@]:0:${requested_num_gpus}}")
  fi
}

has_arg() {
  local wanted="$1"
  shift
  for item in "$@"; do
    if [[ "${item}" == "${wanted}" || "${item}" == "${wanted}="* ]]; then
      return 0
    fi
  done
  return 1
}

count_images() {
  find "${IMAGE_DIR}" -maxdepth 1 -type f \( \
    -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.bmp' -o \
    -iname '*.webp' -o -iname '*.tif' -o -iname '*.tiff' \
  \) | wc -l | tr -d ' '
}

cd "${PROJECT_ROOT}"

gpu_ids=()
detect_gpu_ids gpu_ids
if [[ "${#gpu_ids[@]}" -eq 0 ]]; then
  echo "[multi-gpu] no GPU detected; falling back to one worker with the current environment" >&2
  gpu_ids=("")
fi

total_items="$(count_images)"
if [[ "${total_items}" -le 0 ]]; then
  echo "[multi-gpu] no images found: ${IMAGE_DIR}" >&2
  exit 1
fi

worker_count="${#gpu_ids[@]}"
if [[ "${worker_count}" -gt "${total_items}" ]]; then
  worker_count="${total_items}"
fi
chunk_size=$(( (total_items + worker_count - 1) / worker_count ))
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
log_dir="${MULTI_GPU_LOG_DIR:-${OUTPUT_DIR}/_multi_gpu_logs/images_${timestamp}}"
mkdir -p "${log_dir}"

echo "[multi-gpu] images=${total_items} workers=${worker_count} gpus=${gpu_ids[*]:-none}"
echo "[multi-gpu] log_dir=${log_dir}"

pids=()
for worker_idx in $(seq 0 $((worker_count - 1))); do
  gpu_id="${gpu_ids[${worker_idx}]}"
  offset=$((worker_idx * chunk_size))
  if [[ "${offset}" -ge "${total_items}" ]]; then
    echo "[multi-gpu] worker=${worker_idx} gpu=${gpu_id:-current} skipped empty shard offset=${offset}"
    continue
  fi
  limit="${chunk_size}"
  summary_file="${log_dir}/worker_${worker_idx}_summary.json"
  log_file="${log_dir}/worker_${worker_idx}.log"
  command=(
    "${PYTHON_BIN}" scripts/run_vlm_batch_images.py
    --model "${MODEL_PATH}"
    --prompt-dir "${PROMPT_DIR}"
    --exclude-prompt-names prompt-classification-boundary-type
    --image-dir "${IMAGE_DIR}"
    --output-dir "${OUTPUT_DIR}"
    --device-map auto
  )
  if ! has_arg "--device" "${REMAINING_ARGS[@]}"; then
    command+=(--device cuda:0)
  fi
  command+=("${REMAINING_ARGS[@]}" --summary-file "${summary_file}" --offset "${offset}" --limit "${limit}")

  echo "[multi-gpu] worker=${worker_idx} gpu=${gpu_id:-current} offset=${offset} limit=${limit} log=${log_file}"
  if [[ -n "${gpu_id}" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${command[@]}" >"${log_file}" 2>&1 &
  else
    "${command[@]}" >"${log_file}" 2>&1 &
  fi
  pids+=("$!")
done

failed=0
for worker_idx in "${!pids[@]}"; do
  if ! wait "${pids[${worker_idx}]}"; then
    echo "[multi-gpu] worker=${worker_idx} failed; tail ${log_dir}/worker_${worker_idx}.log" >&2
    tail -n 80 "${log_dir}/worker_${worker_idx}.log" >&2 || true
    failed=1
  fi
done

if [[ "${failed}" -ne 0 ]]; then
  echo "[multi-gpu] failed; logs=${log_dir}" >&2
  exit 2
fi

echo "[multi-gpu] done logs=${log_dir}"

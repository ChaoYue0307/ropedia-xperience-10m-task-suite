#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
STAGING_ROOT="${STAGING_ROOT:-/mnt/kgc/chaoyue/ropedia-h20-side}"
VENV_PY="${VENV_PY:-$PROJECT_ROOT/.venv/bin/python}"

RUN_ID="${RUN_ID:-a100_repro_qwen_v6_eval_smoke1_$(date +%Y%m%d_%H%M%S)}"
DATASET_JSONL="${DATASET_JSONL:-$PROJECT_ROOT/results/omni_finetune/xperience10m_qwen3_omni_128ep_multiscale_cap96_v5_full8gpu_lora_dataset/dataset_a100_eval.jsonl}"
MODEL_DIR="${MODEL_DIR:-$STAGING_ROOT/modelscope_models/Qwen__Qwen3-Omni-30B-A3B-Instruct}"
ADAPTER_DIR="${ADAPTER_DIR:-$PROJECT_ROOT/checkpoints/xperience10m_qwen3_omni_128ep_multiscale_cap96_v6_rank64_lr5e5_full8gpu_lora/adapter_lora}"
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/results/omni_finetune/$RUN_ID}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
DTYPE="${DTYPE:-bfloat16}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
MIN_FREE_MIB="${MIN_FREE_MIB:-60000}"
GPU_WAIT_SECONDS="${GPU_WAIT_SECONDS:-0}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-30}"

for path in "$VENV_PY" "$DATASET_JSONL" "$MODEL_DIR" "$ADAPTER_DIR"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required private GPU reproducibility path: $path" >&2
    exit 2
  fi
done

cd "$PROJECT_ROOT"
mkdir -p "$OUT_DIR"

check_gpu_free() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi

  local bad=0
  local gpu line free_mib total_mib
  IFS=',' read -r -a selected_gpus <<<"$CUDA_VISIBLE_DEVICES"
  for gpu in "${selected_gpus[@]}"; do
    gpu="${gpu//[[:space:]]/}"
    if [[ -z "$gpu" || ! "$gpu" =~ ^[0-9]+$ ]]; then
      continue
    fi
    line="$(nvidia-smi --id="$gpu" --query-gpu=memory.free,memory.total --format=csv,noheader,nounits 2>/dev/null || true)"
    free_mib="${line%%,*}"
    total_mib="${line##*,}"
    free_mib="${free_mib//[[:space:]]/}"
    total_mib="${total_mib//[[:space:]]/}"
    printf 'gpu=%s free_mib=%s total_mib=%s\n' "$gpu" "$free_mib" "$total_mib"
    if [[ -z "$free_mib" || "$free_mib" -lt "$MIN_FREE_MIB" ]]; then
      bad=1
    fi
  done
  return "$bad"
}

deadline=$((SECONDS + GPU_WAIT_SECONDS))
while true; do
  if gpu_status="$(check_gpu_free)"; then
    printf '%s\n' "$gpu_status" >"$OUT_DIR/gpu_preflight.txt"
    break
  fi
  printf '%s\n' "$gpu_status" >"$OUT_DIR/gpu_preflight.txt"
  if (( SECONDS >= deadline )); then
    {
      echo "Not enough free GPU memory for Qwen3-Omni smoke."
      echo "required_free_mib=$MIN_FREE_MIB"
      echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
      printf '%s\n' "$gpu_status"
    } >&2
    exit 3
  fi
  sleep "$GPU_POLL_SECONDS"
done

"$VENV_PY" scripts/omni/patch_qwen3_omni_video_features.py --apply --strict-hash

{
  echo "run_id=$RUN_ID"
  echo "project_root=$PROJECT_ROOT"
  echo "dataset_jsonl=$DATASET_JSONL"
  echo "model_dir=$MODEL_DIR"
  echo "adapter_dir=$ADAPTER_DIR"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "sample_limit=$SAMPLE_LIMIT"
  echo "max_new_tokens=$MAX_NEW_TOKENS"
  echo "min_free_mib=$MIN_FREE_MIB"
  echo "gpu_wait_seconds=$GPU_WAIT_SECONDS"
  echo "started_at=$(date -Is)"
} >"$OUT_DIR/a100_repro_env.txt"

set +e
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$VENV_PY" scripts/omni/eval_qwen3_omni_lora.py \
  --dataset-jsonl "$DATASET_JSONL" \
  --model-id "$MODEL_DIR" \
  --adapter-dir "$ADAPTER_DIR" \
  --run-id "$RUN_ID" \
  --eval-split "$EVAL_SPLIT" \
  --sample-limit "$SAMPLE_LIMIT" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --device-map "$DEVICE_MAP" \
  --dtype "$DTYPE" \
  --local-files-only \
  >"$OUT_DIR/a100_smoke.log" 2>&1
status=$?
set -e

echo "$status" >"$OUT_DIR/exit_code.txt"
echo "finished_at=$(date -Is)" >>"$OUT_DIR/a100_repro_env.txt"
echo "exit_code=$status" >>"$OUT_DIR/a100_repro_env.txt"

if [[ "$status" -ne 0 ]]; then
  tr '\r' '\n' <"$OUT_DIR/a100_smoke.log" 2>/dev/null | tail -n 120 >&2 || true
  exit "$status"
fi

echo "Private GPU Qwen3 v6 reproducibility smoke passed: $OUT_DIR"

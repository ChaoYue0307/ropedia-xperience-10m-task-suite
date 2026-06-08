#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

VENV_PY="${VENV_PY:-$ROOT_DIR/.venv/bin/python}"
DATASET_JSONL="${DATASET_JSONL:-results/omni_finetune/xperience10m_qwen3_omni_128ep_96train_16val_16test_valmon_20260605_dataset/dataset.jsonl}"
MODEL_DIR="${MODEL_DIR:-/home/cy/Ropedia/modelscope_models/Qwen__Qwen3-Omni-30B-A3B-Instruct}"
ADAPTER_DIR="${ADAPTER_DIR:-checkpoints/xperience10m_qwen3_omni_128ep_structured_json_v2_reuse_full8gpu_lora/adapter_lora}"
RUN_ID="${RUN_ID:-xperience10m_qwen3_omni_128ep_structured_json_v3_strict_label_prompt_reuse_lora_eval_test_sharded}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-0}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
DTYPE="${DTYPE:-bfloat16}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
CUDA_DEVICE_GROUPS="${CUDA_DEVICE_GROUPS:-0,1 2,3 4,5 6,7}"

read -r -a GPU_GROUP_ARRAY <<< "$CUDA_DEVICE_GROUPS"
SHARDS="${SHARDS:-${#GPU_GROUP_ARRAY[@]}}"
if [[ "$SHARDS" -lt 1 ]]; then
  echo "SHARDS must be >= 1" >&2
  exit 2
fi
if [[ "${#GPU_GROUP_ARRAY[@]}" -lt "$SHARDS" ]]; then
  echo "CUDA_DEVICE_GROUPS must provide at least SHARDS groups." >&2
  exit 2
fi

COMMON_ARGS=(
  --dataset-jsonl "$DATASET_JSONL"
  --model-id "$MODEL_DIR"
  --adapter-dir "$ADAPTER_DIR"
  --eval-split "$EVAL_SPLIT"
  --sample-limit "$SAMPLE_LIMIT"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --device-map "$DEVICE_MAP"
  --dtype "$DTYPE"
)
if [[ "$LOCAL_FILES_ONLY" == "1" ]]; then
  COMMON_ARGS+=(--local-files-only)
fi

declare -a PIDS=()
declare -a SHARD_DIRS=()

for ((offset = 0; offset < SHARDS; offset++)); do
  shard_run_id="${RUN_ID}_shard${offset}"
  shard_dir="results/omni_finetune/${shard_run_id}"
  mkdir -p "$shard_dir"
  SHARD_DIRS+=("$shard_dir")
  (
    export CUDA_VISIBLE_DEVICES="${GPU_GROUP_ARRAY[$offset]}"
    "$VENV_PY" scripts/omni/eval_qwen3_omni_lora.py \
      "${COMMON_ARGS[@]}" \
      --run-id "$shard_run_id" \
      --sample-offset "$offset" \
      --sample-stride "$SHARDS"
  ) >"$shard_dir/eval.log" 2>&1 &
  pid="$!"
  PIDS+=("$pid")
  echo "$pid" >"$shard_dir/eval.pid"
  echo "launched shard $offset/$SHARDS on CUDA_VISIBLE_DEVICES=${GPU_GROUP_ARRAY[$offset]} pid=$pid"
done

failed=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" != "0" ]]; then
  echo "At least one eval shard failed; inspect ${SHARD_DIRS[*]}" >&2
  exit 1
fi

MERGE_ARGS=(
  --dataset-jsonl "$DATASET_JSONL"
  --output-dir "results/omni_finetune/$RUN_ID"
  --run-id "$RUN_ID"
  --eval-split "$EVAL_SPLIT"
  --model-id "$MODEL_DIR"
  --adapter-dir "$ADAPTER_DIR"
)
for shard_dir in "${SHARD_DIRS[@]}"; do
  MERGE_ARGS+=(--shard-dir "$shard_dir")
done

"$VENV_PY" scripts/omni/merge_qwen3_omni_eval_shards.py "${MERGE_ARGS[@]}"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

VENV_PY="${VENV_PY:-$ROOT_DIR/.venv/bin/python}"
STAGING_ROOT="${STAGING_ROOT:-/mnt/kgc/chaoyue/ropedia-h20-side}"
DATASET_JSONL="${DATASET_JSONL:-$ROOT_DIR/results/omni_finetune/xperience10m_qwen3_omni_128ep_multiscale_cap96_v5_full8gpu_lora_dataset/dataset_a100_eval.jsonl}"
CAPTION_DIR="${CAPTION_DIR:-$ROOT_DIR/results/omni_finetune/xperience10m_128_raw_caption_interactions_task15_20260619_full}"
CAPTION_JSONL="${CAPTION_JSONL:-$CAPTION_DIR/caption_interactions.jsonl}"
CAPTION_MANIFEST="${CAPTION_MANIFEST:-$CAPTION_DIR/caption_interactions_manifest.json}"
MODEL_DIR="${MODEL_DIR:-$STAGING_ROOT/modelscope_models/Qwen__Qwen3-Omni-30B-A3B-Instruct}"
ADAPTER_DIR="${ADAPTER_DIR:-$ROOT_DIR/checkpoints/xperience10m_qwen3_omni_128ep_multiscale_cap96_v6_rank64_lr5e5_full8gpu_lora/adapter_lora}"
RUN_ID="${RUN_ID:-xperience10m_qwen3_omni_v6_interaction_text_task15_a100_20260619T000000Z}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
CANDIDATE_COUNT="${CANDIDATE_COUNT:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-0}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
DTYPE="${DTYPE:-bfloat16}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
CUDA_DEVICE_GROUPS="${CUDA_DEVICE_GROUPS:-0,1 2,3}"

for path in "$VENV_PY" "$DATASET_JSONL" "$CAPTION_JSONL" "$CAPTION_MANIFEST" "$MODEL_DIR" "$ADAPTER_DIR"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required path: $path" >&2
    exit 2
  fi
done

"$VENV_PY" - "$CAPTION_MANIFEST" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
manifest = json.loads(path.read_text())
if manifest.get("status") != "pass":
    raise SystemExit(
        f"Caption manifest is not pass: status={manifest.get('status')} "
        f"processed={manifest.get('processed_file_count')}/{manifest.get('requested_file_count')}"
    )
PY

"$VENV_PY" scripts/omni/patch_qwen3_omni_video_features.py --apply --strict-hash

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

OUT_DIR="results/omni_finetune/$RUN_ID"
mkdir -p "$OUT_DIR"
{
  echo "run_id=$RUN_ID"
  echo "dataset_jsonl=$DATASET_JSONL"
  echo "caption_jsonl=$CAPTION_JSONL"
  echo "caption_manifest=$CAPTION_MANIFEST"
  echo "model_dir=$MODEL_DIR"
  echo "adapter_dir=$ADAPTER_DIR"
  echo "candidate_count=$CANDIDATE_COUNT"
  echo "cuda_device_groups=$CUDA_DEVICE_GROUPS"
  echo "shards=$SHARDS"
  echo "started_at=$(date -Is)"
} >"$OUT_DIR/launch_env.txt"

COMMON_ARGS=(
  --dataset-jsonl "$DATASET_JSONL"
  --caption-jsonl "$CAPTION_JSONL"
  --caption-manifest "$CAPTION_MANIFEST"
  --model-id "$MODEL_DIR"
  --adapter-dir "$ADAPTER_DIR"
  --eval-split "$EVAL_SPLIT"
  --candidate-count "$CANDIDATE_COUNT"
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
    "$VENV_PY" scripts/omni/eval_qwen3_omni_interaction_text_task.py \
      "${COMMON_ARGS[@]}" \
      --run-id "$shard_run_id" \
      --output-dir "$shard_dir" \
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
  echo "At least one task-15 probe shard failed; inspect ${SHARD_DIRS[*]}" >&2
  echo "exit_code=1" >>"$OUT_DIR/launch_env.txt"
  exit 1
fi

"$VENV_PY" scripts/omni/merge_qwen3_omni_interaction_text_task_shards.py \
  --run-id "$RUN_ID" \
  --output-dir "$OUT_DIR" \
  --dataset-jsonl "$DATASET_JSONL" \
  --caption-jsonl "$CAPTION_JSONL" \
  --caption-manifest "$CAPTION_MANIFEST" \
  --model-id "$MODEL_DIR" \
  --adapter-dir "$ADAPTER_DIR" \
  --eval-split "$EVAL_SPLIT" \
  --candidate-count "$CANDIDATE_COUNT" \
  --shard-dir "${SHARD_DIRS[@]}"

echo "finished_at=$(date -Is)" >>"$OUT_DIR/launch_env.txt"
echo "exit_code=0" >>"$OUT_DIR/launch_env.txt"
echo "Qwen3 task-15 interaction text probe complete: $OUT_DIR"

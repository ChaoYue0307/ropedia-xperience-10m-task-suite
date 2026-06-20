#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

VENV_PY="${VENV_PY:-$ROOT_DIR/.venv/bin/python}"
DATASET_JSONL="${DATASET_JSONL:-results/omni_finetune/xperience10m_qwen3_omni_128ep_multiscale_cap96_v5_full8gpu_lora_dataset/dataset_a100_eval.jsonl}"
CAPTION_DIR="${CAPTION_DIR:-results/omni_finetune/xperience10m_128_raw_caption_interactions_task15_20260619_full}"
CAPTION_JSONL="${CAPTION_JSONL:-$CAPTION_DIR/caption_interactions.jsonl}"
CAPTION_MANIFEST="${CAPTION_MANIFEST:-$CAPTION_DIR/caption_interactions_manifest.json}"
RUN_ID="${RUN_ID:-xperience10m_cosmos3_super_interaction_text_task15_textonly_v1_$(date -u +%Y%m%dT%H%M%SZ)}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
MODEL="${MODEL:-/mnt/kgc/chaoyue/ropedia-xperience10m/models/nvidia__Cosmos3-Super_reasoner_overlay}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
CANDIDATE_COUNT="${CANDIDATE_COUNT:-4}"
MAX_TOKENS="${MAX_TOKENS:-64}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-900}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-0}"
SHARDS="${SHARDS:-4}"

for path in "$VENV_PY" "$DATASET_JSONL" "$CAPTION_JSONL" "$CAPTION_MANIFEST"; do
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

OUT_DIR="results/omni_finetune/$RUN_ID"
mkdir -p "$OUT_DIR"
{
  echo "run_id=$RUN_ID"
  echo "dataset_jsonl=$DATASET_JSONL"
  echo "caption_jsonl=$CAPTION_JSONL"
  echo "caption_manifest=$CAPTION_MANIFEST"
  echo "base_url=$BASE_URL"
  echo "model=$MODEL"
  echo "candidate_count=$CANDIDATE_COUNT"
  echo "shards=$SHARDS"
  echo "media_mode=text_only"
  echo "started_at=$(date -Is)"
} >"$OUT_DIR/launch_env.txt"

declare -a PIDS=()
declare -a SHARD_DIRS=()

for ((shard = 0; shard < SHARDS; shard++)); do
  shard_id="${RUN_ID}_shard${shard}"
  shard_dir="results/omni_finetune/${shard_id}"
  mkdir -p "$shard_dir"
  SHARD_DIRS+=("$shard_dir")
  "$VENV_PY" scripts/omni/eval_cosmos3_super_interaction_text_task.py \
    --dataset-jsonl "$DATASET_JSONL" \
    --caption-jsonl "$CAPTION_JSONL" \
    --caption-manifest "$CAPTION_MANIFEST" \
    --run-id "$shard_id" \
    --output-dir "$shard_dir" \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --eval-split "$EVAL_SPLIT" \
    --candidate-count "$CANDIDATE_COUNT" \
    --sample-limit "$SAMPLE_LIMIT" \
    --max-tokens "$MAX_TOKENS" \
    --request-timeout "$REQUEST_TIMEOUT" \
    --sample-offset "$shard" \
    --sample-stride "$SHARDS" >"$shard_dir/eval.log" 2>&1 &
  pid="$!"
  PIDS+=("$pid")
  echo "$pid" >"$shard_dir/eval.pid"
  echo "launched shard $shard/$SHARDS pid=$pid"
done

failed=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" != "0" ]]; then
  echo "At least one Cosmos3-Super task-15 shard failed; inspect ${SHARD_DIRS[*]}" >&2
  echo "exit_code=1" >>"$OUT_DIR/launch_env.txt"
  exit 1
fi

"$VENV_PY" scripts/omni/merge_cosmos3_super_interaction_text_task_shards.py \
  --run-id "$RUN_ID" \
  --output-dir "$OUT_DIR" \
  --dataset-jsonl "$DATASET_JSONL" \
  --caption-jsonl "$CAPTION_JSONL" \
  --caption-manifest "$CAPTION_MANIFEST" \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --eval-split "$EVAL_SPLIT" \
  --candidate-count "$CANDIDATE_COUNT" \
  --shard-dir "${SHARD_DIRS[@]}"

echo "finished_at=$(date -Is)" >>"$OUT_DIR/launch_env.txt"
echo "exit_code=0" >>"$OUT_DIR/launch_env.txt"
echo "Cosmos3-Super task-15 interaction-text probe complete: $OUT_DIR"

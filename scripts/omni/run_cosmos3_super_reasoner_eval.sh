#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

VENV_PY="${VENV_PY:-$ROOT_DIR/.venv/bin/python}"
DATASET_JSONL="${DATASET_JSONL:-results/omni_finetune/xperience10m_qwen3_omni_128ep_96train_16val_16test_valmon_20260605_dataset/dataset.jsonl}"
RUN_ID="${RUN_ID:-xperience10m_cosmos3_super_reasoner_128ep_test_full_20260607}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
MODEL="${MODEL:-cosmos3-super-local}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-0}"
CONCURRENCY="${CONCURRENCY:-1}"
MEDIA_MODE="${MEDIA_MODE:-video_url}"
MAX_TOKENS="${MAX_TOKENS:-96}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-900}"

"$VENV_PY" scripts/omni/eval_cosmos3_super_reasoner.py \
  --dataset-jsonl "$DATASET_JSONL" \
  --run-id "$RUN_ID" \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --sample-limit "$SAMPLE_LIMIT" \
  --concurrency "$CONCURRENCY" \
  --media-mode "$MEDIA_MODE" \
  --max-tokens "$MAX_TOKENS" \
  --request-timeout "$REQUEST_TIMEOUT"

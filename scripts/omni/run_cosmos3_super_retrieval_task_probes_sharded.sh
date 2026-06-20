#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

VENV_PY="${VENV_PY:-$ROOT_DIR/.venv/bin/python}"
DATASET_JSONL="${DATASET_JSONL:-results/omni_finetune/xperience10m_qwen3_omni_128ep_multiscale_cap96_v5_full8gpu_lora_dataset/dataset_a100_eval.jsonl}"
RUN_ID="${RUN_ID:-xperience10m_cosmos3_super_retrieval_task_probes_a100_textonly_prompatch_v2_20260620}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
MODEL="${MODEL:-/mnt/kgc/chaoyue/ropedia-xperience10m/models/nvidia__Cosmos3-Super_reasoner_overlay}"
TASKS="${TASKS:-hand_trajectory_forecast,cross_modal_retrieval,modality_reconstruction,imu_to_hand_pose,camera_view_sync_retrieval}"
CANDIDATE_COUNT="${CANDIDATE_COUNT:-4}"
FUTURE_FRAMES="${FUTURE_FRAMES:-100}"
MAX_TOKENS="${MAX_TOKENS:-96}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-900}"
MEDIA_MODE="${MEDIA_MODE:-video_url}"
SHARDS="${SHARDS:-2}"

MERGE_SCRIPT="scripts/omni/merge_qwen3_omni_retrieval_task_probe_shards.py"
OUT_DIR="results/omni_finetune/${RUN_ID}"
mkdir -p "$OUT_DIR"

for (( shard=0; shard<SHARDS; shard++ )); do
  shard_id="${RUN_ID}_shard${shard}"
  shard_dir="results/omni_finetune/${shard_id}"
  "$VENV_PY" scripts/omni/eval_cosmos3_super_retrieval_task_probes.py \
    --dataset-jsonl "$DATASET_JSONL" \
    --run-id "$shard_id" \
    --output-dir "$shard_dir" \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --tasks "$TASKS" \
    --candidate-count "$CANDIDATE_COUNT" \
    --future-frames "$FUTURE_FRAMES" \
    --max-tokens "$MAX_TOKENS" \
    --request-timeout "$REQUEST_TIMEOUT" \
    --media-mode "$MEDIA_MODE" \
    --sample-offset "$shard" \
    --sample-stride "$SHARDS" &
done

wait

"$VENV_PY" "$MERGE_SCRIPT" \
  --run-id "$RUN_ID" \
  --output-dir "$OUT_DIR" \
  --shard-dir $(for (( shard=0; shard<SHARDS; shard++ )); do printf ' results/omni_finetune/%s_shard%d' "$RUN_ID" "$shard"; done)

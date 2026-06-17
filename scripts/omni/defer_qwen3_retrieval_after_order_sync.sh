#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

RETRIEVAL_RUN_ID="${RETRIEVAL_RUN_ID:-xperience10m_qwen3_omni_v6_retrieval_task_probes_a100_20260617T175919Z}"
WAITER_RUN_ID="${WAITER_RUN_ID:-${RETRIEVAL_RUN_ID}_waiter}"
CUDA_DEVICE_GROUPS_VALUE="${CUDA_DEVICE_GROUPS_VALUE:-0,1 2,3}"

SCORING_COMMAND=$(
  printf '%s ' \
    "RUN_ID=${RETRIEVAL_RUN_ID}" \
    "TASKS=caption_grounding" \
    "CANDIDATE_COUNT=4" \
    "MAX_NEW_TOKENS=64" \
    "CUDA_DEVICE_GROUPS=\"${CUDA_DEVICE_GROUPS_VALUE}\"" \
    "SHARDS=2" \
    "LOCAL_FILES_ONLY=1" \
    "scripts/omni/run_qwen3_omni_retrieval_task_probes_sharded.sh"
)

RUN_ID="$WAITER_RUN_ID" \
MIN_FREE_GPU_COUNT="${MIN_FREE_GPU_COUNT:-4}" \
MIN_FREE_MB_PER_GPU="${MIN_FREE_MB_PER_GPU:-60000}" \
MAX_BUSY_UTIL="${MAX_BUSY_UTIL:-10}" \
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-30}" \
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-43200}" \
SCORING_COMMAND="$SCORING_COMMAND" \
bash scripts/omni/launch_all_task_model_scoring_when_free.sh

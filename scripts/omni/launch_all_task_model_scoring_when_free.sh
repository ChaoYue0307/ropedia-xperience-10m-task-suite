#!/usr/bin/env bash
set -euo pipefail

# Guarded waiter for private GPU scoring jobs.
# Provide SCORING_COMMAND with the exact command to run once enough GPUs are idle.

WORKSPACE="${WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RUN_ID="${RUN_ID:-xperience10m_all_task_model_scoring_$(date -u +%Y%m%dT%H%M%SZ)}"
RESULTS_DIR="${RESULTS_DIR:-${WORKSPACE}/results/omni_finetune/deferred_launchers}"
MIN_FREE_GPU_COUNT="${MIN_FREE_GPU_COUNT:-4}"
MIN_FREE_MB_PER_GPU="${MIN_FREE_MB_PER_GPU:-60000}"
MAX_BUSY_UTIL="${MAX_BUSY_UTIL:-10}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-60}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-21600}"
SCORING_COMMAND="${SCORING_COMMAND:-}"

mkdir -p "${RESULTS_DIR}"
WAITER_LOG="${RESULTS_DIR}/${RUN_ID}.waiter.log"
STATUS_JSONL="${RESULTS_DIR}/${RUN_ID}.status.jsonl"
RUN_LOG="${RESULTS_DIR}/${RUN_ID}.run.log"

log() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${WAITER_LOG}"
}

write_status() {
  local status="$1"
  local detail="$2"
  python3 - "$STATUS_JSONL" "$RUN_ID" "$status" "$detail" <<'PY'
import json
import sys
from datetime import datetime, timezone

path, run_id, status, detail = sys.argv[1:5]
record = {
    "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "run_id": run_id,
    "status": status,
    "detail": detail,
}
with open(path, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(record, sort_keys=True) + "\n")
PY
}

gpu_snapshot() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 1
  fi
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
}

free_gpu_count() {
  gpu_snapshot | awk -F',' -v min_free="${MIN_FREE_MB_PER_GPU}" -v max_util="${MAX_BUSY_UTIL}" '
    {
      used=$2+0; total=$3+0; util=$4+0;
      free=total-used;
      if (free >= min_free && util <= max_util) count += 1;
    }
    END { print count+0 }
  '
}

if [[ -z "${SCORING_COMMAND}" ]]; then
  log "SCORING_COMMAND is required; exiting without launching."
  write_status "not_launched" "SCORING_COMMAND is required"
  exit 2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  log "nvidia-smi is unavailable; exiting without launching."
  write_status "not_launched" "nvidia-smi unavailable"
  exit 3
fi

start_epoch="$(date +%s)"
log "waiting for ${MIN_FREE_GPU_COUNT} GPUs with at least ${MIN_FREE_MB_PER_GPU} MB free and util <= ${MAX_BUSY_UTIL}%"
write_status "waiting" "guard started"

while true; do
  now_epoch="$(date +%s)"
  elapsed=$((now_epoch - start_epoch))
  snapshot="$(gpu_snapshot || true)"
  count="$(printf '%s\n' "${snapshot}" | awk -F',' -v min_free="${MIN_FREE_MB_PER_GPU}" -v max_util="${MAX_BUSY_UTIL}" '
    {
      used=$2+0; total=$3+0; util=$4+0;
      free=total-used;
      if (free >= min_free && util <= max_util) count += 1;
    }
    END { print count+0 }
  ')"
  log "free_gpu_count=${count}; elapsed_seconds=${elapsed}"
  printf '%s\n' "${snapshot}" >> "${WAITER_LOG}"

  if (( count >= MIN_FREE_GPU_COUNT )); then
    log "capacity available; launching scoring command"
    write_status "launching" "capacity available"
    nohup bash -lc "cd '${WORKSPACE}' && ${SCORING_COMMAND}" > "${RUN_LOG}" 2>&1 &
    pid="$!"
    log "launched pid=${pid}; run_log=${RUN_LOG}"
    write_status "launched" "pid=${pid}"
    exit 0
  fi

  if (( elapsed >= MAX_WAIT_SECONDS )); then
    log "timed out without launching"
    write_status "timed_out" "capacity not available"
    exit 4
  fi

  sleep "${CHECK_INTERVAL_SECONDS}"
done

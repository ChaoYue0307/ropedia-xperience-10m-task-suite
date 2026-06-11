#!/usr/bin/env bash
set -euo pipefail

# Wait for a verified Qwen package, then run a bounded full-parameter
# feasibility pilot on the remote 8-GPU worker. This intentionally saves no
# model weights or checkpoints.

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT"

QWEN_SUMMARY="${QWEN_SUMMARY:-results/omni_finetune/verified_public/xperience10m_qwen3_omni_128ep_multiscale_cap96_v5_full8gpu_lora_eval_test_full/verified_result_summary.json}"
POLL_SECONDS="${POLL_SECONDS:-120}"
RUN_ID="${RUN_ID:-xperience10m_qwen3_omni_128ep_fullparam_pilot128_after_qwen_v5_preemptible_8gpu_20260609}"
DATASET_JSONL="${DATASET_JSONL:-results/omni_finetune/xperience10m_qwen3_omni_128ep_multiscale_cap96_v5_full8gpu_lora_dataset/dataset.jsonl}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-1024}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-128}"
EPOCHS="${EPOCHS:-16}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-14400}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"

qwen_train_or_eval_running() {
  pgrep -af '[t]rain_qwen3_omni_lora.py|[e]val_qwen3_omni_lora.py' >/dev/null
}

echo "$(date) waiting for verified Qwen package: $QWEN_SUMMARY"
while true; do
  if [[ -f "$QWEN_SUMMARY" ]] && grep -Eq '"status"[[:space:]]*:[[:space:]]*"verified"' "$QWEN_SUMMARY"; then
    echo "$(date) verified Qwen package is ready"
    break
  fi
  sleep "$POLL_SECONDS"
done

while qwen_train_or_eval_running; do
  echo "$(date) Qwen train/eval process still running; waiting"
  sleep "$POLL_SECONDS"
done

if [[ ! -s "$DATASET_JSONL" ]]; then
  echo "$(date) blocked: missing dataset JSONL: $DATASET_JSONL" >&2
  exit 2
fi

echo "$(date) starting full-parameter Qwen pilot: $RUN_ID"
export RUN_ID DATASET_JSONL MAX_TRAIN_SAMPLES MAX_TRAIN_STEPS EPOCHS TIMEOUT_SECONDS NUM_PROCESSES
exec scripts/omni/run_qwen3_omni_fullparam_smoke_8gpu.sh

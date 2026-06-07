#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT"

QWEN_SUMMARY="${QWEN_SUMMARY:-results/omni_finetune/verified_public/xperience10m_qwen3_omni_128ep_structured_json_v4_4epoch_full8gpu_lora_eval_test_full/verified_result_summary.json}"
POLL_SECONDS="${POLL_SECONDS:-120}"

RUN_ID="${RUN_ID:-xperience10m_cosmos3_super_forward_dynamics_lora_overfit_after_qwen_v4_20260608}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-1}"
MAX_STEPS="${MAX_STEPS:-3}"
OVERRIDE_RESOLUTION_TIER="${OVERRIDE_RESOLUTION_TIER:-256}"
DEVICE_MAP="${DEVICE_MAP:-balanced}"
DTYPE="${DTYPE:-bfloat16}"
TIMESTEP_SAMPLING="${TIMESTEP_SAMPLING:-uniform}"

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

echo "$(date) starting Cosmos3-Super forward-dynamics LoRA overfit: $RUN_ID"
export RUN_ID MAX_TRAIN_SAMPLES MAX_STEPS OVERRIDE_RESOLUTION_TIER DEVICE_MAP DTYPE TIMESTEP_SAMPLING
exec scripts/omni/run_cosmos3_super_forward_dynamics_lora.sh

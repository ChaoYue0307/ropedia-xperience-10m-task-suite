#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/cy/Ropedia/ropedia-episode-task-suite}"
BASE_RUN_ID="${BASE_RUN_ID:-xperience10m_qwen3_omni_128ep_fullsplit_fast8gpu}"
SMOKE_RUN_ID="${SMOKE_RUN_ID:-xperience10m_qwen3_omni_128ep_fullsplit_fast8gpu_lora_fsdp_smoke_v3}"
SMOKE_EXIT_EVENT="${SMOKE_EXIT_EVENT:-train_exit_fsdp_smoke_v3_no_val}"
FULL_RUN_ID="${FULL_RUN_ID:-xperience10m_qwen3_omni_128ep_fullsplit_fast8gpu_lora_fsdp_full_train_noval}"
MODEL_ID="${MODEL_ID:-/home/cy/Ropedia/modelscope_models/Qwen__Qwen3-Omni-30B-A3B-Instruct}"
BACKBONE_CONFIG="${BACKBONE_CONFIG:-configs/omni_backbones/qwen3_omni_lora.json}"

cd "$PROJECT_DIR"

RUN_DIR="results/omni_finetune/${BASE_RUN_ID}"
STATUS="${RUN_DIR}/status.jsonl"
DATASET_JSONL="results/omni_finetune/${BASE_RUN_ID}_dataset/dataset.jsonl"
FULL_LOG="${RUN_DIR}/train_fsdp_full_train_noval.log"

mkdir -p "$RUN_DIR"

is_full_training_active() {
  pgrep -af "scripts/omni/train_qwen3_omni_lora.py.*${FULL_RUN_ID}" >/dev/null 2>&1
}

is_smoke_active() {
  pgrep -af "scripts/omni/train_qwen3_omni_lora.py.*${SMOKE_RUN_ID}" >/dev/null 2>&1
}

if is_full_training_active; then
  echo "full run already active"
  exit 0
fi

while true; do
  if grep -q "${SMOKE_EXIT_EVENT}.*returncode\":0" "$STATUS"; then
    break
  fi
  if grep -q "${SMOKE_EXIT_EVENT}.*returncode\":[1-9]" "$STATUS"; then
    echo "{\"event\":\"train_full_blocked_smoke_failed\",\"time\":$(date +%s),\"smoke_run_id\":\"${SMOKE_RUN_ID}\"}" >> "$STATUS"
    exit 1
  fi
  if ! is_smoke_active; then
    echo "{\"event\":\"train_full_blocked_smoke_missing_exit\",\"time\":$(date +%s),\"smoke_run_id\":\"${SMOKE_RUN_ID}\"}" >> "$STATUS"
    exit 2
  fi
  sleep 30
done

if is_full_training_active; then
  echo "full run already active after smoke"
  exit 0
fi

echo "{\"event\":\"train_start_fsdp_full_train_noval\",\"time\":$(date +%s),\"run_id\":\"${FULL_RUN_ID}\",\"train_split\":\"train\",\"val_split_reserved\":\"val\",\"test_split_reserved\":\"test\",\"num_processes\":8}" >> "$STATUS"

set +e
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python -m accelerate.commands.launch \
  --num_processes 8 \
  --mixed_precision bf16 \
  --use_fsdp \
  --fsdp_sharding_strategy FULL_SHARD \
  --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
  --fsdp_transformer_layer_cls_to_wrap Qwen3OmniMoeThinkerTextDecoderLayer \
  --fsdp_use_orig_params true \
  scripts/omni/train_qwen3_omni_lora.py \
  --dataset-jsonl "$DATASET_JSONL" \
  --model-id "$MODEL_ID" \
  --backbone-config "$BACKBONE_CONFIG" \
  --run-id "$FULL_RUN_ID" \
  --train-split train \
  --val-split __none__ \
  --epochs 1 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --max-train-samples 0 \
  --max-val-samples 0 \
  --local-files-only \
  --gradient-checkpointing \
  --progress-every 10 \
  > "$FULL_LOG" 2>&1
rc=$?
set -e

echo "{\"event\":\"train_exit_fsdp_full_train_noval\",\"time\":$(date +%s),\"run_id\":\"${FULL_RUN_ID}\",\"returncode\":${rc}}" >> "$STATUS"
exit "$rc"

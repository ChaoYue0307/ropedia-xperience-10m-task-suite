#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
VENV_PY="${VENV_PY:-$PROJECT_ROOT/.venv/bin/python}"

RUN_ID="${RUN_ID:-xperience10m_cosmos3_super_forward_dynamics_lora_overfit}"
DATASET_JSONL="${DATASET_JSONL:-$PROJECT_ROOT/results/omni_finetune/xperience10m_cosmos3_camera_pose_targets_20260608/dataset_with_cosmos_actions.jsonl}"
MODEL_DIR="${MODEL_DIR:-$HOME/Ropedia/cosmos3_models/nv-community__Cosmos3-Super}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/results/omni_finetune/$RUN_ID}"
SPLIT="${SPLIT:-train}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-1}"
MAX_STEPS="${MAX_STEPS:-10}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
DEVICE_MAP="${DEVICE_MAP:-balanced}"
DTYPE="${DTYPE:-bfloat16}"
SEED="${SEED:-123}"
TARGET_MODULES="${TARGET_MODULES:-}"
TIMESTEP_SAMPLING="${TIMESTEP_SAMPLING:-uniform}"
OVERRIDE_RESOLUTION_TIER="${OVERRIDE_RESOLUTION_TIER:-}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
FSDP_TRANSFORMER_LAYER="${FSDP_TRANSFORMER_LAYER:-Cosmos3VLTextMoTDecoderLayer}"
FSDP_ACTIVATION_CHECKPOINTING="${FSDP_ACTIVATION_CHECKPOINTING:-true}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "$NUM_PROCESSES" != "1" && "$DEVICE_MAP" != "none" ]]; then
  DEVICE_MAP="none"
fi

train_args=(
  "$PROJECT_ROOT/scripts/omni/train_cosmos3_super_forward_dynamics_lora.py"
  --workspace "$PROJECT_ROOT"
  --dataset-jsonl "$DATASET_JSONL"
  --model-dir "$MODEL_DIR"
  --run-id "$RUN_ID"
  --output-dir "$OUTPUT_DIR"
  --split "$SPLIT"
  --max-train-samples "$MAX_TRAIN_SAMPLES"
  --max-steps "$MAX_STEPS"
  --learning-rate "$LEARNING_RATE"
  --device-map "$DEVICE_MAP"
  --dtype "$DTYPE"
  --seed "$SEED"
  --timestep-sampling "$TIMESTEP_SAMPLING"
)

if [[ -n "$OVERRIDE_RESOLUTION_TIER" ]]; then
  train_args+=(--override-resolution-tier "$OVERRIDE_RESOLUTION_TIER")
fi

if [[ -n "$TARGET_MODULES" ]]; then
  train_args+=(--target-modules "$TARGET_MODULES")
fi

if [[ "$GRADIENT_CHECKPOINTING" == "0" ]]; then
  train_args+=(--no-gradient-checkpointing)
fi

if [[ "$DRY_RUN" == "1" ]]; then
  train_args+=(--dry-run)
fi

if [[ "$NUM_PROCESSES" == "1" ]]; then
  cmd=("$VENV_PY" "${train_args[@]}")
else
  cmd=(
    "$VENV_PY" -m accelerate.commands.launch
    --num_processes "$NUM_PROCESSES"
    --mixed_precision bf16
    --use_fsdp
    --fsdp_sharding_strategy FULL_SHARD
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP
    --fsdp_transformer_layer_cls_to_wrap "$FSDP_TRANSFORMER_LAYER"
    --fsdp_use_orig_params true
    --fsdp_cpu_ram_efficient_loading true
    --fsdp_sync_module_states true
    --fsdp_activation_checkpointing "$FSDP_ACTIVATION_CHECKPOINTING"
    "${train_args[@]}"
  )
fi

cd "$PROJECT_ROOT"
exec "${cmd[@]}"

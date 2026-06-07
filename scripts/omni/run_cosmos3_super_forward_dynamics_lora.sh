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
DEVICE_MAP="${DEVICE_MAP:-balanced}"
DTYPE="${DTYPE:-bfloat16}"
SEED="${SEED:-123}"
TIMESTEP_SAMPLING="${TIMESTEP_SAMPLING:-uniform}"
OVERRIDE_RESOLUTION_TIER="${OVERRIDE_RESOLUTION_TIER:-}"
DRY_RUN="${DRY_RUN:-0}"

args=(
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
  args+=(--override-resolution-tier "$OVERRIDE_RESOLUTION_TIER")
fi

if [[ "$DRY_RUN" == "1" ]]; then
  args+=(--dry-run)
fi

cd "$PROJECT_ROOT"
exec "$VENV_PY" "${args[@]}"

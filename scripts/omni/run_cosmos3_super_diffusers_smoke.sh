#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
VENV_PY="${VENV_PY:-$PROJECT_ROOT/.venv/bin/python}"

RUN_ID="${RUN_ID:-xperience10m_cosmos3_super_diffusers_runtime_smoke}"
MODEL_DIR="${MODEL_DIR:-$HOME/Ropedia/cosmos3_models/nv-community__Cosmos3-Super}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/results/omni_finetune/$RUN_ID}"
DEVICE_MAP="${DEVICE_MAP:-balanced}"
NUM_FRAMES="${NUM_FRAMES:-5}"
HEIGHT="${HEIGHT:-256}"
WIDTH="${WIDTH:-256}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-1}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-1.0}"
FLOW_SHIFT="${FLOW_SHIFT:-10.0}"
SEED="${SEED:-123}"
GENERATE="${GENERATE:-0}"
ENABLE_SAFETY_CHECK="${ENABLE_SAFETY_CHECK:-0}"
ALLOW_REMOTE_FILES="${ALLOW_REMOTE_FILES:-0}"

args=(
  "$PROJECT_ROOT/scripts/omni/cosmos3_super_diffusers_smoke.py"
  --workspace "$PROJECT_ROOT"
  --model-dir "$MODEL_DIR"
  --output-dir "$OUTPUT_DIR"
  --run-id "$RUN_ID"
  --device-map "$DEVICE_MAP"
  --num-frames "$NUM_FRAMES"
  --height "$HEIGHT"
  --width "$WIDTH"
  --num-inference-steps "$NUM_INFERENCE_STEPS"
  --guidance-scale "$GUIDANCE_SCALE"
  --flow-shift "$FLOW_SHIFT"
  --seed "$SEED"
)

if [[ "$GENERATE" == "1" ]]; then
  args+=(--generate)
fi
if [[ "$ENABLE_SAFETY_CHECK" == "1" ]]; then
  args+=(--enable-safety-check)
fi
if [[ "$ALLOW_REMOTE_FILES" == "1" ]]; then
  args+=(--allow-remote-files)
fi

cd "$PROJECT_ROOT"
exec "$VENV_PY" "${args[@]}"

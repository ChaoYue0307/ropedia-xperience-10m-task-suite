#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

VENV_PY="${VENV_PY:-$ROOT_DIR/.venv/bin/python}"
DATASET_JSONL="${DATASET_JSONL:-results/omni_finetune/xperience10m_qwen3_omni_128ep_96train_16val_16test_valmon_20260605_dataset/dataset.jsonl}"
MODEL_DIR="${MODEL_DIR:-/home/cy/Ropedia/cosmos3_models/nv-community__Cosmos3-Super}"
RUN_ID="${RUN_ID:-xperience10m_cosmos3_super_training_readiness}"
LOAD_PIPELINE="${LOAD_PIPELINE:-0}"
DEVICE_MAP="${DEVICE_MAP:-balanced}"

ARGS=(
  --dataset-jsonl "$DATASET_JSONL"
  --model-dir "$MODEL_DIR"
  --run-id "$RUN_ID"
  --device-map "$DEVICE_MAP"
)

if [[ "$LOAD_PIPELINE" == "1" ]]; then
  ARGS+=(--load-pipeline)
fi

"$VENV_PY" scripts/omni/probe_cosmos3_super_training_readiness.py "${ARGS[@]}"

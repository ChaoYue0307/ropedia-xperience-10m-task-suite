#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_ROOT/results/omni_finetune}"
VENV_PY="${VENV_PY:-$PROJECT_ROOT/.venv/bin/python}"

SOURCE_DATASET_RUN_ID="${SOURCE_DATASET_RUN_ID:-xperience10m_qwen3_omni_128ep_96train_16val_16test_valmon_20260605}"
SOURCE_DATASET_JSONL="${SOURCE_DATASET_JSONL:-$RESULT_ROOT/${SOURCE_DATASET_RUN_ID}_dataset/dataset.jsonl}"
SOURCE_DATASET_MANIFEST="${SOURCE_DATASET_MANIFEST:-$RESULT_ROOT/${SOURCE_DATASET_RUN_ID}_dataset/dataset_manifest.json}"
COSMOS_MODEL_DIR="${COSMOS_MODEL_DIR:-$HOME/Ropedia/cosmos3_models/nvidia__Cosmos3-Nano}"

DATASET_RUN_ID="${DATASET_RUN_ID:-xperience10m_cosmos3_nano_128ep_future_window_h5_compat}"
TRAIN_RUN_ID="${TRAIN_RUN_ID:-${DATASET_RUN_ID}_adapter}"
EVAL_RUN_ID="${EVAL_RUN_ID:-${TRAIN_RUN_ID}_eval_test_full}"
HORIZON_WINDOWS="${HORIZON_WINDOWS:-5}"
MAX_PAIRS_PER_EPISODE="${MAX_PAIRS_PER_EPISODE:-0}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-0}"
TOP_K="${TOP_K:-5}"

EXPECTED_TRAIN_EPISODES="${EXPECTED_TRAIN_EPISODES:-89}"
EXPECTED_VAL_EPISODES="${EXPECTED_VAL_EPISODES:-16}"
EXPECTED_TEST_EPISODES="${EXPECTED_TEST_EPISODES:-14}"
EXPECTED_NUM_PROCESSES="${EXPECTED_NUM_PROCESSES:-1}"

RUN_DIR="$RESULT_ROOT/$DATASET_RUN_ID"
DATASET_DIR="$RESULT_ROOT/${DATASET_RUN_ID}_dataset"
TRAIN_DIR="$RESULT_ROOT/$TRAIN_RUN_ID"
EVAL_DIR="$RESULT_ROOT/$EVAL_RUN_ID"
STATUS_JSONL="$RUN_DIR/status.jsonl"
LOG="$RUN_DIR/run.log"
LOCK_DIR="$RUN_DIR/run.lock"

mkdir -p "$RUN_DIR" "$DATASET_DIR" "$TRAIN_DIR" "$EVAL_DIR"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "Run already active or stale lock exists: $LOCK_DIR" >&2
  exit 1
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT

exec > >(tee -a "$LOG") 2>&1
cd "$PROJECT_ROOT"

json_log() {
  "$VENV_PY" - "$STATUS_JSONL" "$@" <<'PY'
import json
import sys
import time

path = sys.argv[1]
payload = {"time": time.time()}
for item in sys.argv[2:]:
    key, value = item.split("=", 1)
    if value.isdigit():
        value = int(value)
    payload[key] = value
with open(path, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(payload, sort_keys=True) + "\n")
print(json.dumps(payload, sort_keys=True), flush=True)
PY
}

json_log event=cosmos3_compat_start dataset_run_id="$DATASET_RUN_ID" train_run_id="$TRAIN_RUN_ID" eval_run_id="$EVAL_RUN_ID"

"$VENV_PY" scripts/omni/export_cosmos3_future_window_dataset.py \
  --workspace "$PROJECT_ROOT" \
  --source-dataset-jsonl "$SOURCE_DATASET_JSONL" \
  --source-dataset-manifest "$SOURCE_DATASET_MANIFEST" \
  --run-id "$DATASET_RUN_ID" \
  --run-dir "$RUN_DIR" \
  --output-dir "$DATASET_DIR" \
  --horizon-windows "$HORIZON_WINDOWS" \
  --max-pairs-per-episode "$MAX_PAIRS_PER_EPISODE" \
  --cosmos-model-dir "$COSMOS_MODEL_DIR"
json_log event=cosmos3_dataset_done dataset_jsonl="$DATASET_DIR/dataset.jsonl"

"$VENV_PY" scripts/omni/eval_cosmos3_future_window_retrieval.py \
  --workspace "$PROJECT_ROOT" \
  --dataset-jsonl "$DATASET_DIR/dataset.jsonl" \
  --run-id "$TRAIN_RUN_ID" \
  --eval-run-id "$EVAL_RUN_ID" \
  --results-dir "$TRAIN_DIR" \
  --eval-output-dir "$EVAL_DIR" \
  --cosmos-model-dir "$COSMOS_MODEL_DIR" \
  --max-train-samples "$MAX_TRAIN_SAMPLES" \
  --max-eval-samples "$MAX_EVAL_SAMPLES" \
  --top-k "$TOP_K"
json_log event=cosmos3_eval_done metrics="$EVAL_DIR/metrics.json"

"$VENV_PY" scripts/omni/validate_omni_finetune_run.py \
  --workspace "$PROJECT_ROOT" \
  --run-id "$DATASET_RUN_ID" \
  --dataset-run-id "$DATASET_RUN_ID" \
  --train-run-id "$TRAIN_RUN_ID" \
  --backbone cosmos_world_model \
  --require-stage training \
  --expected-train-episodes "$EXPECTED_TRAIN_EPISODES" \
  --expected-val-episodes "$EXPECTED_VAL_EPISODES" \
  --expected-test-episodes "$EXPECTED_TEST_EPISODES" \
  --expected-dataset-train-episodes "$EXPECTED_TRAIN_EPISODES" \
  --expected-dataset-val-episodes "$EXPECTED_VAL_EPISODES" \
  --expected-dataset-test-episodes "$EXPECTED_TEST_EPISODES" \
  --expected-num-processes "$EXPECTED_NUM_PROCESSES" \
  --allow-zero-val-training \
  --output "$RUN_DIR/validation_training_${TRAIN_RUN_ID}.json"
json_log event=cosmos3_validation_training_done output="$RUN_DIR/validation_training_${TRAIN_RUN_ID}.json"

"$VENV_PY" scripts/omni/validate_omni_finetune_run.py \
  --workspace "$PROJECT_ROOT" \
  --run-id "$DATASET_RUN_ID" \
  --dataset-run-id "$DATASET_RUN_ID" \
  --train-run-id "$TRAIN_RUN_ID" \
  --eval-run-id "$EVAL_RUN_ID" \
  --backbone cosmos_world_model \
  --require-stage eval \
  --expected-train-episodes "$EXPECTED_TRAIN_EPISODES" \
  --expected-val-episodes "$EXPECTED_VAL_EPISODES" \
  --expected-test-episodes "$EXPECTED_TEST_EPISODES" \
  --expected-dataset-train-episodes "$EXPECTED_TRAIN_EPISODES" \
  --expected-dataset-val-episodes "$EXPECTED_VAL_EPISODES" \
  --expected-dataset-test-episodes "$EXPECTED_TEST_EPISODES" \
  --expected-num-processes "$EXPECTED_NUM_PROCESSES" \
  --allow-zero-val-training \
  --output "$RUN_DIR/validation_eval_${EVAL_RUN_ID}.json"
json_log event=cosmos3_validation_eval_done output="$RUN_DIR/validation_eval_${EVAL_RUN_ID}.json"

"$VENV_PY" scripts/omni/package_verified_omni_result.py \
  --workspace "$PROJECT_ROOT" \
  --dataset-run-id "$DATASET_RUN_ID" \
  --train-run-id "$TRAIN_RUN_ID" \
  --eval-run-id "$EVAL_RUN_ID" \
  --backbone cosmos_world_model
json_log event=cosmos3_package_done output="$RESULT_ROOT/verified_public/$EVAL_RUN_ID"

"$VENV_PY" scripts/omni/audit_verified_omni_package.py \
  --workspace "$PROJECT_ROOT" \
  --package-dir "$RESULT_ROOT/verified_public/$EVAL_RUN_ID" \
  --backbone cosmos_world_model \
  --output "$RUN_DIR/audit_verified_public_${EVAL_RUN_ID}.json"
json_log event=cosmos3_audit_done output="$RUN_DIR/audit_verified_public_${EVAL_RUN_ID}.json"

json_log event=complete run_id="$DATASET_RUN_ID"

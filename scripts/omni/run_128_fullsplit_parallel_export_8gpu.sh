#!/usr/bin/env bash
set -euo pipefail

# Full selected-episode Qwen3-Omni LoRA run:
# 96 train episodes, 16 validation episodes, 16 sealed test episodes.
# The test split is exported for final evaluation but never used for training.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-/home/cy/Ropedia/modelscope_data/xperience10m_128}"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_ROOT/results/omni_finetune}"
SELECTION_JSON="${SELECTION_JSON:-$RESULT_ROOT/xperience10m_128_episode_selection.json}"
VENV_PY="${VENV_PY:-$PROJECT_ROOT/.venv/bin/python}"
MODEL_DIR="${MODEL_DIR:-/home/cy/Ropedia/modelscope_models/Qwen__Qwen3-Omni-30B-A3B-Instruct}"

RUN_ID="${RUN_ID:-xperience10m_qwen3_omni_128ep_fullsplit_fast8gpu}"
TARGET_EPISODES="${TARGET_EPISODES:-128}"
EXPECTED_TRAIN_EPISODES="${EXPECTED_TRAIN_EPISODES:-96}"
EXPECTED_VAL_EPISODES="${EXPECTED_VAL_EPISODES:-16}"
EXPECTED_TEST_EPISODES="${EXPECTED_TEST_EPISODES:-16}"
EXPORT_WORKERS="${EXPORT_WORKERS:-8}"
MAX_WINDOWS_PER_EPISODE="${MAX_WINDOWS_PER_EPISODE:-32}"
MAX_VIDEO_FRAMES="${MAX_VIDEO_FRAMES:-16}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-512}"
EVAL_SAMPLE_LIMIT="${EVAL_SAMPLE_LIMIT:-0}"
EPOCHS="${EPOCHS:-1}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
MIN_JSON_VALIDITY="${MIN_JSON_VALIDITY:-0.98}"

RUN_DIR="$RESULT_ROOT/$RUN_ID"
DATASET_RUN_ID="${RUN_ID}_dataset"
DATASET_DIR="$RESULT_ROOT/$DATASET_RUN_ID"
MANIFEST="$RUN_DIR/episode_manifest.json"
DATASET_JSONL="$DATASET_DIR/dataset.jsonl"
LOG="$RUN_DIR/run.log"
STATUS_JSONL="$RUN_DIR/status.jsonl"
LOCK_DIR="$RUN_DIR/run.lock"
ADAPTER_DIR="$PROJECT_ROOT/checkpoints/${RUN_ID}_lora/adapter_lora"
EVAL_DIR="$RESULT_ROOT/${RUN_ID}_eval"

mkdir -p "$RUN_DIR" "$DATASET_DIR"
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

json_log event=preflight_start run_id="$RUN_ID"
"$VENV_PY" - "$DATA_ROOT" "$TARGET_EPISODES" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
target = int(sys.argv[2])
episodes = [path.parent for path in root.rglob("annotation.hdf5")]
complete = [episode for episode in episodes if len(list(episode.glob("*.mp4"))) >= 6]
mp4_count = sum(1 for _ in root.rglob("*.mp4"))
payload = {
    "annotation_count": len(episodes),
    "complete6_count": len(complete),
    "mp4_count": mp4_count,
}
print(json.dumps({"event": "data_count", **payload}, sort_keys=True))
if payload["annotation_count"] < target or payload["complete6_count"] < target or payload["mp4_count"] < target * 6:
    raise SystemExit(f"selected data is not ready: {payload}")
PY
json_log event=preflight_done

if pgrep -af "train_qwen3_omni_lora.py" >/dev/null 2>&1; then
  json_log event=blocked_existing_training
  exit 2
fi

json_log event=manifest_start
"$VENV_PY" scripts/omni/build_selection_episode_manifest.py \
  --workspace "$PROJECT_ROOT" \
  --data-root "$DATA_ROOT" \
  --selection-json "$SELECTION_JSON" \
  --output "$MANIFEST" \
  --report-output "$RUN_DIR/MANIFEST_REPORT.md" \
  --include-split train \
  --include-split val \
  --include-split test \
  --min-train-episodes "$EXPECTED_TRAIN_EPISODES" \
  --min-val-episodes "$EXPECTED_VAL_EPISODES"

"$VENV_PY" - "$MANIFEST" "$EXPECTED_TRAIN_EPISODES" "$EXPECTED_VAL_EPISODES" "$EXPECTED_TEST_EPISODES" <<'PY'
import json
import sys
from collections import Counter

manifest_path = sys.argv[1]
expected = {"train": int(sys.argv[2]), "val": int(sys.argv[3]), "test": int(sys.argv[4])}
payload = json.load(open(manifest_path, "r", encoding="utf-8"))
episodes = payload.get("episodes", [])
counts = Counter(ep.get("split") for ep in episodes)
if dict(counts) != expected:
    raise SystemExit(f"unexpected episode split counts: {dict(counts)} != {expected}")
ids = [ep.get("episode_id") for ep in episodes]
if len(ids) != len(set(ids)):
    raise SystemExit("duplicate episode ids in manifest")
print(json.dumps({"event": "manifest_guard_ok", "episode_count": len(episodes), "split_counts": dict(counts)}, sort_keys=True))
PY
json_log event=manifest_done manifest="$MANIFEST"

"$VENV_PY" scripts/omni/validate_omni_finetune_run.py \
  --workspace "$PROJECT_ROOT" \
  --run-id "$RUN_ID" \
  --require-stage manifest \
  --expected-train-episodes "$EXPECTED_TRAIN_EPISODES" \
  --expected-val-episodes "$EXPECTED_VAL_EPISODES" \
  --expected-test-episodes "$EXPECTED_TEST_EPISODES" \
  --output "$RUN_DIR/validation_manifest.json"
json_log event=validation_manifest_done output="$RUN_DIR/validation_manifest.json"

json_log event=parallel_export_start dataset_run_id="$DATASET_RUN_ID" workers="$EXPORT_WORKERS"
"$VENV_PY" scripts/omni/parallel_export_qwen3_omni_action_dataset.py \
  --workspace "$PROJECT_ROOT" \
  --manifest "$MANIFEST" \
  --run-id "$DATASET_RUN_ID" \
  --output-dir "$DATASET_DIR" \
  --num-workers "$EXPORT_WORKERS" \
  --max-windows-per-episode "$MAX_WINDOWS_PER_EPISODE" \
  --max-video-frames "$MAX_VIDEO_FRAMES" \
  --audio-source fisheye_cam0 \
  --audio-sample-rate 16000 \
  --audio-band-count 16
json_log event=parallel_export_done dataset_jsonl="$DATASET_JSONL"

"$VENV_PY" - "$DATASET_JSONL" <<'PY'
import json
import sys
from collections import Counter, defaultdict

counts = Counter()
episodes = defaultdict(set)
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    for line in handle:
        row = json.loads(line)
        split = row.get("split")
        counts[split] += 1
        episodes[split].add(row.get("episode_id"))
if not counts.get("train") or not counts.get("val") or not counts.get("test"):
    raise SystemExit(f"missing exported split samples: {dict(counts)}")
print(json.dumps({
    "event": "dataset_guard_ok",
    "sample_split_counts": dict(counts),
    "episode_split_counts": {split: len(values) for split, values in episodes.items()},
}, sort_keys=True))
PY

"$VENV_PY" scripts/omni/validate_omni_finetune_run.py \
  --workspace "$PROJECT_ROOT" \
  --run-id "$RUN_ID" \
  --require-stage dataset \
  --expected-train-episodes "$EXPECTED_TRAIN_EPISODES" \
  --expected-val-episodes "$EXPECTED_VAL_EPISODES" \
  --expected-test-episodes "$EXPECTED_TEST_EPISODES" \
  --output "$RUN_DIR/validation_dataset.json"
json_log event=validation_dataset_done output="$RUN_DIR/validation_dataset.json"

json_log event=train_start run_id="${RUN_ID}_lora" num_processes="$NUM_PROCESSES"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
"$VENV_PY" -m accelerate.commands.launch \
  --num_processes "$NUM_PROCESSES" \
  --mixed_precision bf16 \
  scripts/omni/train_qwen3_omni_lora.py \
  --dataset-jsonl "$DATASET_JSONL" \
  --model-id "$MODEL_DIR" \
  --run-id "${RUN_ID}_lora" \
  --train-split train \
  --val-split val \
  --epochs "$EPOCHS" \
  --batch-size 1 \
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
  --max-train-samples 0 \
  --max-val-samples "$MAX_VAL_SAMPLES" \
  --local-files-only \
  --gradient-checkpointing \
  --progress-every 10
json_log event=train_done run_id="${RUN_ID}_lora" adapter_dir="$ADAPTER_DIR"

"$VENV_PY" scripts/omni/validate_omni_finetune_run.py \
  --workspace "$PROJECT_ROOT" \
  --run-id "$RUN_ID" \
  --require-stage training \
  --expected-train-episodes "$EXPECTED_TRAIN_EPISODES" \
  --expected-val-episodes "$EXPECTED_VAL_EPISODES" \
  --expected-test-episodes "$EXPECTED_TEST_EPISODES" \
  --expected-num-processes "$NUM_PROCESSES" \
  --output "$RUN_DIR/validation_training.json"
json_log event=validation_training_done output="$RUN_DIR/validation_training.json"

json_log event=eval_start run_id="${RUN_ID}_eval"
"$VENV_PY" scripts/omni/eval_qwen3_omni_lora.py \
  --dataset-jsonl "$DATASET_JSONL" \
  --model-id "$MODEL_DIR" \
  --adapter-dir "$ADAPTER_DIR" \
  --run-id "${RUN_ID}_eval" \
  --eval-split test \
  --sample-limit "$EVAL_SAMPLE_LIMIT" \
  --local-files-only
json_log event=eval_done run_id="${RUN_ID}_eval" metrics="$EVAL_DIR/metrics.json"

"$VENV_PY" scripts/omni/validate_omni_finetune_run.py \
  --workspace "$PROJECT_ROOT" \
  --run-id "$RUN_ID" \
  --require-stage eval \
  --expected-train-episodes "$EXPECTED_TRAIN_EPISODES" \
  --expected-val-episodes "$EXPECTED_VAL_EPISODES" \
  --expected-test-episodes "$EXPECTED_TEST_EPISODES" \
  --expected-num-processes "$NUM_PROCESSES" \
  --min-json-validity "$MIN_JSON_VALIDITY" \
  --output "$RUN_DIR/validation_eval.json"
json_log event=validation_eval_done output="$RUN_DIR/validation_eval.json"

"$VENV_PY" scripts/omni/omni_finetune_runbook.py \
  --run-id "$RUN_ID" \
  --manifest "$MANIFEST" \
  --metric-file "$EVAL_DIR/metrics.json" || true

json_log event=complete run_id="$RUN_ID"

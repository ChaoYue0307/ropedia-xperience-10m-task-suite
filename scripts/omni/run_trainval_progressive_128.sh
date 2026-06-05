#!/usr/bin/env bash
set -euo pipefail

# Launch a progressive Qwen3-Omni LoRA run on currently prepared train/val
# episodes from the fixed 128-episode Xperience-10M selection. Held-out test
# episodes are sealed and are not exported or evaluated here.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data/xperience10m_128}"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_ROOT/results/omni_finetune}"
SELECTION_JSON="${SELECTION_JSON:-$RESULT_ROOT/xperience10m_128_episode_selection.json}"
VENV_PY="${VENV_PY:-$PROJECT_ROOT/.venv/bin/python}"
MODEL_DIR="${MODEL_DIR:-Qwen/Qwen3-Omni-30B-A3B-Instruct}"
BACKBONE_CONFIG="${BACKBONE_CONFIG:-configs/omni_backbones/qwen3_omni_lora.json}"

RUN_ID="${RUN_ID:-xperience10m_qwen3_omni_128ep_trainval_progressive}"
MIN_TRAIN_EPISODES="${MIN_TRAIN_EPISODES:-80}"
MIN_VAL_EPISODES="${MIN_VAL_EPISODES:-12}"
MAX_WINDOWS_PER_EPISODE="${MAX_WINDOWS_PER_EPISODE:-128}"
MAX_VIDEO_FRAMES="${MAX_VIDEO_FRAMES:-16}"
TRAIN_VAL_SPLIT="${TRAIN_VAL_SPLIT:-__none__}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-0}"
EPOCHS="${EPOCHS:-1}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
USE_FSDP="${USE_FSDP:-1}"
FSDP_TRANSFORMER_LAYER="${FSDP_TRANSFORMER_LAYER:-Qwen3OmniMoeThinkerTextDecoderLayer}"
FSDP_CPU_RAM_EFFICIENT_LOADING="${FSDP_CPU_RAM_EFFICIENT_LOADING:-true}"
FSDP_SYNC_MODULE_STATES="${FSDP_SYNC_MODULE_STATES:-true}"
FSDP_ACTIVATION_CHECKPOINTING="${FSDP_ACTIVATION_CHECKPOINTING:-true}"

RUN_DIR="$RESULT_ROOT/$RUN_ID"
DATASET_RUN_ID="${RUN_ID}_dataset"
DATASET_DIR="$RESULT_ROOT/$DATASET_RUN_ID"
MANIFEST="$RUN_DIR/episode_manifest_trainval.json"
DATASET_JSONL="$DATASET_DIR/dataset.jsonl"
LOG="$RUN_DIR/trainval_progressive.log"
STATUS_JSONL="$RUN_DIR/status.jsonl"
LOCK_DIR="$RUN_DIR/trainval.lock"

mkdir -p "$RUN_DIR" "$DATASET_DIR"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "Progressive train/val run already running or stale lock exists: $LOCK_DIR" >&2
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

if pgrep -af "train_qwen3_omni_lora.py.*${RUN_ID}" >/dev/null 2>&1; then
  json_log event=train_already_running run_id="$RUN_ID"
  exit 0
fi

if [ -f "$RUN_DIR/training_metadata.json" ]; then
  json_log event=train_already_complete metadata="$RUN_DIR/training_metadata.json"
  exit 0
fi

json_log event=manifest_start run_id="$RUN_ID"
"$VENV_PY" scripts/omni/build_selection_episode_manifest.py \
  --workspace "$PROJECT_ROOT" \
  --data-root "$DATA_ROOT" \
  --selection-json "$SELECTION_JSON" \
  --output "$MANIFEST" \
  --report-output "$RUN_DIR/TRAINVAL_MANIFEST_REPORT.md" \
  --include-split train \
  --include-split val \
  --min-train-episodes "$MIN_TRAIN_EPISODES" \
  --min-val-episodes "$MIN_VAL_EPISODES"
json_log event=manifest_done manifest="$MANIFEST"

"$VENV_PY" - "$MANIFEST" <<'PY'
import json
import sys
from collections import Counter

payload = json.load(open(sys.argv[1], "r", encoding="utf-8"))
counts = Counter(ep.get("split") for ep in payload.get("episodes", []))
if counts.get("test", 0):
    raise SystemExit(f"test episodes leaked into train/val manifest: {counts}")
if counts.get("train", 0) <= 0 or counts.get("val", 0) <= 0:
    raise SystemExit(f"train/val manifest is empty or incomplete: {counts}")
print(json.dumps({"event": "manifest_guard_ok", "split_counts": dict(counts)}, sort_keys=True))
PY

json_log event=export_dataset_start dataset_run_id="$DATASET_RUN_ID"
"$VENV_PY" scripts/omni/export_qwen3_omni_action_dataset.py \
  --manifest "$MANIFEST" \
  --run-id "$DATASET_RUN_ID" \
  --output-dir "$DATASET_DIR" \
  --max-windows-per-episode "$MAX_WINDOWS_PER_EPISODE" \
  --max-video-frames "$MAX_VIDEO_FRAMES"
json_log event=export_dataset_done dataset_jsonl="$DATASET_JSONL"

"$VENV_PY" - "$DATASET_JSONL" <<'PY'
import json
import sys
from collections import Counter

counts = Counter()
episodes = set()
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    for line in handle:
        row = json.loads(line)
        counts[row.get("split")] += 1
        episodes.add(row.get("episode_id"))
if counts.get("test", 0):
    raise SystemExit(f"test samples leaked into train/val dataset: {counts}")
print(json.dumps({"event": "dataset_guard_ok", "split_counts": dict(counts), "episodes": len(episodes)}, sort_keys=True))
PY

json_log event=train_start run_id="$RUN_ID"
train_cmd=(
  "$VENV_PY" -m accelerate.commands.launch
  --num_processes "$NUM_PROCESSES"
  --mixed_precision bf16
)
if [[ "$USE_FSDP" == "1" ]]; then
  train_cmd+=(
    --use_fsdp
    --fsdp_sharding_strategy FULL_SHARD
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP
    --fsdp_transformer_layer_cls_to_wrap "$FSDP_TRANSFORMER_LAYER"
    --fsdp_use_orig_params true
    --fsdp_cpu_ram_efficient_loading "$FSDP_CPU_RAM_EFFICIENT_LOADING"
    --fsdp_sync_module_states "$FSDP_SYNC_MODULE_STATES"
    --fsdp_activation_checkpointing "$FSDP_ACTIVATION_CHECKPOINTING"
  )
fi
train_cmd+=(
  scripts/omni/train_qwen3_omni_lora.py
  --dataset-jsonl "$DATASET_JSONL"
  --model-id "$MODEL_DIR"
  --backbone-config "$BACKBONE_CONFIG"
  --run-id "$RUN_ID"
  --train-split train
  --val-split "$TRAIN_VAL_SPLIT"
  --epochs "$EPOCHS"
  --batch-size 1
  --gradient-accumulation-steps 8
  --max-train-samples 0
  --max-val-samples "$MAX_VAL_SAMPLES"
  --local-files-only
  --gradient-checkpointing
  --progress-every 20
)
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
"${train_cmd[@]}"
json_log event=train_done run_id="$RUN_ID"

json_log event=complete run_id="$RUN_ID"

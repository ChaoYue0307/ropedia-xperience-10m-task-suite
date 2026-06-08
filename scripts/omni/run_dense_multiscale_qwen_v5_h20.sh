#!/usr/bin/env bash
set -euo pipefail

# H20-oriented next-stage run:
# 1. Build the full compact 106k dense/multiscale hierarchical index.
# 2. Run compact hierarchical simple/NN baselines.
# 3. Export a balanced raw-media multiscale Qwen dataset capped per
#    episode/scale to keep the current trainer's per-rank JSONL load bounded.
# 4. Train/evaluate a distinct Qwen3-Omni LoRA v5 run on all 8 H20 GPUs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
ROPEDIA_WORKSPACE="${ROPEDIA_WORKSPACE:-$HOME/Ropedia}"
DATA_ROOT="${DATA_ROOT:-$ROPEDIA_WORKSPACE/modelscope_data/xperience10m_128}"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_ROOT/results/omni_finetune}"
SELECTION_JSON="${SELECTION_JSON:-$RESULT_ROOT/xperience10m_128_episode_selection.json}"
VENV_PY="${VENV_PY:-$PROJECT_ROOT/.venv/bin/python}"
MODEL_DIR="${MODEL_DIR:-$ROPEDIA_WORKSPACE/modelscope_models/Qwen__Qwen3-Omni-30B-A3B-Instruct}"
BACKBONE_CONFIG="${BACKBONE_CONFIG:-configs/omni_backbones/qwen3_omni_lora.json}"

RUN_ID="${RUN_ID:-xperience10m_qwen3_omni_128ep_multiscale_cap96_v5_full8gpu_lora}"
COMPACT_RUN_ID="${COMPACT_RUN_ID:-xperience10m_128ep_dense_multiscale_hierarchical_v1_20260608}"
BASELINE_RUN_ID="${BASELINE_RUN_ID:-xperience10m_128ep_dense_multiscale_hierarchical_baselines_v1_20260608}"
DATASET_RUN_ID="${DATASET_RUN_ID:-${RUN_ID}_dataset}"
EXPORT_WORKERS="${EXPORT_WORKERS:-8}"
MAX_WINDOWS_PER_EPISODE="${MAX_WINDOWS_PER_EPISODE:-96}"
MAX_VIDEO_FRAMES="${MAX_VIDEO_FRAMES:-16}"
EPOCHS="${EPOCHS:-1}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-1024}"
EVAL_SAMPLE_LIMIT="${EVAL_SAMPLE_LIMIT:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
EVAL_SHARDS="${EVAL_SHARDS:-8}"
EVAL_CUDA_DEVICE_GROUPS="${EVAL_CUDA_DEVICE_GROUPS:-0 1 2 3 4 5 6 7}"
EVAL_DEVICE_MAP="${EVAL_DEVICE_MAP:-auto}"
EVAL_DTYPE="${EVAL_DTYPE:-bfloat16}"

RUN_DIR="$RESULT_ROOT/$RUN_ID"
MANIFEST="$RUN_DIR/episode_manifest.json"
STATUS_JSONL="$RUN_DIR/status.jsonl"
LOG="$RUN_DIR/run.log"
COMPACT_DIR="$RESULT_ROOT/$COMPACT_RUN_ID"
BASELINE_DIR="$RESULT_ROOT/$BASELINE_RUN_ID"
DATASET_DIR="$RESULT_ROOT/$DATASET_RUN_ID"
MERGED_DATASET_JSONL="$DATASET_DIR/dataset.jsonl"
ADAPTER_DIR="$PROJECT_ROOT/checkpoints/${RUN_ID}/adapter_lora"
EVAL_DIR="$RESULT_ROOT/${RUN_ID}_eval_test_full"
LOCK_DIR="$RUN_DIR/run.lock"

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

json_log event=start run_id="$RUN_ID" max_windows_per_episode="$MAX_WINDOWS_PER_EPISODE"

if [[ ! -s "$COMPACT_DIR/dataset_manifest.json" ]]; then
  json_log event=compact_dense_start run_id="$COMPACT_RUN_ID"
  "$VENV_PY" scripts/omni/build_dense_multiscale_hierarchical_128.py \
    --run-id "$COMPACT_RUN_ID" \
    --output-dir "$COMPACT_DIR"
  json_log event=compact_dense_done manifest="$COMPACT_DIR/dataset_manifest.json"
else
  json_log event=compact_dense_skip manifest="$COMPACT_DIR/dataset_manifest.json"
fi

if [[ ! -s "$BASELINE_DIR/summary_report.json" ]]; then
  json_log event=compact_baseline_start run_id="$BASELINE_RUN_ID"
  CUDA_VISIBLE_DEVICES="${BASELINE_CUDA_VISIBLE_DEVICES:-0}" \
  "$VENV_PY" scripts/omni/run_dense_multiscale_hierarchical_baselines.py \
    --dataset-jsonl "$COMPACT_DIR/dense_multiscale_windows.jsonl" \
    --dataset-manifest "$COMPACT_DIR/dataset_manifest.json" \
    --run-id "$BASELINE_RUN_ID" \
    --output-dir "$BASELINE_DIR" \
    --include-neural \
    --neural-device cuda
  json_log event=compact_baseline_done summary="$BASELINE_DIR/summary_report.json"
else
  json_log event=compact_baseline_skip summary="$BASELINE_DIR/summary_report.json"
fi

if [[ ! -s "$MANIFEST" ]]; then
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
    --min-train-episodes 96 \
    --min-val-episodes 16
  json_log event=manifest_done manifest="$MANIFEST"
else
  json_log event=manifest_skip manifest="$MANIFEST"
fi

export_component() {
  local scale_id="$1"
  local window_frames="$2"
  local stride_frames="$3"
  local context_frames="$4"
  local out_dir="$RESULT_ROOT/${DATASET_RUN_ID}_${scale_id}"
  if [[ -s "$out_dir/dataset_manifest.json" ]]; then
    json_log event=component_export_skip scale_id="$scale_id" dataset_dir="$out_dir"
    return 0
  fi
  json_log event=component_export_start scale_id="$scale_id" window_frames="$window_frames" stride_frames="$stride_frames" context_frames="$context_frames" output_dir="$out_dir"
  "$VENV_PY" scripts/omni/parallel_export_qwen3_omni_action_dataset.py \
    --workspace "$PROJECT_ROOT" \
    --manifest "$MANIFEST" \
    --run-id "${DATASET_RUN_ID}_${scale_id}" \
    --output-dir "$out_dir" \
    --num-workers "$EXPORT_WORKERS" \
    --window-frames "$window_frames" \
    --stride-frames "$stride_frames" \
    --qwen-context-frames "$context_frames" \
    --max-windows-per-episode "$MAX_WINDOWS_PER_EPISODE" \
    --max-video-frames "$MAX_VIDEO_FRAMES" \
    --audio-source fisheye_cam0 \
    --audio-sample-rate 16000 \
    --audio-band-count 16
  json_log event=component_export_done scale_id="$scale_id" dataset_dir="$out_dir"
}

export_component dense_20f_stride10 20 10 120
export_component medium_40f_stride20 40 20 160
export_component long_80f_stride40 80 40 240

if [[ ! -s "$DATASET_DIR/dataset_manifest.json" ]]; then
  json_log event=merge_start dataset_run_id="$DATASET_RUN_ID"
  "$VENV_PY" scripts/omni/merge_qwen3_multiscale_datasets.py \
    --run-id "$DATASET_RUN_ID" \
    --output-dir "$DATASET_DIR" \
    --component "dense_20f_stride10=$RESULT_ROOT/${DATASET_RUN_ID}_dense_20f_stride10" \
    --component "medium_40f_stride20=$RESULT_ROOT/${DATASET_RUN_ID}_medium_40f_stride20" \
    --component "long_80f_stride40=$RESULT_ROOT/${DATASET_RUN_ID}_long_80f_stride40"
  json_log event=merge_done dataset_jsonl="$MERGED_DATASET_JSONL"
else
  json_log event=merge_skip dataset_jsonl="$MERGED_DATASET_JSONL"
fi

"$VENV_PY" - "$DATASET_DIR/dataset_manifest.json" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
counts = manifest.get("split_counts", {})
if not counts.get("train") or not counts.get("val") or not counts.get("test"):
    raise SystemExit(f"missing split rows: {counts}")
print(json.dumps({"event": "dataset_guard_ok", "split_counts": counts, "scale_counts": manifest.get("scale_counts")}, sort_keys=True))
PY

if pgrep -af "train_qwen3_omni_lora.py" >/dev/null 2>&1; then
  json_log event=blocked_existing_qwen_training run_id="$RUN_ID"
  pgrep -af "train_qwen3_omni_lora.py" || true
  exit 2
fi

if [[ ! -s "$ADAPTER_DIR/adapter_config.json" ]]; then
  json_log event=train_start run_id="$RUN_ID" num_processes="$NUM_PROCESSES"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  "$VENV_PY" -m accelerate.commands.launch \
    --num_processes "$NUM_PROCESSES" \
    --mixed_precision bf16 \
    --use_fsdp \
    --fsdp_sharding_strategy FULL_SHARD \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_transformer_layer_cls_to_wrap Qwen3OmniMoeThinkerTextDecoderLayer \
    --fsdp_use_orig_params true \
    --fsdp_cpu_ram_efficient_loading true \
    --fsdp_sync_module_states true \
    --fsdp_activation_checkpointing true \
    scripts/omni/train_qwen3_omni_lora.py \
    --dataset-jsonl "$MERGED_DATASET_JSONL" \
    --model-id "$MODEL_DIR" \
    --backbone-config "$BACKBONE_CONFIG" \
    --run-id "$RUN_ID" \
    --train-split train \
    --val-split val \
    --epochs "$EPOCHS" \
    --batch-size 1 \
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
    --max-train-samples 0 \
    --max-val-samples "$MAX_VAL_SAMPLES" \
    --local-files-only \
    --gradient-checkpointing \
    --progress-every 25
  json_log event=train_done adapter_dir="$ADAPTER_DIR"
else
  json_log event=train_skip adapter_dir="$ADAPTER_DIR"
fi

if [[ ! -s "$EVAL_DIR/metrics.json" ]]; then
  json_log event=eval_start run_id="${RUN_ID}_eval_test_full" eval_sample_limit="$EVAL_SAMPLE_LIMIT" eval_shards="$EVAL_SHARDS"
  VENV_PY="$VENV_PY" \
  DATASET_JSONL="$MERGED_DATASET_JSONL" \
  MODEL_DIR="$MODEL_DIR" \
  ADAPTER_DIR="$ADAPTER_DIR" \
  RUN_ID="${RUN_ID}_eval_test_full" \
  EVAL_SPLIT=test \
  SAMPLE_LIMIT="$EVAL_SAMPLE_LIMIT" \
  MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
  SHARDS="$EVAL_SHARDS" \
  CUDA_DEVICE_GROUPS="$EVAL_CUDA_DEVICE_GROUPS" \
  DEVICE_MAP="$EVAL_DEVICE_MAP" \
  DTYPE="$EVAL_DTYPE" \
  LOCAL_FILES_ONLY=1 \
  scripts/omni/run_qwen3_omni_lora_eval_sharded.sh
  json_log event=eval_done metrics="$EVAL_DIR/metrics.json"
else
  json_log event=eval_skip metrics="$EVAL_DIR/metrics.json"
fi

json_log event=complete run_id="$RUN_ID"

#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-/home/cy/Ropedia/ropedia-episode-task-suite}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/cy/Ropedia}"
VENV_PY="${VENV_PY:-$WORKSPACE/.venv/bin/python}"
RUN_ID="${RUN_ID:-xperience10m_qwen3_omni_32ep}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/modelscope_data}"
MAX_EPISODES="${MAX_EPISODES:-32}"
MAX_WINDOWS_PER_EPISODE="${MAX_WINDOWS_PER_EPISODE:-128}"
MAX_VIDEO_FRAMES="${MAX_VIDEO_FRAMES:-16}"
EPOCHS="${EPOCHS:-1}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
VAL_SPLIT="${VAL_SPLIT:-val}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-Omni-30B-A3B-Instruct}"
LOCAL_MODEL_DIR="${LOCAL_MODEL_DIR:-$PROJECT_ROOT/modelscope_models/Qwen__Qwen3-Omni-30B-A3B-Instruct}"

RESULT_DIR="$WORKSPACE/results/omni_finetune/$RUN_ID"
DATASET_RUN_ID="${RUN_ID}_dataset"
DATASET_DIR="$WORKSPACE/results/omni_finetune/$DATASET_RUN_ID"
MANIFEST="$RESULT_DIR/episode_manifest.json"
LOG_DIR="$RESULT_DIR/logs"
mkdir -p "$LOG_DIR" "$LOCAL_MODEL_DIR"

exec > >(tee -a "$LOG_DIR/pipeline.log") 2>&1

cd "$WORKSPACE"

phase() {
  echo "PHASE: $1"
  "$VENV_PY" - <<PY
import json, time
path = "$RESULT_DIR/pipeline_status.jsonl"
with open(path, "a", encoding="utf-8") as fp:
    fp.write(json.dumps({"event": "phase", "phase": "$1", "timestamp": time.time()}) + "\\n")
PY
}

phase "preflight"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits
"$VENV_PY" - <<'PY'
mods = ["torch", "transformers", "accelerate", "peft", "qwen_omni_utils", "soundfile", "librosa", "imageio_ffmpeg", "modelscope"]
for mod in mods:
    __import__(mod)
    print(f"{mod}: ok")
PY

phase "download_qwen3_omni_instruct"
if ! compgen -G "$LOCAL_MODEL_DIR/*.safetensors" > /dev/null && ! compgen -G "$LOCAL_MODEL_DIR/*.bin" > /dev/null; then
  if command -v modelscope >/dev/null 2>&1; then
    modelscope download --model "$MODEL_ID" --local_dir "$LOCAL_MODEL_DIR"
  else
    "$VENV_PY" -m modelscope download --model "$MODEL_ID" --local_dir "$LOCAL_MODEL_DIR"
  fi
else
  echo "Model weights already present in $LOCAL_MODEL_DIR"
fi

phase "build_manifest"
"$VENV_PY" scripts/omni/build_episode_manifest.py \
  --data-root "$DATA_ROOT" \
  --max-episodes "$MAX_EPISODES" \
  --train-fraction 0.8 \
  --val-fraction 0.0 \
  --test-fraction 0.2 \
  --output "$MANIFEST"

EVAL_SPLIT="$("$VENV_PY" - <<PY
import json
payload = json.load(open("$MANIFEST", "r", encoding="utf-8"))
counts = payload.get("summary", {}).get("split_counts", {})
requested = "$EVAL_SPLIT"
if counts.get(requested, 0):
    print(requested)
elif counts.get("test", 0):
    print("test")
elif counts.get("val", 0):
    print("val")
else:
    print("train")
PY
)"
echo "Using eval split: $EVAL_SPLIT"

phase "export_dataset"
"$VENV_PY" scripts/omni/export_qwen3_omni_action_dataset.py \
  --manifest "$MANIFEST" \
  --run-id "$DATASET_RUN_ID" \
  --max-windows-per-episode "$MAX_WINDOWS_PER_EPISODE" \
  --max-video-frames "$MAX_VIDEO_FRAMES"

DATASET_JSONL="$DATASET_DIR/dataset.jsonl"

phase "qwen_zero_shot_smoke"
"$VENV_PY" scripts/omni/qwen3_omni_inference_smoke.py \
  --dataset-jsonl "$DATASET_JSONL" \
  --model-id "$LOCAL_MODEL_DIR" \
  --split "$EVAL_SPLIT" \
  --sample-limit 3 \
  --run-id "${RUN_ID}_zero_shot" \
  --local-files-only || true

phase "train_8gpu_lora"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
"$VENV_PY" -m accelerate.commands.launch \
  --num_processes 8 \
  --mixed_precision bf16 \
  scripts/omni/train_qwen3_omni_lora.py \
  --dataset-jsonl "$DATASET_JSONL" \
  --model-id "$LOCAL_MODEL_DIR" \
  --run-id "${RUN_ID}_lora" \
  --train-split "$TRAIN_SPLIT" \
  --val-split "$VAL_SPLIT" \
  --epochs "$EPOCHS" \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --max-train-samples 0 \
  --max-val-samples 64 \
  --local-files-only

phase "eval_lora"
"$VENV_PY" scripts/omni/eval_qwen3_omni_lora.py \
  --dataset-jsonl "$DATASET_JSONL" \
  --model-id "$LOCAL_MODEL_DIR" \
  --adapter-dir "$WORKSPACE/checkpoints/${RUN_ID}_lora/adapter_lora" \
  --run-id "${RUN_ID}_eval" \
  --eval-split "$EVAL_SPLIT" \
  --local-files-only

phase "runbook"
"$VENV_PY" scripts/omni/omni_finetune_runbook.py \
  --run-id "$RUN_ID" \
  --manifest "$MANIFEST" \
  --metric-file "$WORKSPACE/results/omni_finetune/${RUN_ID}_eval/metrics.json" || true

phase "complete"
echo "DONE: $RUN_ID"

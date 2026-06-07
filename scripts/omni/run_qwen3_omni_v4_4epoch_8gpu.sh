#!/usr/bin/env bash
set -euo pipefail

# Stronger Qwen3-Omni LoRA continuation over the already exported 128-episode
# 96/16/16 dataset. This launcher intentionally reuses the sealed split and
# writes a distinct run id so it cannot overwrite the public v3 diagnostic.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$PROJECT_DIR"

RUN_ID="${RUN_ID:-xperience10m_qwen3_omni_128ep_structured_json_v4_4epoch_full8gpu_lora}"
DATASET_JSONL="${DATASET_JSONL:-results/omni_finetune/xperience10m_qwen3_omni_128ep_96train_16val_16test_valmon_20260605_dataset/dataset.jsonl}"
MODEL_ID="${MODEL_ID:-$HOME/Ropedia/modelscope_models/Qwen__Qwen3-Omni-30B-A3B-Instruct}"
BACKBONE_CONFIG="${BACKBONE_CONFIG:-configs/omni_backbones/qwen3_omni_lora.json}"
EPOCHS="${EPOCHS:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-512}"

RUN_DIR="results/omni_finetune/${RUN_ID}"
LOG="${RUN_DIR}/train.launch.log"
STATUS="${RUN_DIR}/launch_status.jsonl"
mkdir -p "$RUN_DIR"

json_status() {
  .venv/bin/python - "$STATUS" "$@" <<'PY'
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

if [[ ! -s "$DATASET_JSONL" ]]; then
  json_status event=blocked_missing_dataset dataset_jsonl="$DATASET_JSONL"
  exit 2
fi

if pgrep -af "train_qwen3_omni_lora.py.*--run-id ${RUN_ID}" >/dev/null 2>&1; then
  json_status event=already_running run_id="$RUN_ID"
  pgrep -af "train_qwen3_omni_lora.py.*--run-id ${RUN_ID}"
  exit 0
fi

if pgrep -af "train_qwen3_omni_lora.py" >/dev/null 2>&1; then
  json_status event=blocked_other_training run_id="$RUN_ID"
  pgrep -af "train_qwen3_omni_lora.py"
  exit 3
fi

cmd=(
  .venv/bin/python -m accelerate.commands.launch
  --num_processes 8
  --mixed_precision bf16
  --use_fsdp
  --fsdp_sharding_strategy FULL_SHARD
  --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP
  --fsdp_transformer_layer_cls_to_wrap Qwen3OmniMoeThinkerTextDecoderLayer
  --fsdp_use_orig_params true
  --fsdp_cpu_ram_efficient_loading true
  --fsdp_sync_module_states true
  --fsdp_activation_checkpointing true
  scripts/omni/train_qwen3_omni_lora.py
  --dataset-jsonl "$DATASET_JSONL"
  --model-id "$MODEL_ID"
  --backbone-config "$BACKBONE_CONFIG"
  --run-id "$RUN_ID"
  --train-split train
  --val-split val
  --epochs "$EPOCHS"
  --batch-size 1
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
  --max-train-samples 0
  --max-val-samples "$MAX_VAL_SAMPLES"
  --local-files-only
  --gradient-checkpointing
  --progress-every 10
)

json_status event=launch_start run_id="$RUN_ID" epochs="$EPOCHS" dataset_jsonl="$DATASET_JSONL"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
nohup "${cmd[@]}" > "$LOG" 2>&1 < /dev/null &
pid=$!
sleep 3

if ps -p "$pid" >/dev/null 2>&1; then
  json_status event=launch_detached run_id="$RUN_ID" pid="$pid" log="$LOG"
  echo "launched run_id=${RUN_ID} pid=${pid} log=${LOG}"
  exit 0
fi

json_status event=launch_failed run_id="$RUN_ID" log="$LOG"
tail -120 "$LOG" || true
exit 1

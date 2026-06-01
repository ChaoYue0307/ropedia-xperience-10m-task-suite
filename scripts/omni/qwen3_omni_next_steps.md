# Qwen3-Omni-Instruct Fine-Tuning Notes

This directory separates the Xperience-10M -> Qwen3-Omni plan into two layers.

1. Native Qwen3-Omni inputs: synchronized 2x3 multi-view video mosaic, aligned
   audio clips, and text prompts.
2. Ropedia-specific adapter inputs: depth, pose/SLAM, mocap, contacts, IMU, and
   calibration features.

The v1 objective is embodied episode understanding and question answering, not a
deployable robot-control policy. The default backbone is
`Qwen/Qwen3-Omni-30B-A3B-Instruct`; Thinking is reserved for later inference
comparison. Assistant outputs are strict JSON with these fields:

```json
{
  "action": "unknown",
  "subtask": "unknown",
  "objects": [],
  "contact": "unknown",
  "transition": "unknown",
  "next_action": "unknown",
  "evidence_window": {"start_frame": 0, "end_frame": 0}
}
```

Suggested progression:

1. Phase 0: preflight accelerator runtime, CUDA/PyTorch, free disk, dataset access, ffmpeg,
   HOMIE, and local Qwen3-Omni-Instruct weights.
2. Phase 1: one-episode smoke with adapter-only plus JSONL/media validation.
3. Phase 2: three-episode overfit for adapter-only and Qwen LoRA.
4. Phase 3: 32-episode pilot with held-out episodes and all comparisons.
5. Phase 4: scale to 64 only if the 32-episode run is stable and the sensor
   bridge beats video/audio/text-only LoRA on at least three primary metrics.

Concrete command sequence:

```bash
python scripts/omni/build_episode_manifest.py \
  --data-root /path/to/xperience10m_data \
  --max-episodes 32 \
  --output results/omni_finetune/episode_manifest.json

python scripts/omni/export_qwen3_omni_action_dataset.py \
  --manifest results/omni_finetune/episode_manifest.json \
  --run-id xperience10m_qwen3_omni_32ep_dataset

python scripts/omni/qwen3_omni_inference_smoke.py \
  --dataset-jsonl results/omni_finetune/xperience10m_qwen3_omni_32ep_dataset/dataset.jsonl \
  --split test \
  --sample-limit 3 \
  --run-id xperience10m_qwen3_omni_32ep_zero_shot

python scripts/omni/train_qwen3_omni_lora.py \
  --dataset-jsonl results/omni_finetune/xperience10m_qwen3_omni_32ep_dataset/dataset.jsonl \
  --run-id xperience10m_qwen3_omni_32ep_lora

python scripts/omni/eval_qwen3_omni_lora.py \
  --dataset-jsonl results/omni_finetune/xperience10m_qwen3_omni_32ep_dataset/dataset.jsonl \
  --adapter-dir checkpoints/xperience10m_qwen3_omni_32ep_lora/adapter_lora \
  --run-id xperience10m_qwen3_omni_32ep_eval

python scripts/omni/qwen3_omni_sensor_bridge.py \
  --sensor-adapter-model results/omni_finetune/adapter_only/adapter_only/sensor_adapter_model.pt \
  --qwen-config ../modelscope_models/Qwen__Qwen3-Omni-30B-A3B-Instruct/config.json

python scripts/omni/omni_finetune_runbook.py \
  --run-id xperience10m_qwen3_omni_32ep \
  --metric-file results/omni_finetune/xperience10m_qwen3_omni_32ep_eval/metrics.json
```

The bridge step is intentionally after native Qwen video/audio/text LoRA has
overfit a tiny shard and evaluated on held-out episodes. The full 32-episode
pilot should compare the existing 12-task baseline, adapter-only baseline,
frozen Qwen zero-shot, Qwen LoRA without sensor bridge, and Qwen LoRA with
sensor bridge before any scale-up decision.

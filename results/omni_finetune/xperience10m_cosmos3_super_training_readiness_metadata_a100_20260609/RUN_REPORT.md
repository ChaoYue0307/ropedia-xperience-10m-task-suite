# Cosmos3-Super Training Readiness

- Run id: `xperience10m_cosmos3_super_training_readiness_metadata_a100_20260609`
- Model dir: `/mnt/kgc/chaoyue/ropedia-h20-side/modelscope_models/nv-community__Cosmos3-Super`
- Dataset: `/mnt/kgc/chaoyue/ropedia-h20-side/ropedia-episode-task-suite/results/omni_finetune/xperience10m_qwen3_omni_128ep_96train_16val_16test_valmon_20260605_dataset/dataset.jsonl`
- Samples: `3808`
- Diffusers runtime supported: `False`
- Chat SFT supported: `False`
- Status: `blocked_until_trainer_implemented`
- Weights updated: `False`

## Blockers

- Diffusers runtime is missing Cosmos3 classes: ['Cosmos3OmniPipeline', 'Cosmos3OmniTransformer', 'Cosmos3AVAEAudioTokenizer', 'AutoencoderKLWan', 'UniPCMultistepScheduler']
- Transformers cannot load model_type cosmos3_omni as a local causal-generation model; Qwen-style answer-token CE fine-tuning is unavailable in this environment.

## Next Steps

- Run scripts/omni/audit_cosmos3_super_training_contract.py on the camera-pose action-target JSONL and require no blockers.
- Run scripts/omni/train_cosmos3_super_forward_dynamics_lora.py as a one-sample or one-episode overfit before a full 96/16/16 adapter run.
- Publish a separate Cosmos3-Super model repository only after the trainer produces new adapter/checkpoint weights and held-out evaluation artifacts.

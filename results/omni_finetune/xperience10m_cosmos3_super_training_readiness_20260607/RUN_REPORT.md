# Cosmos3-Super Training Readiness

- Run id: `xperience10m_cosmos3_super_training_readiness_20260607`
- Model dir: `/home/cy/Ropedia/cosmos3_models/nv-community__Cosmos3-Super`
- Dataset: `/home/cy/Ropedia/ropedia-episode-task-suite/results/omni_finetune/xperience10m_qwen3_omni_128ep_96train_16val_16test_valmon_20260605_dataset/dataset.jsonl`
- Samples: `3808`
- Diffusers runtime supported: `True`
- Chat SFT supported: `False`
- Status: `blocked_until_trainer_implemented`
- Weights updated: `False`

## Blockers

- Transformers cannot load model_type cosmos3_omni as a local causal-generation model; Qwen-style answer-token CE fine-tuning is unavailable in this environment.
- Repository has no Cosmos3 diffusion/action target packer or supervised loss implementation for xperience10m_episode_json_qa_v1; a readiness probe cannot produce adapter weights.

## Next Steps

- Implement a Cosmos3-Super training data packer that maps each Xperience-10M window to prompt, video/action latent inputs, timesteps, and loss indexes expected by Cosmos3OmniTransformer.forward.
- Wire LoRA only onto the checkpoint-declared target modules q_proj_moe_gen,k_proj_moe_gen,v_proj_moe_gen,o_proj_moe_gen and use the rectified_flow_training_config loss weights.
- Run a one-episode overfit with --load-pipeline enabled, then a 96/16/16 held-out adapter run only after the probe status has no blockers.

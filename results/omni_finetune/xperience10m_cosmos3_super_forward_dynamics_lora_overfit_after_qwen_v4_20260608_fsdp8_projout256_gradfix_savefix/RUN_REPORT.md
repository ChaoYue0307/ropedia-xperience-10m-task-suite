# Cosmos3-Super Forward-Dynamics LoRA

- Run id: `xperience10m_cosmos3_super_forward_dynamics_lora_overfit_after_qwen_v4_20260608_fsdp8_projout256_gradfix_savefix`
- Status: `complete`
- Weights updated: `True`
- Dataset: `/home/cy/Ropedia/ropedia-episode-task-suite/results/omni_finetune/xperience10m_cosmos3_camera_pose_targets_20260608/dataset_with_cosmos_actions.jsonl`
- Train samples: `1`
- Max steps: `10`
- Final loss: `4.932724952697754`
- Adapter dir: `/home/cy/Ropedia/ropedia-episode-task-suite/results/omni_finetune/xperience10m_cosmos3_super_forward_dynamics_lora_overfit_after_qwen_v4_20260608_fsdp8_projout256_gradfix_savefix/adapter_lora`

## Scope

This adapter trains Cosmos3-Super camera-pose forward dynamics. Raw camera-pose actions are conditioning, and the loss supervises future vision velocity tokens. It is not a JSON Reasoner SFT run and does not supervise `preds_action`.

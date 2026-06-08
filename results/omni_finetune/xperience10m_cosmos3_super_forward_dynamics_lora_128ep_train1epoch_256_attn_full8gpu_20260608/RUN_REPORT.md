# Cosmos3-Super Forward-Dynamics LoRA

- Run id: `xperience10m_cosmos3_super_forward_dynamics_lora_128ep_train1epoch_256_attn_full8gpu_20260608`
- Status: `complete`
- Weights updated: `True`
- Dataset: `/home/cy/Ropedia/ropedia-episode-task-suite/results/omni_finetune/xperience10m_cosmos3_camera_pose_targets_20260608/dataset_with_cosmos_actions.jsonl`
- Train samples: `2848`
- Max steps: `356`
- Final loss: `1.0785235166549683`
- Adapter dir: `/home/cy/Ropedia/ropedia-episode-task-suite/results/omni_finetune/xperience10m_cosmos3_super_forward_dynamics_lora_128ep_train1epoch_256_attn_full8gpu_20260608/adapter_lora`

## Scope

This adapter trains Cosmos3-Super camera-pose forward dynamics. Raw camera-pose actions are conditioning, and the loss supervises future vision velocity tokens. It is not a JSON Reasoner SFT run and does not supervise `preds_action`.

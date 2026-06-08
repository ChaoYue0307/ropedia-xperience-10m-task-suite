# Cosmos3-Super Forward-Dynamics LoRA Result

- Backbone: `cosmos3_super_forward_dynamics`
- Training run: `xperience10m_cosmos3_super_forward_dynamics_lora_128ep_train1epoch_256_attn_full8gpu_20260608`
- Evaluation run: `xperience10m_cosmos3_super_forward_dynamics_lora_128ep_train1epoch_256_attn_full8gpu_20260608_eval_test_full_fsdp`
- Status: `verified`
- Train rows: `2848`
- Val rows: `512`
- Test rows: `448`
- Train final loss: `1.0785235166549683`
- Val forward-dynamics MSE: `4.008244896889664`
- Test forward-dynamics MSE: `3.6853174321087345`
- Adapter parameters: `26214400`

This is a camera-pose proxy forward-dynamics LoRA over Cosmos3-Super. It supervises future vision velocity tokens, not semantic JSON labels.

Raw Xperience-10M media/annotations, base-model weights, LoRA adapter weights, checkpoints, and large archives are not included.

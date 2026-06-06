# Verified Omni Fine-Tuning Result

- Backbone: `qwen3_omni_lora`
- Dataset run: `xperience10m_qwen3_omni_128ep_96train_16val_16test_valmon_20260605`
- Training run: `xperience10m_qwen3_omni_128ep_structured_json_v2_reuse_full8gpu_lora`
- Evaluation run: `xperience10m_qwen3_omni_128ep_structured_json_v2_reuse_full8gpu_lora_eval_test_full`
- Validation status: `verified`
- Held-out eval split: `test`
- Held-out episodes: `14`
- Prediction rows: `448`

## Primary Metrics

- json_validity_rate: `0.9977678571428571`
- action_macro_f1: `0.0024331644885523347`
- subtask_accuracy: `0.002232142857142857`
- transition_accuracy: `0.9709821428571429`
- next_action_accuracy: `0.029017857142857144`
- contact_accuracy: `0.71875`
- object_micro_f1: `0.30160427807486634`
- held_out_episode_count: `14`

Raw Xperience-10M files, base-model weights, adapter or checkpoint weights, full checkpoints, and large archives are not included.

Use this package as the source for README, website, and Hugging Face updates.

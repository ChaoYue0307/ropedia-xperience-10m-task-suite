# Verified Omni Fine-Tuning Result

- Backbone: `qwen3_omni_lora`
- Dataset run: `xperience10m_qwen3_omni_128ep_96train_16val_16test_valmon_20260605`
- Training run: `xperience10m_qwen3_omni_128ep_structured_json_v4_4epoch_full8gpu_lora`
- Evaluation run: `xperience10m_qwen3_omni_128ep_structured_json_v4_4epoch_full8gpu_lora_eval_test_full`
- Validation status: `verified`
- Held-out eval split: `test`
- Held-out episodes: `14`
- Prediction rows: `448`

## Primary Metrics

- json_validity_rate: `1.0`
- action_macro_f1: `0.0018678269676001454`
- subtask_accuracy: `0.0`
- transition_accuracy: `0.9732142857142857`
- next_action_accuracy: `0.033482142857142856`
- contact_accuracy: `0.7299107142857143`
- object_micro_f1: `0.31099781500364165`
- held_out_episode_count: `14`

Raw Xperience-10M files, base-model weights, adapter or checkpoint weights, full checkpoints, and large archives are not included.

Use this package as the source for README, website, and Hugging Face updates.

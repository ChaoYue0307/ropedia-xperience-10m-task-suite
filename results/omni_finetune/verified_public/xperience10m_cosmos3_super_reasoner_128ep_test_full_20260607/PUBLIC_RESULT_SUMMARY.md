# Verified Omni Fine-Tuning Result

- Backbone: `cosmos3_super_reasoner`
- Dataset run: `xperience10m_qwen3_omni_128ep_96train_16val_16test_valmon_20260605`
- Training run: `xperience10m_cosmos3_super_reasoner_base_vllm_8gpu_20260607`
- Evaluation run: `xperience10m_cosmos3_super_reasoner_128ep_test_full_20260607`
- Validation status: `verified`
- Held-out eval split: `test`
- Held-out episodes: `14`
- Prediction rows: `448`

## Primary Metrics

- json_validity_rate: `0.5111607142857143`
- action_macro_f1: `0.0008284021201089245`
- subtask_accuracy: `0.0`
- transition_accuracy: `0.36830357142857145`
- next_action_accuracy: `0.013392857142857142`
- contact_accuracy: `0.32142857142857145`
- object_micro_f1: `0.13704276146316333`
- held_out_episode_count: `14`

Raw Xperience-10M files, base-model weights, adapter or checkpoint weights, full checkpoints, and large archives are not included.

Use this package as the source for README, website, and Hugging Face updates.

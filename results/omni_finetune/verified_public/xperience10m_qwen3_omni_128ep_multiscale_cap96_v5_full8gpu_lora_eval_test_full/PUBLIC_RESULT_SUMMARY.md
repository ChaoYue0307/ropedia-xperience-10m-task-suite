# Verified Omni Fine-Tuning Result

- Backbone: `qwen3_omni_lora`
- Dataset run: `xperience10m_qwen3_omni_128ep_multiscale_cap96_v5_full8gpu_lora`
- Training run: `xperience10m_qwen3_omni_128ep_multiscale_cap96_v5_full8gpu_lora`
- Evaluation run: `xperience10m_qwen3_omni_128ep_multiscale_cap96_v5_full8gpu_lora_eval_test_full`
- Validation status: `verified`
- Held-out eval split: `test`
- Held-out episodes: `14`
- Prediction rows: `4032`

## Primary Metrics

- json_validity_rate: `1.0`
- action_macro_f1: `0.002289711036077459`
- subtask_accuracy: `0.011194029850746268`
- transition_accuracy: `0.9908234126984127`
- next_action_accuracy: `0.053618594823032224`
- contact_accuracy: `0.7864583333333334`
- object_micro_f1: `0.31614599936244814`
- held_out_episode_count: `14`

Raw Xperience-10M files, base-model weights, adapter or checkpoint weights, full checkpoints, and large archives are not included.

Use this package as the source for README, website, and Hugging Face updates.

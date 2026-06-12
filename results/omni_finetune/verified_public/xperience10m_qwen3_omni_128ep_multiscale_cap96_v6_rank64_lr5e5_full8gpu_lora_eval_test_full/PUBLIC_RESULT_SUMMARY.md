# Verified Omni Fine-Tuning Result

- Backbone: `qwen3_omni_lora`
- Dataset run: `xperience10m_qwen3_omni_128ep_multiscale_cap96_v5_full8gpu_lora`
- Training run: `xperience10m_qwen3_omni_128ep_multiscale_cap96_v6_rank64_lr5e5_full8gpu_lora`
- Evaluation run: `xperience10m_qwen3_omni_128ep_multiscale_cap96_v6_rank64_lr5e5_full8gpu_lora_eval_test_full`
- Validation status: `verified`
- Held-out eval split: `test`
- Held-out episodes: `14`
- Prediction rows: `4032`

## Primary Metrics

- json_validity_rate: `0.9990079365079365`
- action_macro_f1: `0.0028830723979596335`
- subtask_accuracy: `0.0037313432835820895`
- transition_accuracy: `0.9898313492063492`
- next_action_accuracy: `0.04305335446381405`
- contact_accuracy: `0.8177083333333334`
- object_micro_f1: `0.3064982378331287`
- held_out_episode_count: `14`

Raw Xperience-10M files, base-model weights, adapter or checkpoint weights, full checkpoints, and large archives are not included.

Use this package as the source for README, website, and Hugging Face updates.

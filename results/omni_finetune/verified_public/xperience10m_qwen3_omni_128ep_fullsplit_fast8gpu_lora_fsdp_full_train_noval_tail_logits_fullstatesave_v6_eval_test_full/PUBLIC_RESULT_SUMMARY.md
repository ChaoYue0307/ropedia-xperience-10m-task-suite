# Verified Omni Fine-Tuning Result

- Backbone: `qwen3_omni_lora`
- Dataset run: `xperience10m_qwen3_omni_128ep_fullsplit_fast8gpu`
- Training run: `xperience10m_qwen3_omni_128ep_fullsplit_fast8gpu_lora_fsdp_full_train_noval_tail_logits_fullstatesave_v6`
- Evaluation run: `xperience10m_qwen3_omni_128ep_fullsplit_fast8gpu_lora_fsdp_full_train_noval_tail_logits_fullstatesave_v6_eval_test_full`
- Validation status: `verified`
- Held-out eval split: `test`
- Held-out episodes: `14`
- Prediction rows: `448`

## Primary Metrics

- json_validity_rate: `0.8526785714285714`
- action_macro_f1: `0.00213753459655099`
- subtask_accuracy: `0.004464285714285714`
- transition_accuracy: `0.828125`
- next_action_accuracy: `0.022321428571428572`
- contact_accuracy: `0.6517857142857143`
- object_micro_f1: `0.23062730627306272`
- held_out_episode_count: `14`

Raw Xperience-10M files, base-model weights, adapter or checkpoint weights, full checkpoints, and large archives are not included.

Use this package as the source for README, website, and Hugging Face updates.

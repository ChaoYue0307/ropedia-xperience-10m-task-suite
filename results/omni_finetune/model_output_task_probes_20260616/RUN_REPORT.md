# Existing Model-Output Task Probes

Generated: `2026-06-18T15:25:22+00:00`

This package scores only task targets already present in verified held-out
prediction JSON. It does not run new inference and does not infer targets that
are absent from a model branch.

| Method | ID | Status | Scored tasks | Task 13 macro-F1 | Task 16 macro-F1 | Task 20 MAE | Evidence |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| Qwen3-Omni v6 LoRA | qwen3_omni_v6_lora | scored | action_object_relation | n/a | 0.000222 | n/a | results/omni_finetune/verified_public/xperience10m_qwen3_omni_128ep_multiscale_cap96_v6_rank64_lr5e5_full8gpu_lora_eval_test_full/eval/predictions.jsonl |
| Cosmos3-Super Reasoner | cosmos3_super_reasoner | scored | action_object_relation, time_to_transition | n/a | 0.000000 | 52.946 | results/omni_finetune/verified_public/xperience10m_cosmos3_super_reasoner_128ep_test_full_20260607/eval/predictions.jsonl |
| Cosmos3-Nano Future Window | cosmos3_nano_future_window | scored | long_horizon_next_action | 0.002491 | n/a | n/a | results/omni_finetune/verified_public/xperience10m_cosmos3_nano_128ep_future_window_h5_compat_adapter_eval_test_full/eval/future_predictions.jsonl |

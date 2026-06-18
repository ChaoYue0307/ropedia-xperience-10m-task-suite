# Existing Model-Output Task Probes

Generated: `2026-06-18T20:58:30+00:00`

This package scores only task targets already present in verified held-out
prediction JSON. It does not run new inference and does not infer targets that
are absent from a model branch.

| Method | ID | Status | Scored tasks | Task 13 macro-F1 | Task 14 macro-F1 | Task 16 macro-F1 | Task 17 micro-F1 | Task 20 MAE | Evidence |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen3-Omni v6 LoRA | qwen3_omni_v6_lora | scored | action_object_relation | n/a | n/a | 0.000222 | n/a | n/a | results/omni_finetune/verified_public/xperience10m_qwen3_omni_128ep_multiscale_cap96_v6_rank64_lr5e5_full8gpu_lora_eval_test_full/eval/predictions.jsonl |
| Cosmos3-Super Reasoner | cosmos3_super_reasoner | scored | action_object_relation, long_horizon_next_action, time_to_transition | 0.008808 | n/a | 0.000000 | n/a | 52.946 | results/omni_finetune/verified_public/xperience10m_cosmos3_super_reasoner_128ep_test_full_20260607/eval/predictions.jsonl |
| Cosmos3-Nano Future Window | cosmos3_nano_future_window | scored | long_horizon_next_action, modality_reconstruction, next_subtask_forecast, object_set_forecast, time_to_transition | 0.002491 | 0.006615 | n/a | 0.017820 | 33.810 | results/omni_finetune/verified_public/xperience10m_cosmos3_nano_128ep_future_window_h5_compat_adapter_eval_test_full/eval/future_predictions.jsonl |

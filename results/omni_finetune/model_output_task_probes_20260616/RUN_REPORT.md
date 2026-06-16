# Existing Model-Output Task Probes

Generated: `2026-06-16T13:35:37+00:00`

This package scores only task targets already present in verified held-out
prediction JSON. It does not run new inference and does not infer targets that
are absent from a model branch.

| Method | ID | Status | Scored rows | Task 16 macro-F1 | Evidence |
| --- | --- | --- | ---: | ---: | --- |
| Qwen3-Omni v6 LoRA | qwen3_omni_v6_lora | scored | 4014 | 0.000222 | results/omni_finetune/verified_public/xperience10m_qwen3_omni_128ep_multiscale_cap96_v6_rank64_lr5e5_full8gpu_lora_eval_test_full/eval/predictions.jsonl |
| Cosmos3-Super Reasoner | cosmos3_super_reasoner | scored | 446 | 0.000000 | results/omni_finetune/verified_public/xperience10m_cosmos3_super_reasoner_128ep_test_full_20260607/eval/predictions.jsonl |
| Cosmos3-Nano Future Window | cosmos3_nano_future_window | unsupported_without_required_fields | n/a | n/a | verified future-window predictions do not contain object-set fields |

# Qwen3-Omni v1-v6 Run Lineage

Generated: `2026-06-21T09:58:19+00:00`.

Scope: Verified public-safe Qwen3-Omni LoRA/eval packages over the selected Xperience-10M 128-episode surface.

Interpretation rule: Do not confuse the Qwen run versions with the project-level public result layers. The 20-task matrix uses Qwen3-Omni v6 LoRA; v5 remains the pinned prior release; v1-v4 are lineage and ablation evidence.

## Compact Lineage

| Version | What changed | Eval samples | JSON validity | Action macro-F1 | Subtask acc. | Contact acc. | Public role |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v1 | Selected-128 validation-aware LoRA baseline | 448 | 0.8750 | 0.0027 | 0.0067 | 0.6451 | superseded lineage evidence, not the current 20-task Qwen row |
| v2 | Structured-JSON reuse full-8-GPU LoRA | 448 | 0.9978 | 0.0024 | 0.0022 | 0.7188 | superseded lineage evidence, not the current 20-task Qwen row |
| v3 | Strict-label prompt evaluation | 448 | 1.0000 | 0.0022 | 0.0022 | 0.7210 | superseded prompt/eval lineage evidence |
| v4 | Four-epoch structured-JSON LoRA | 448 | 1.0000 | 0.0019 | 0.0000 | 0.7299 | superseded lineage evidence, not the current 20-task Qwen row |
| v5 | Multiscale cap96 LoRA | 4032 | 1.0000 | 0.0023 | 0.0112 | 0.7865 | pinned prior release row and comparison baseline |
| v6 | Rank64 lr5e-5 multiscale LoRA | 4032 | 0.9990 | 0.0029 | 0.0037 | 0.8177 | current public 20-task Qwen3-Omni v6 LoRA row |

## Run IDs And Packages

| Version | Train run | Eval run | Role | Package |
| --- | --- | --- | --- | --- |
| v1 | xperience10m_qwen3_omni_128ep_96train_16val_16test_valmon_20260605_lora | xperience10m_qwen3_omni_128ep_96train_16val_16test_valmon_20260605_eval | First verified 96/16/16 selected-episode Qwen3-Omni LoRA package; establishes dataset, training, eval, and packaging plumbing. | results/omni_finetune/verified_public/xperience10m_qwen3_omni_128ep_96train_16val_16test_valmon_20260605_eval |
| v2 | xperience10m_qwen3_omni_128ep_structured_json_v2_reuse_full8gpu_lora | xperience10m_qwen3_omni_128ep_structured_json_v2_reuse_full8gpu_lora_eval_test_full | Reuses the selected-128 split with a stricter structured JSON answer contract and full 8-GPU LoRA training. | results/omni_finetune/verified_public/xperience10m_qwen3_omni_128ep_structured_json_v2_reuse_full8gpu_lora_eval_test_full |
| v3 | xperience10m_qwen3_omni_128ep_structured_json_v2_reuse_full8gpu_lora | xperience10m_qwen3_omni_128ep_structured_json_v3_strict_label_prompt_reuse_lora_eval_test_full | Strict-label prompt/eval pass over the v2 adapter; improves JSON validity without introducing a new adapter training run. | results/omni_finetune/verified_public/xperience10m_qwen3_omni_128ep_structured_json_v3_strict_label_prompt_reuse_lora_eval_test_full |
| v4 | xperience10m_qwen3_omni_128ep_structured_json_v4_4epoch_full8gpu_lora | xperience10m_qwen3_omni_128ep_structured_json_v4_4epoch_full8gpu_lora_eval_test_full | Four-epoch full-8-GPU LoRA run on the same selected split; useful for overfit/metric tradeoff analysis. | results/omni_finetune/verified_public/xperience10m_qwen3_omni_128ep_structured_json_v4_4epoch_full8gpu_lora_eval_test_full |
| v5 | xperience10m_qwen3_omni_128ep_multiscale_cap96_v5_full8gpu_lora | xperience10m_qwen3_omni_128ep_multiscale_cap96_v5_full8gpu_lora_eval_test_full | Dense/multiscale selected-128 run with 4,032 held-out predictions; kept as the pinned prior release because several metrics remain stronger than v6. | results/omni_finetune/verified_public/xperience10m_qwen3_omni_128ep_multiscale_cap96_v5_full8gpu_lora_eval_test_full |
| v6 | xperience10m_qwen3_omni_128ep_multiscale_cap96_v6_rank64_lr5e5_full8gpu_lora | xperience10m_qwen3_omni_128ep_multiscale_cap96_v6_rank64_lr5e5_full8gpu_lora_eval_test_full | Current verified Qwen3-Omni row: rank64/lr5e-5 multiscale LoRA plus task-specific probe artifacts used for the 20/20 Qwen matrix coverage. | results/omni_finetune/verified_public/xperience10m_qwen3_omni_128ep_multiscale_cap96_v6_rank64_lr5e5_full8gpu_lora_eval_test_full |

## Related Engineering Artifacts

| Artifact | Path | Role |
| --- | --- | --- |
| Full-parameter gates | results/omni_finetune/QWEN3_FULL_PARAMETER_GATES_20260609.md | Feasibility and short-train gates; not a public 20-task matrix method row. |
| Alternate fullsplit v6 package | results/omni_finetune/verified_public/xperience10m_qwen3_omni_128ep_fullsplit_fast8gpu_lora_fsdp_full_train_noval_tail_logits_fullstatesave_v6_eval_test_full | Verified alternate no-validation/fullsplit artifact retained for audit, not the current matrix row. |

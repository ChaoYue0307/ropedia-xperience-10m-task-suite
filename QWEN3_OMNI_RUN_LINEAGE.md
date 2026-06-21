# Qwen3-Omni v1-v6 Run Lineage

Generated: `2026-06-21T10:54:46+00:00`.

Scope: Verified public-safe Qwen3-Omni LoRA/eval packages over the selected Xperience-10M 128-episode surface.

Interpretation rule: Do not confuse the Qwen run versions with the project evidence lines. The project evidence lines are one public sample episode and selected 128-episode artifacts. Qwen v1-v6 are only the Qwen3-Omni run lineage inside the selected-128 line. The 20-task matrix uses Qwen3-Omni v6 LoRA; v5 remains the pinned prior release; v1-v4 are lineage and ablation evidence.

Read the versions as an engineering audit trail, not as six separate benchmark
rows. v1-v4 explain how the Qwen3-Omni pipeline was hardened, v5 is the pinned
prior multiscale release, and v6 is the current 20-task Qwen3-Omni row.

## Compact Lineage

| Version | Run | Purpose | Change from previous | Eval samples | JSON validity | Action macro-F1 | Contact acc. | Use now |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v1 | Selected-128 validation-aware LoRA baseline | Prove that the selected-128 split, LoRA training, held-out eval, validation, and public packaging loop works end to end. | First verified Qwen3-Omni selected-128 LoRA run. | 448 | 0.8750 | 0.0027 | 0.6451 | Use only as lineage evidence for the first working pipeline. |
| v2 | Structured-JSON reuse full-8-GPU LoRA | Make the answer format schema-checked and reduce invalid JSON before expanding scale. | Reused the selected-128 split with a stricter structured-JSON answer contract and full 8-GPU LoRA training. | 448 | 0.9978 | 0.0024 | 0.7188 | Use as evidence that schema-constrained evaluation improved validity and contact accuracy over v1. |
| v3 | Strict-label prompt evaluation | Separate prompt/eval formatting effects from adapter-training effects. | Evaluated the v2 adapter with stricter labels and prompts; no new adapter training. | 448 | 1.0000 | 0.0022 | 0.7210 | Use as prompt/eval ablation evidence, not as a separate trained model. |
| v4 | Four-epoch structured-JSON LoRA | Test whether longer structured-JSON LoRA training improves the same selected split. | Trained a new four-epoch full-8-GPU LoRA adapter on the structured-JSON setup. | 448 | 1.0000 | 0.0019 | 0.7299 | Use as overfit and metric-tradeoff evidence before the multiscale export. |
| v5 | Multiscale cap96 LoRA | Move from the 448-sample compact eval to a denser multiscale 4,032-sample held-out eval. | Introduced the multiscale cap96 export and larger held-out evaluation surface. | 4032 | 1.0000 | 0.0023 | 0.7865 | Use as the pinned prior release; it remains stronger on JSON validity, subtask, next-action, object, and transition metrics. |
| v6 | Rank64 lr5e-5 multiscale LoRA | Promote the current public Qwen3-Omni 20-task row with multiscale LoRA plus task-specific probes. | Kept the multiscale setup, changed LoRA rank/lr to rank64/lr5e-5, and added verified task-specific probes for full 20-task coverage. | 4032 | 0.9990 | 0.0029 | 0.8177 | Use as the current public 20-task Qwen row; it improves action macro-F1 and contact accuracy while v5 remains the prior comparator. |

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

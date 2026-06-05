# Qwen3-Omni LoRA Evaluation

- Base model: `<workspace-parent>/modelscope_models/Qwen__Qwen3-Omni-30B-A3B-Instruct`
- Adapter: `<project>/checkpoints/xperience10m_qwen3_omni_128ep_fullsplit_fast8gpu_lora_fsdp_full_train_noval_tail_logits_fullstatesave_v6/adapter_lora`
- Dataset: `<project>/results/omni_finetune/xperience10m_qwen3_omni_128ep_fullsplit_fast8gpu_dataset/dataset.jsonl`
- Eval split: `test`
- Samples: `448`
- Episodes: `14`
- Accuracy: `0.0223`
- Macro-F1: `0.0021`
- Unseen eval labels: `144`

Artifacts include `metrics.json`, `predictions.csv`, `per_class_metrics.csv`, and `confusion_matrix.csv`.

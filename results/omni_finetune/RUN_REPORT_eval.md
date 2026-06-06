# Qwen3-Omni LoRA Evaluation

- Base model: `<workspace-parent>/modelscope_models/Qwen__Qwen3-Omni-30B-A3B-Instruct`
- Adapter: `<project>/checkpoints/xperience10m_qwen3_omni_128ep_96train_16val_16test_valmon_20260605_lora/adapter_lora`
- Dataset: `<project>/results/omni_finetune/xperience10m_qwen3_omni_128ep_96train_16val_16test_valmon_20260605_dataset/dataset.jsonl`
- Eval split: `test`
- Samples: `448`
- Episodes: `14`
- Accuracy: `0.0246`
- Macro-F1: `0.0027`
- Unseen eval labels: `144`

Artifacts include `metrics.json`, `predictions.csv`, `per_class_metrics.csv`, and `confusion_matrix.csv`.

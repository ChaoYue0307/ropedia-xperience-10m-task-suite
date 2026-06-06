# Qwen3-Omni LoRA Evaluation

- Base model: `/home/cy/Ropedia/modelscope_models/Qwen__Qwen3-Omni-30B-A3B-Instruct`
- Adapter: `/home/cy/Ropedia/ropedia-episode-task-suite/checkpoints/xperience10m_qwen3_omni_128ep_structured_json_v2_reuse_full8gpu_lora/adapter_lora`
- Dataset: `results/omni_finetune/xperience10m_qwen3_omni_128ep_96train_16val_16test_valmon_20260605_dataset/dataset.jsonl`
- Eval split: `test`
- Samples: `448`
- Episodes: `14`
- Accuracy: `0.0290`
- Macro-F1: `0.0024`
- Unseen eval labels: `144`

Artifacts include `metrics.json`, `predictions.csv`, `per_class_metrics.csv`, and `confusion_matrix.csv`.

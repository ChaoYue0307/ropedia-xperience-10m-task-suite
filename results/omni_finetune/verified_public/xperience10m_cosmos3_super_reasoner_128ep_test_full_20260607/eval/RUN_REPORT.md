# Cosmos3-Super Reasoner Evaluation

- Model: `cosmos3-super-local`
- API base URL: `http://127.0.0.1:8000/v1`
- Dataset: `results/omni_finetune/xperience10m_qwen3_omni_128ep_96train_16val_16test_valmon_20260605_dataset/dataset.jsonl`
- Eval split: `test`
- Media mode: `video_url`
- Samples: `448`
- Episodes: `14`
- Accuracy: `0.0335`
- Macro-F1: `0.0008`
- JSON validity: `0.5112`

This run uses the staged Cosmos3-Super Reasoner base weights through vLLM. It is an 8-GPU zero-shot/in-context evaluation, not a fine-tuned Cosmos adapter release.

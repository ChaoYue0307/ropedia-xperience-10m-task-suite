# Qwen3-Omni LoRA Training

- Base model: `/path/to/ropedia_workspace/modelscope_models/Qwen__Qwen3-Omni-30B-A3B-Instruct`
- Dataset: `results/omni_finetune/xperience10m_qwen3_omni_32ep_dataset/dataset.jsonl`
- Train samples: `128`
- Validation samples: `0`
- Processes: `8`
- Epochs: `1`
- Final train loss: `10.936364`

Only LoRA parameters are trained; the base Qwen3-Omni weights remain frozen.

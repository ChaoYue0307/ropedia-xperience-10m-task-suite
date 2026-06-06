# Qwen3-Omni LoRA Training

- Base model: `<workspace-parent>/modelscope_models/Qwen__Qwen3-Omni-30B-A3B-Instruct`
- Dataset run: `xperience10m_qwen3_omni_128ep_96train_16val_16test_valmon_20260605`
- Training run: `xperience10m_qwen3_omni_128ep_96train_16val_16test_valmon_20260605_lora`
- Checkpoint: `<project>/checkpoints/xperience10m_qwen3_omni_128ep_96train_16val_16test_valmon_20260605_lora/adapter_lora`
- Processes: `8`
- Train samples: `2848`
- Validation samples: `512`
- Epochs: `1`
- Global step: `356`
- Train loss: `0.413046`
- Validation loss: `0.033066`

This is the validation-aware diagnostic run. Raw Xperience-10M files, base-model weights, and adapter weights are not committed to this repo.

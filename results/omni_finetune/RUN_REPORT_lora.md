# Qwen3-Omni LoRA Training

- Backbone profile: `Qwen3-Omni LoRA`
- Dataset contract: `xperience10m_episode_json_qa_v1`
- Training objective: `structured_episode_understanding_json_qa`
- Base model: `<model-cache>/Qwen__Qwen3-Omni-30B-A3B-Instruct`
- Dataset: `results/omni_finetune/xperience10m_qwen3_omni_128ep_fullsplit_fast8gpu_dataset/dataset.jsonl`
- Train samples: `2848`
- Validation samples: `0`
- Processes: `8`
- Epochs: `1`
- Loss: answer-token cross entropy over supervised JSON tokens
- Logit projection: `assistant-answer tail only`
- Final train loss: `0.412178`

Only LoRA parameters are trained; the base Qwen3-Omni weights remain frozen.

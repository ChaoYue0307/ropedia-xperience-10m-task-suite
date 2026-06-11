# Qwen3-Omni LoRA Training

- Backbone profile: `Qwen3-Omni LoRA`
- Dataset contract: `xperience10m_episode_json_qa_v1`
- Training objective: `structured_episode_understanding_json_qa`
- Base model: `/home/cy/Ropedia/modelscope_models/Qwen__Qwen3-Omni-30B-A3B-Instruct`
- Dataset: `results/omni_finetune/xperience10m_qwen3_omni_128ep_96train_16val_16test_valmon_20260605_dataset/dataset.jsonl`
- Train samples: `512`
- Validation samples: `0`
- Processes: `8`
- Epochs: `8`
- Tuning mode: `full`
- Save mode: `none`
- Optimizer init: `after_model_prepare`
- Loss: answer-token cross entropy over supervised JSON tokens
- Logit projection: `assistant-answer tail only`
- Final train loss: `0.443408`

Full thinker parameters are trainable. This mode is intended for feasibility smokes unless a future sharded full-checkpoint saver is added.

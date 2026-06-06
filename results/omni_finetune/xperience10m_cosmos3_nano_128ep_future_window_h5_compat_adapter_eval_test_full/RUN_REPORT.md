# Cosmos3-Nano Future-Window Compatibility Run

- Dataset: `/home/cy/Ropedia/ropedia-episode-task-suite/results/omni_finetune/xperience10m_cosmos3_nano_128ep_future_window_h5_compat_dataset/dataset.jsonl`
- Train samples: `2403`
- Validation samples: `432`
- Held-out test samples: `378`
- Held-out episodes: `14`
- Future retrieval MRR: `0.022139`
- Future retrieval recall@5: `0.015873`
- Temporal consistency: `0.095238`
- Feature reconstruction error: `3479.218317`

This run validates the Cosmos3-Nano future-window contract on the same selected episode split.
It does not fine-tune or publish Cosmos base weights; full Cosmos diffusion LoRA fine-tuning is the next step after the Cosmos Diffusers training stack is installed.

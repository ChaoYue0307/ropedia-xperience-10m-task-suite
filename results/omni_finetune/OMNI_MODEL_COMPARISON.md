# Omni Model Comparison

Generated: `2026-06-06T23:26:13+00:00`

Compare only rows with the same scope and target. Single-episode raw-feature metrics, 128-episode metadata baselines, Qwen3 structured JSON metrics, and Cosmos3 future-window metrics answer different questions.

## Current Result Versions

| version | status | scope | source |
| --- | --- | --- | --- |
| Single-Episode Public-Sample Task Suite | verified | one public Xperience-10M sample episode | `results/episode_task_suite/summary_report.json` |
| 128-Episode Aligned Simple/NN Baselines | pass | selected 128-episode 96/16/16 split | `results/omni_finetune/multi_episode_128_task_baselines/BASELINE_ALIGNMENT_REPORT.md` |
| 128-Episode Foundation-Model Branches | partial_verified | selected 128-episode split and compatible derived windows | `results/omni_finetune/verified_public/` |

Read the three rows this way:

- Version 1 is the public-sample 12-task harness with minimal and neural heads.
- Version 2 is the selected 128-episode same-split simple/NN baseline alignment.
- Version 3 is the verified model-branch layer: the current final Qwen3-Omni LoRA package is the JSON-task diagnostic result, while Cosmos3-Nano is a future-window compatibility result rather than a full Cosmos diffusion fine-tune.

## 128-Episode Task Baselines

| task | simple | neural |
| --- | ---: | ---: |
| Action Recognition | macro_f1 0.0002 | macro_f1 0.0000 |
| Procedure Step Recognition | macro_f1 0.0000 | macro_f1 0.0000 |
| Action Boundary Detection | macro_f1 0.5220 | macro_f1 0.4582 |
| Next-Action Prediction | macro_f1 0.0002 | macro_f1 0.0000 |
| Hand Trajectory Forecasting | mpjpe |  |
| Contact State Prediction | macro_f1 0.5168 | macro_f1 0.2195 |
| Object Relevance Prediction | micro_f1 0.1822 | micro_f1 0.1054 |
| Language Grounding | mrr 0.0128 |  |
| Cross-Modal Retrieval | mrr |  |
| Cross-Modal Reconstruction | r2 |  |
| Temporal Order Verification | f1 0.3271 |  |
| Multimodal Synchronization Detection | f1 |  |

## Verified Model Branches

| branch | backbone | eval samples | held-out episodes | key metrics |
| --- | --- | ---: | ---: | --- |
| Cosmos3-Nano Future-Window World Model | `cosmos_world_model` | 378 | 14 | future_retrieval_mrr=0.0221, temporal_consistency=0.0952, transition_accuracy=0.9683, contact_accuracy=0.7434 |
| Qwen3-Omni LoRA | `qwen3_omni_lora` | 448 | 14 | json_validity_rate=0.8750, action_macro_f1=0.0027, transition_accuracy=0.8504, contact_accuracy=0.6451 |
| Qwen3-Omni LoRA | `qwen3_omni_lora` | 448 | 14 | json_validity_rate=0.8527, action_macro_f1=0.0021, transition_accuracy=0.8281, contact_accuracy=0.6518 |
| Qwen3-Omni LoRA | `qwen3_omni_lora` | 448 | 14 | json_validity_rate=0.9978, action_macro_f1=0.0024, transition_accuracy=0.9710, contact_accuracy=0.7188 |

## Pending

- Use the final Qwen3 full-eval package as the current Qwen result; older Qwen package rows remain historical diagnostics for comparison.
- Promote Cosmos3 from compatibility adapter to full Cosmos3 fine-tuning only after a separate environment with matching Diffusers/Cosmos dependencies is prepared.

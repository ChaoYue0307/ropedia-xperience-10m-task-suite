# Omni Model Comparison

Generated: `2026-06-07T09:05:41+00:00`

Compare only rows with the same scope and target. Single-episode raw-feature metrics, 128-episode metadata baselines, Qwen3 structured JSON metrics, and the two Cosmos3 targets answer different questions: Nano future-window retrieval versus Super structured JSON Reasoner evaluation.

## Current Result Versions

| version | status | scope | source |
| --- | --- | --- | --- |
| Single-Episode Public-Sample Task Suite | verified | one public Xperience-10M sample episode | `results/episode_task_suite/summary_report.json` |
| 128-Episode Aligned Simple/NN Baselines | pass | selected 128-episode 96/16/16 split | `results/omni_finetune/multi_episode_128_task_baselines/BASELINE_ALIGNMENT_REPORT.md` |
| 128-Episode Foundation-Model Branches | partial_verified | selected 128-episode split and compatible derived windows | `results/omni_finetune/verified_public/` |

Read the three rows this way:

- Version 1 is the public-sample 12-task harness with minimal and neural heads.
- Version 2 is the selected 128-episode same-split simple/NN baseline alignment.
- Version 3 is the verified model-branch layer: the current final Qwen3-Omni LoRA package is the JSON-task diagnostic result, Cosmos3-Nano is a future-window compatibility result, and Cosmos3-Super Reasoner is a base-weight JSON-task evaluation rather than a new fine-tuned weight release.

## Model-Family Grouped View

- Use model_groups when comparing one-episode and 128-episode artifacts within the same model family.
- Task-head baselines have both a one-episode public-sample run and a 128-episode same-split metadata/text run.
- Qwen3-Omni has a one-episode sensor-adapter smoke test and separate 128-episode LoRA diagnostic packages; only the final 128-episode adapter belongs in the Qwen LoRA model repo.
- Cosmos3-Nano has a 128-episode future-window compatibility package.
- Cosmos3-Super has a 128-episode base-weight Reasoner evaluation on the JSON task; create a separate Cosmos model repo only after real Cosmos adapter/fine-tuned weights exist.

### Minimal and Neural Task Heads

This is the cleanest 1-episode versus 128-episode grouping for the same simple/NN task-head family, but the feature surface changes from raw public-sample features to public-safe 128-episode metadata/text features.

- Weight repo policy: https://huggingface.co/cy0307/ropedia-xperience-10m-task-baselines

| scope | status | run | counts | metrics | source |
| --- | --- | --- | --- | --- | --- |
| 1 episode | verified | Single-Episode Public-Sample Task Suite | 1 episodes, 1161 windows/samples |  | `results/episode_task_suite/summary_report.json` |
| 128 episode | pass | 128-Episode Aligned Simple/NN Baselines | 3808 windows/samples |  | `results/omni_finetune/multi_episode_128_task_baselines/BASELINE_ALIGNMENT_REPORT.md` |

### Qwen3-Omni LoRA

The one-episode Qwen entry is only a sensor-adapter smoke test with Qwen3 weights unloaded. The 128-episode entries are real held-out LoRA diagnostics; the current final adapter belongs in the separate Qwen model repo.

- Weight repo policy: https://huggingface.co/cy0307/ropedia-qwen3-omni-lora-128ep

| scope | status | run | counts | metrics | source |
| --- | --- | --- | --- | --- | --- |
| 1 episode | verified_smoke | Qwen3-Omni Sensor-Adapter Smoke | 1 episodes, 59 windows/samples | accuracy=0.0000, macro_f1=0.0000 | `results/omni_exploration/qwen3_adapter_smoke/metrics.json` |
| 128 episode | verified | Qwen3-Omni LoRA | 119 episodes, 3808 windows/samples, 448 eval | json_validity_rate=0.8750, action_macro_f1=0.0027, transition_accuracy=0.8504, contact_accuracy=0.6451 | `results/omni_finetune/verified_public/xperience10m_qwen3_omni_128ep_96train_16val_16test_valmon_20260605_eval/verified_result_summary.json` |
| 128 episode | verified | Qwen3-Omni LoRA | 119 episodes, 3808 windows/samples, 448 eval | json_validity_rate=0.8527, action_macro_f1=0.0021, transition_accuracy=0.8281, contact_accuracy=0.6518 | `results/omni_finetune/verified_public/xperience10m_qwen3_omni_128ep_fullsplit_fast8gpu_lora_fsdp_full_train_noval_tail_logits_fullstatesave_v6_eval_test_full/verified_result_summary.json` |
| 128 episode | verified current | Qwen3-Omni LoRA | 119 episodes, 3808 windows/samples, 448 eval | json_validity_rate=0.9978, action_macro_f1=0.0024, transition_accuracy=0.9710, contact_accuracy=0.7188 | `results/omni_finetune/verified_public/xperience10m_qwen3_omni_128ep_structured_json_v2_reuse_full8gpu_lora_eval_test_full/verified_result_summary.json` |

### Cosmos3-Nano Future-Window World Model

The current 128-episode Cosmos result is a public-safe future-window compatibility adapter. It is not yet a full Cosmos diffusion/LoRA weight release.

- Weight repo policy: planned: cy0307/ropedia-cosmos3-nano-future-window-lora-128ep after real adapter weights exist

| scope | status | run | counts | metrics | source |
| --- | --- | --- | --- | --- | --- |
| 1 episode | not_run | Cosmos3-Nano One-Episode Fine-Tune |  |  |  |
| 128 episode | verified current | Cosmos3-Nano Future-Window World Model | 119 episodes, 3213 windows/samples, 378 eval | future_retrieval_mrr=0.0221, temporal_consistency=0.0952, transition_accuracy=0.9683, contact_accuracy=0.7434 | `results/omni_finetune/verified_public/xperience10m_cosmos3_nano_128ep_future_window_h5_compat_adapter_eval_test_full/verified_result_summary.json` |

### Cosmos3-Super Reasoner

Cosmos3-Super is now represented by a verified 448-window held-out Reasoner evaluation on the same JSON task as Qwen3. It uses staged base weights through vLLM, so it is a model-branch diagnostic, not a weight release.

- Weight repo policy: none for this run; staged base weights only, no new fine-tuned weights

| scope | status | run | counts | metrics | source |
| --- | --- | --- | --- | --- | --- |
| 1 episode | not_run | Cosmos3-Super One-Episode Fine-Tune |  |  |  |
| 128 episode | verified current | Cosmos3-Super Reasoner | 119 episodes, 3808 windows/samples, 448 eval | json_validity_rate=0.5112, action_macro_f1=0.0008, transition_accuracy=0.3683, contact_accuracy=0.3214 | `results/omni_finetune/verified_public/xperience10m_cosmos3_super_reasoner_128ep_test_full_20260607/verified_result_summary.json` |

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
| Cosmos3-Super Reasoner | `cosmos3_super_reasoner` | 448 | 14 | json_validity_rate=0.5112, action_macro_f1=0.0008, transition_accuracy=0.3683, contact_accuracy=0.3214 |
| Qwen3-Omni LoRA | `qwen3_omni_lora` | 448 | 14 | json_validity_rate=0.8750, action_macro_f1=0.0027, transition_accuracy=0.8504, contact_accuracy=0.6451 |
| Qwen3-Omni LoRA | `qwen3_omni_lora` | 448 | 14 | json_validity_rate=0.8527, action_macro_f1=0.0021, transition_accuracy=0.8281, contact_accuracy=0.6518 |
| Qwen3-Omni LoRA | `qwen3_omni_lora` | 448 | 14 | json_validity_rate=0.9978, action_macro_f1=0.0024, transition_accuracy=0.9710, contact_accuracy=0.7188 |

## Pending

- Use the final Qwen3 full-eval package as the current Qwen result; older Qwen package rows remain historical diagnostics for comparison.
- Promote Cosmos3 from Nano compatibility and Super base-weight evaluation to true fine-tuning only after a dedicated Cosmos adapter/diffusion training path produces new weights.

# Research Takeaways

This generated note summarizes what the current public Xperience-10M sample
pipeline actually shows. It is built from committed metric artifacts, not
from hand-edited score text.

## Scope

- validated episodes: 1
- frames: 5,821
- aligned windows: 1,161
- current feature dimension: 8,546
- raw Xperience-10M data is not redistributed
- Audio from the sample MP4 stream is represented in the current feature vector

## Takeaways

### One episode can become a real benchmark contract

The public sample is converted into 5,821 frames, 1,161 aligned 20-frame windows, and an 8,546-dimensional feature contract.

| Metric | Value |
| --- | ---: |
| `frames` | 5,821 |
| `windows` | 1,161 |
| `feature_dim` | 8,546 |

Source: `docs/data/summary_metrics.json`.

Current scope: This benchmark defines the task contract; cross-episode generalization is evaluated in the multi-episode stage.

### Chronological splits expose action-class shift

Earlier all-feature action classifiers reach high macro-F1 on their local split, but the 12-task chronological action/subtask heads are much harder because later held-out windows include unseen labels.

| Metric | Value |
| --- | ---: |
| `all_feature_action_macro_f1` | 0.9829 |
| `suite_action_macro_f1` | 0.0500 |
| `suite_subtask_macro_f1` | 0.0506 |
| `unseen_action_test_classes` | 4 |

Source: `results/episode_task_suite/summary_report.json`.

Current scope: This split is useful for studying label shift; broad action-recognition conclusions need held-out episodes.

### Small neural heads help dynamic and temporal probes

The MLP heads substantially improve hand trajectory forecasting, temporal-order verification, and motion/visual synchronization.

| Metric | Value |
| --- | ---: |
| `hand_mpjpe_minimal` | 0.8647 |
| `hand_mpjpe_neural` | 0.1079 |
| `hand_mpjpe_relative_improvement` | 0.8753 |
| `temporal_order_f1_minimal` | 0.5400 |
| `temporal_order_f1_neural` | 0.8520 |
| `misalignment_f1_minimal` | 0.5052 |
| `misalignment_f1_neural` | 0.7153 |

Source: `results/episode_task_suite/neural_mlp/*/metrics.json`.

Current scope: These gains are measured within one episode and are candidates for held-out-episode testing.

### Retrieval and reconstruction remain the harder multimodal problems

Ridge/cosine retrieval remains stronger than the neural projection on this sample, and cross-modal reconstruction still has negative R2.

| Metric | Value |
| --- | ---: |
| `retrieval_mrr_minimal` | 0.2693 |
| `retrieval_mrr_neural` | 0.1300 |
| `retrieval_top5_minimal` | 0.3678 |
| `reconstruction_r2_minimal` | -0.0153 |
| `reconstruction_r2_neural` | -0.0102 |

Source: `results/episode_task_suite/cross_modal_retrieval/metrics.json`.

Current scope: The current reconstruction task predicts feature vectors; depth, mesh, NeRF, and Gaussian-splatting outputs are future task variants.

### Audio helps some tasks and hurts others on the public sample

Audio improves the primary metric on 6 of 12 tasks, while raw log-mel replacement improves over the current handcrafted block on 6 of 12 tasks. The largest current-audio gain appears in feature reconstruction, not in action classification.

| Metric | Value |
| --- | ---: |
| `tasks_where_current_audio_improves` | 6 |
| `mean_current_audio_delta` | 0.0418 |
| `tasks_where_raw_replacement_improves` | 6 |
| `mean_raw_replacement_delta_vs_current` | 0.0936 |
| `reconstruction_current_audio_delta` | 0.6524 |
| `object_relevance_current_audio_delta` | 0.0102 |

Source: `results/audio_ablation/audio_ablation_summary.json`.

Current scope: This is a single-episode ablation over fixed ridge heads. It validates that audio is wired into the task suite and shows where it changes metrics; it does not prove cross-episode audio generalization.

### The next scientific unit is held-out episodes, not more adjacent windows

The prepared Qwen3-Omni path now targets a selected 128-episode pilot; held-out metrics will be reported after staging, training, and evaluation complete.

| Metric | Value |
| --- | ---: |
| `target_episodes` | 128 |
| `selected_sessions` | 128 |
| `valid_candidates` | 12,102 |

Source: `results/omni_finetune/MULTI_EPISODE_ACCESS_STATUS.md`.

Current scope: The selected-episode Qwen3-Omni fine-tune requires completed data staging and held-out evaluation.

## How To Read These Results

- High single-episode scores are useful pipeline checks for the current task contracts.
- Low chronological action/subtask scores are informative because they expose later-label shift.
- Neural gains on trajectory/order/alignment make those tasks good candidates for the next fine-tuning stage.
- Audio ablation is task-specific: audio representation choices help some probes and hurt others.
- Retrieval and reconstruction remain the main multimodal representation challenges.
- The next credible model-quality result needs held-out episodes.

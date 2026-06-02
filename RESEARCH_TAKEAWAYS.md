# Research Takeaways

This generated note summarizes what the current public Xperience-10M sample
pipeline actually shows. It is built from committed metric artifacts, not
from hand-entered benchmark claims.

## Scope

- validated episodes: 1
- frames: 5,821
- aligned windows: 1,161
- current feature dimension: 8,378
- raw Xperience-10M data is not redistributed
- audio is documented and visualized, but not yet featurized

## Takeaways

### One episode can become a real benchmark contract

The public sample is converted into 5,821 frames, 1,161 aligned 20-frame windows, and an 8,378-dimensional feature contract.

| Metric | Value |
| --- | ---: |
| `frames` | 5,821 |
| `windows` | 1,161 |
| `feature_dim` | 8,378 |

Source: `docs/data/summary_metrics.json`.

Boundary: This is a task-development benchmark, not cross-episode generalization.

### Chronological splits expose action-class shift

Earlier all-feature action classifiers reach high macro-F1 on their local split, but the 12-task chronological action/subtask heads are much harder because later held-out windows include unseen labels.

| Metric | Value |
| --- | ---: |
| `all_feature_action_macro_f1` | 0.9791 |
| `suite_action_macro_f1` | 0.0500 |
| `suite_subtask_macro_f1` | 0.0495 |
| `unseen_action_test_classes` | 4 |

Source: `results/episode_task_suite/summary_report.json`.

Boundary: This is an important leakage/split lesson, not evidence that action recognition is solved.

### Small neural heads help dynamic and temporal probes

The MLP heads substantially improve hand trajectory forecasting, temporal-order verification, and motion/visual synchronization.

| Metric | Value |
| --- | ---: |
| `hand_mpjpe_minimal` | 0.8223 |
| `hand_mpjpe_neural` | 0.1116 |
| `hand_mpjpe_relative_improvement` | 0.8642 |
| `temporal_order_f1_minimal` | 0.5487 |
| `temporal_order_f1_neural` | 0.8718 |
| `misalignment_f1_minimal` | 0.4866 |
| `misalignment_f1_neural` | 0.7335 |

Source: `results/episode_task_suite/neural_mlp/*/metrics.json`.

Boundary: These gains are within one episode and should be re-tested on held-out episodes.

### Retrieval and reconstruction remain the harder multimodal problems

Ridge/cosine retrieval remains stronger than the neural projection on this sample, and cross-modal reconstruction still has negative R2.

| Metric | Value |
| --- | ---: |
| `retrieval_mrr_minimal` | 0.2634 |
| `retrieval_mrr_neural` | 0.1530 |
| `retrieval_top5_minimal` | 0.3764 |
| `reconstruction_r2_minimal` | -0.0160 |
| `reconstruction_r2_neural` | -0.0102 |

Source: `results/episode_task_suite/cross_modal_retrieval/metrics.json`.

Boundary: The current reconstruction task is feature-vector reconstruction, not depth, mesh, NeRF, or Gaussian splatting.

### The next scientific unit is held-out episodes, not more adjacent windows

The prepared Qwen3-Omni path targets 32 episodes from 32 sessions, but it remains data-gated until access and held-out evaluation complete.

| Metric | Value |
| --- | ---: |
| `target_episodes` | 32 |
| `selected_sessions` | 32 |
| `valid_candidates` | 680 |

Source: `results/omni_finetune/MULTI_EPISODE_ACCESS_STATUS.md`.

Boundary: No real 32-episode fine-tune is claimed until gated data is available locally and held-out evaluation runs.

## How To Read These Results

- High single-episode scores are useful pipeline checks, not broad embodied-AI claims.
- Low chronological action/subtask scores are informative because they expose later-label shift.
- Neural gains on trajectory/order/alignment make those tasks good candidates for the next fine-tuning stage.
- Retrieval and reconstruction remain the main multimodal representation challenges.
- The next credible model-quality result needs held-out episodes.

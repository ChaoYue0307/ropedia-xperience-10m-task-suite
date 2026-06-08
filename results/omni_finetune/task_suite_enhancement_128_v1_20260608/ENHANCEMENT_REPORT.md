# 128-Episode Task Suite Enhancement Pack

Run id: `task_suite_enhancement_128_v1_20260608`

This non-overwriting enhancement pack records how to push the current 128-episode task suite harder without adding more raw episodes.

## Current Evidence

- Current public export windows: `3808`
- Window split counts: `train 2848 / val 512 / test 448`
- Selected episode split: `train 96 / val 16 / test 16`
- Windowed episode ids in baseline CSV: `train 89 / val 16 / test 14`
- Qwen3 v4 JSON validity: `1.0000`
- Qwen3 v4 action macro-F1: `0.001868`
- Qwen3 v4 subtask accuracy: `0.000000`
- Qwen3 v4 unseen-label sample share: `0.7076`

## Dense-Window Scenarios

| scenario | estimated windows | multiplier | role |
| --- | ---: | ---: | --- |
| `current_export` | 3808 | 1.0 | current public 128-episode JSON-task export |
| `dense_20f_stride20` | 30422 | 7.99 | non-overlap dense coverage over each observed episode frame span |
| `dense_20f_stride10` | 60725 | 15.95 | 2x overlap action/subtask densification |
| `dense_20f_stride5` | 121331 | 31.86 | high-overlap action boundary and transition stress setting |
| `medium_40f_stride20` | 30303 | 7.96 | subtask/procedure context window |
| `long_80f_stride40` | 15067 | 3.96 | procedure and world-model context window |
| `multiscale_20s10_40s20_80s40` | 106095 | 27.86 | recommended no-new-episode v5 export: short action windows plus medium/long procedure context |

## Highest-Priority Bottlenecks

| task | priority | simple score | bottleneck | next action |
| --- | --- | ---: | --- | --- |
| Next-Action Prediction | highest | 0.000200 | fine-grained label explosion and held-out unseen labels | add hierarchical action/subtask families plus label-normalized scoring |
| Action Recognition | highest | 0.000175 | fine-grained label explosion and held-out unseen labels | add hierarchical action/subtask families plus label-normalized scoring |
| Procedure Step Recognition | highest | 0.000000 | fine-grained label explosion and held-out unseen labels | add hierarchical action/subtask families plus label-normalized scoring |
| Cross-Modal Retrieval | high |  | missing raw 128-episode feature blocks | export compact raw-feature shards for this task before model comparison |
| Hand Trajectory Forecasting | high |  | missing raw 128-episode feature blocks | export compact raw-feature shards for this task before model comparison |
| Multimodal Synchronization Detection | high |  | missing raw 128-episode feature blocks | export compact raw-feature shards for this task before model comparison |
| Cross-Modal Reconstruction | high |  | missing raw 128-episode feature blocks | export compact raw-feature shards for this task before model comparison |
| Language Grounding | medium | 0.012786 | weak public-safe metadata/text baseline | add dense windows and stronger fusion baselines before interpreting model quality |

## Recommended Next Run

Use `multiscale_20s10_40s20_80s40` as the next export target, then train a Qwen3 v5 hierarchical-target LoRA/partial-unfreeze run against the unchanged 96/16/16 episode split.

In parallel, export compact raw 128-episode feature shards for trajectory, retrieval, reconstruction, and synchronization tasks so the simple and neural baselines can be fully aligned beyond the JSON-supported labels.

The current artifacts remain the baseline; future runs should write new run ids and publish separate verified packages.

# Evaluation Protocol

This file defines how the public Xperience-10M sample episode is turned
into benchmark-style tasks, how the baselines are evaluated, and what the
reported metrics are allowed to mean.

## Protocol At A Glance

| Item | Current protocol |
| --- | --- |
| Source scope | 1 public Xperience-10M sample episode |
| Frames | 5,821 |
| Sliding windows | 1,161 windows, 20 frames each, stride 5 frames |
| Current feature vector | 8,546 dimensions |
| Split | chronological 70/30 train/test by time |
| Baselines | minimal interpretable heads plus compact neural MLP heads |
| Audio | AAC stream extracted from the sample MP4 and included in the current baseline vector |
| Raw data | not redistributed |

## Data Unit

The basic unit is a 20-frame aligned window built from one synchronized
public episode. Feature blocks are documented in
`results/episode_task_suite/feature_manifest.json`; the committed window
table is `results/episode_task_suite/windows.csv`.

## Split Policy

The current suite uses `single_episode_chronological`: The split preserves time order so future episode segments are not mixed randomly into the train set. It is still one episode; cross-episode generalization is evaluated in the multi-episode stage.

This makes some classification metrics intentionally harsh: later test
segments can contain action or subtask labels not present in the train
segment. Those cases are recorded in the task metrics as `unseen_test_classes`.

## Feature And Head Policy

- Input contract: 8,546-dimensional current feature vector.
- Source manifest: `results/episode_task_suite/feature_manifest.json`.
- Normalization: Scalers are fit on train windows only for the baseline heads.
- Audio status: Audio is represented in the current feature vector.

Minimal heads are used first because they make task contracts easy to inspect.
Neural MLP heads reuse the same windows, splits, and feature tensors; they
are not foundation models.

## Task Contracts

| Task | Family | Unit | Input -> target | Primary metric | Minimal | Neural |
| --- | --- | --- | --- | --- | ---: | ---: |
| timeline_action | supervised classification | single window | current 20-frame all-feature window -> current action label | macro_f1 (higher better) | 0.0500 | 0.0148 |
| timeline_subtask | supervised classification | single window | current 20-frame all-feature window -> current subtask label | macro_f1 (higher better) | 0.0506 | 0.0281 |
| transition_detection | temporal diagnostic | single window | current 20-frame all-feature window -> action boundary versus steady | macro_f1 (higher better) | 0.6118 | 0.5862 |
| next_action | short-horizon prediction | single window | current 20-frame all-feature window at time t -> action label at t + 20 frames | macro_f1 (higher better) | 0.0593 | 0.0419 |
| hand_trajectory_forecast | trajectory regression | single window | current all-feature window -> future left/right hand 3D joints for 10 frames | mpjpe (lower better) | 0.8647 | 0.1079 |
| contact_prediction | binary classification | single window | non-contact and non-caption feature blocks -> any body contact | macro_f1 (higher better) | 1.0000 | 1.0000 |
| object_relevance | multi-label classification | single window | non-caption feature blocks -> current relevant object set | micro_f1 (higher better) | 0.1803 | 0.1679 |
| caption_grounding | retrieval | caption query | caption object/interaction query plus candidate sensor windows -> matching time window | mrr (higher better) | 0.0160 | 0.0168 |
| cross_modal_retrieval | retrieval | sensor query | motion, IMU, and camera query features -> matching depth/video window | top5_accuracy (higher better) | 0.3678 | 0.1983 |
| modality_reconstruction | cross-modal regression | single window | motion, IMU, and camera features -> depth/video feature vector | r2 (higher better) | -0.0153 | -0.0102 |
| temporal_order | pairwise diagnostic | adjacent window pair | two adjacent windows -> correct versus reversed order | f1 (higher better) | 0.5400 | 0.8520 |
| misalignment_detection | pairwise diagnostic | paired modality window | motion side plus visual/depth side -> aligned versus shifted by 8 windows | f1 (higher better) | 0.5052 | 0.7153 |

## Leakage Controls

- Use chronological train/test splits instead of random window shuffling.
- Fit scalers and learned projections on train windows only.
- Keep future labels, future mocap, contact labels, object labels, and caption labels on the target side unless a task explicitly treats language as the query.
- For cross-modal tasks, split query-side and candidate-side feature blocks before training and ranking.
- Report unseen test classes when the chronological split exposes labels absent from the train segment.

## Current Limitations

- Cross-episode generalization for Qwen3-Omni has a first verified diagnostic pilot, but strong model quality is not yet shown.
- Feature-vector reconstruction is separate from pixel depth, mesh, NeRF, or Gaussian reconstruction.
- The verified Qwen3-Omni diagnostic pilot has weak held-out metrics and needs validation-aware rerunning before larger model-quality claims.
- Full audio-visual representation learning still needs multi-episode training; the current report includes single-episode audio/no-audio ablations.

## Scale-Up Gate

The next Qwen3-Omni quality pilot requires all of the following before
claiming improved held-out model quality:

- selected prepared Xperience-10M episodes
- held-out episode split with no train/test episode leakage
- nonzero validation samples during training
- manifest, training metadata, progress logs, metrics, predictions, and run report
- held-out evaluation on test episodes rather than train windows

Current status: verified diagnostic pilot; quality target not met. Read
`docs/data/omni_finetune_verified_result.json` before interpreting any
Qwen3-Omni metric.

## Machine-Readable Copy

The JSON mirror is `docs/data/evaluation_protocol.json`.

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
| Current feature vector | 8,378 dimensions |
| Split | chronological 70/30 train/test by time |
| Baselines | minimal interpretable heads plus compact neural MLP heads |
| Audio | present in MP4 streams and visualized, but not featurized in the current baseline vector |
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

- Input contract: 8,378-dimensional current feature vector.
- Source manifest: `results/episode_task_suite/feature_manifest.json`.
- Normalization: Scalers are fit on train windows only for the baseline heads.
- Audio status: Audio is present in sample MP4 streams and visualized in the atlas, but not extracted into the current 8,378-d feature vector.

Minimal heads are used first because they make task contracts debuggable.
Neural MLP heads reuse the same windows, splits, and feature tensors; they
are not foundation models.

## Task Contracts

| Task | Family | Unit | Input -> target | Primary metric | Minimal | Neural |
| --- | --- | --- | --- | --- | ---: | ---: |
| timeline_action | supervised classification | single window | current 20-frame all-feature window -> current action label | macro_f1 (higher better) | 0.0500 | 0.0263 |
| timeline_subtask | supervised classification | single window | current 20-frame all-feature window -> current subtask label | macro_f1 (higher better) | 0.0495 | 0.0175 |
| transition_detection | temporal diagnostic | single window | current 20-frame all-feature window -> action boundary versus steady | macro_f1 (higher better) | 0.6552 | 0.6485 |
| next_action | short-horizon prediction | single window | current 20-frame all-feature window at time t -> action label at t + 20 frames | macro_f1 (higher better) | 0.0593 | 0.0235 |
| hand_trajectory_forecast | trajectory regression | single window | current all-feature window -> future left/right hand 3D joints for 10 frames | mpjpe (lower better) | 0.8223 | 0.1116 |
| contact_prediction | binary classification | single window | non-contact and non-caption feature blocks -> any body contact | macro_f1 (higher better) | 1.0000 | 1.0000 |
| object_relevance | multi-label classification | single window | non-caption feature blocks -> current relevant object set | micro_f1 (higher better) | 0.1839 | 0.1798 |
| caption_grounding | retrieval | caption query | caption object/interaction query plus candidate sensor windows -> matching time window | mrr (higher better) | 0.0172 | 0.0178 |
| cross_modal_retrieval | retrieval | sensor query | motion, IMU, and camera query features -> matching depth/video window | top5_accuracy (higher better) | 0.3764 | 0.2155 |
| modality_reconstruction | cross-modal regression | single window | motion, IMU, and camera features -> depth/video feature vector | r2 (higher better) | -0.0160 | -0.0102 |
| temporal_order | pairwise diagnostic | adjacent window pair | two adjacent windows -> correct versus reversed order | f1 (higher better) | 0.5487 | 0.8718 |
| misalignment_detection | pairwise diagnostic | paired modality window | motion side plus visual/depth side -> aligned versus shifted by 8 windows | f1 (higher better) | 0.4866 | 0.7335 |

## Leakage Controls

- Use chronological train/test splits instead of random window shuffling.
- Fit scalers and learned projections on train windows only.
- Keep future labels, future mocap, contact labels, object labels, and caption labels on the target side unless a task explicitly treats language as the query.
- For cross-modal tasks, split query-side and candidate-side feature blocks before training and ranking.
- Report unseen test classes when the chronological split exposes labels absent from the train segment.

## Current Limitations

- Cross-episode generalization is evaluated in the later multi-episode stage.
- Feature-vector reconstruction is separate from pixel depth, mesh, NeRF, or Gaussian reconstruction.
- Qwen3-Omni setup artifacts are preparation artifacts until the 32-episode held-out pilot runs.
- Audio-visual learning needs an extracted audio feature block; audio is documented and visualized but not featurized in the current baseline vector.

## Scale-Up Gate

The full Qwen3-Omni fine-tuning pilot requires all of the following before
reporting held-out model metrics:

- at least 32 valid Xperience-10M episodes
- held-out episode split with no train/test episode leakage
- manifest, training metadata, progress logs, metrics, predictions, and run report
- held-out evaluation on test episodes rather than train windows

Current status: prepared but data-gated. Read
`results/omni_finetune/DATA_BLOCKER_REPORT.md` and
`results/omni_finetune/MULTI_EPISODE_ACCESS_STATUS.md` before interpreting any
Qwen3-Omni artifact.

## Machine-Readable Copy

The JSON mirror is `docs/data/evaluation_protocol.json`.

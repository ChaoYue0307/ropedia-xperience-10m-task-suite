# Episode Task Suite

Script:

```text
scripts/episode_task_suite.py
```

This script turns the single public Ropedia sample episode into many end-to-end tasks. It is designed for learning, debugging, and task design. It is **not** a generalization benchmark because the data is still one episode.

Run:

```bash
cd /path/to/Ropedia
source .venv/bin/activate
python scripts/episode_task_suite.py
```

Output:

```text
outputs/episode_task_suite/
```

Shared setup:

```text
sample episode: 5821 frames
windows:        1161
window size:    20 frames
stride:         5 frames
feature dim:    8378
split:          chronological, first 70% train and last 30% test
```

## Implemented Tasks

| Task | Input | Output | Main artifact |
|---|---|---|---|
| `timeline_action` | all modality window | current action label | `timeline_action/metrics.json` |
| `timeline_subtask` | all modality window | current subtask label | `timeline_subtask/metrics.json` |
| `transition_detection` | all modality window | steady vs action boundary | `transition_detection/metrics.json` |
| `next_action` | current all modality window | action 20 frames later | `next_action/metrics.json` |
| `hand_trajectory_forecast` | current all modality window | future 10-frame left/right hand joints | `hand_trajectory_forecast/predictions.npz` |
| `contact_prediction` | non-contact modalities | any body contact in window | `contact_prediction/metrics.json` |
| `object_relevance` | non-caption modalities | relevant object set | `object_relevance/predictions.csv` |
| `caption_grounding` | caption objects/interaction query + sensor candidates | matching time window | `caption_grounding/metrics.json` |
| `cross_modal_retrieval` | motion/IMU/camera query | matching depth/video window | `cross_modal_retrieval/metrics.json` |
| `modality_reconstruction` | motion/IMU/camera | depth/video feature vector | `modality_reconstruction/predictions.npz` |
| `temporal_order` | two adjacent windows | whether order is correct | `temporal_order/metrics.json` |
| `misalignment_detection` | motion+visual pair | aligned vs shifted | `misalignment_detection/metrics.json` |

## Current Results

```text
timeline_action:
  accuracy: 0.0292
  macro_f1: 0.0500
  note: future test region contains unseen action classes

timeline_subtask:
  accuracy: 0.0581
  macro_f1: 0.0495
  note: future test region contains unseen subtask classes

transition_detection:
  accuracy: 0.9253
  macro_f1: 0.6552
  boundary_f1: 0.2143

next_action:
  accuracy: 0.0345
  macro_f1: 0.0593
  note: same unseen-future-class problem as timeline_action

hand_trajectory_forecast:
  MPJPE: 0.8223
  final-frame MPJPE: 1.0650

contact_prediction:
  accuracy: 1.0000
  note: degenerate on this sample because the binary contact label has only one class

object_relevance:
  micro_f1: 0.1839
  macro_f1: 0.0643

caption_grounding:
  top1: 0.0029
  top5: 0.0115
  MRR: 0.0172

cross_modal_retrieval:
  top1: 0.1494
  top5: 0.3764
  top10: 0.4741
  MRR: 0.2634

modality_reconstruction:
  R2: -0.0160

temporal_order:
  accuracy: 0.4612
  f1: 0.5487

misalignment_detection:
  accuracy: 0.5029
  f1: 0.4866
```

## How To Read These Results

Low scores are useful here. They show which tasks are not learnable from this one chronological sample with this minimal model.

The strongest signal is `cross_modal_retrieval`: motion/IMU/camera features can retrieve the matching depth/video window better than random. That means the modalities are synchronized and contain shared temporal structure.

The weakest supervised timeline tasks are weak mainly because of the split. The last 30% of a single ordered episode contains actions/subtasks not present in the first 70%, so a classifier trained on the first part cannot predict labels it never saw.

For serious research, keep the same task code but change the dataset unit:

```text
many episodes -> train episodes -> test unseen episodes
```

For single-episode learning, these tasks are best used as:

- data pipeline tests
- modality ablations
- label-alignment checks
- self-supervised retrieval experiments
- debugging templates before scaling to many episodes

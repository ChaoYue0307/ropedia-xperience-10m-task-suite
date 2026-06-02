# Episode Task Suite

Script:

```text
scripts/episode_task_suite.py
```

This script turns the single public Xperience-10M sample episode into many end-to-end tasks. It is designed for learning, debugging, and task design. It is **not** a generalization benchmark because the data is still one episode.

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
| --- | --- | --- | --- |
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

## Minimal Model Architectures

All tasks share the same window builder unless a task explicitly removes a
feature block to avoid label leakage.

```text
raw sample episode
  -> 20-frame sliding windows, stride 5
  -> all-modality feature vector X_all, 8,378 dimensions
  -> chronological split, first 70% train and last 30% test
  -> train-only z-score scaler
  -> task-specific minimal head
```

The task suite intentionally uses simple heads:

| Family | Formula | Tasks |
| --- | --- | --- |
| Linear softmax | `softmax(z(X)W + b)`, cross-entropy, L2 | `timeline_action`, `timeline_subtask`, `transition_detection`, `next_action`, `contact_prediction`, `temporal_order`, `misalignment_detection` |
| Ridge regression/projection | dual ridge regression with L2=10 on z-scored X/Y | `hand_trajectory_forecast`, `caption_grounding`, `cross_modal_retrieval`, `modality_reconstruction` |
| Multi-label logistic | `sigmoid(z(X)W + b)`, weighted object heads | `object_relevance` |

Task-specific architecture details:

| Task | Input tensor/vector | Minimal head | Output target |
| --- | --- | --- | --- |
| `timeline_action` | `X_all`, 8,378d | class-weighted linear softmax | current action label |
| `timeline_subtask` | `X_all`, 8,378d | class-weighted linear softmax | current subtask label |
| `transition_detection` | `X_all`, 8,378d | class-weighted linear softmax | steady vs transition near action boundary |
| `next_action` | `X_all(t)`, 8,378d | class-weighted linear softmax | action at `t+20` frames |
| `hand_trajectory_forecast` | `X_all(t)`, 8,378d | ridge regression | future 10 frames of left/right hand joints, 1,260d |
| `contact_prediction` | all features except `body_contacts` and caption text, 7,335d | linear softmax on observed labels | any body contact in window |
| `object_relevance` | all features except caption text, 7,482d | multi-label logistic regression | 34-object multi-hot vector |
| `caption_grounding` | sensor features, 7,482d, projected into 896d text space | ridge projection plus cosine ranking | matching time window for a text query |
| `cross_modal_retrieval` | motion/IMU/camera, 2,247d, projected into 5,096d visual space | ridge projection plus cosine ranking | matching depth/video window |
| `modality_reconstruction` | motion/IMU/camera, 2,247d | ridge regression | depth/video feature vector, 5,096d |
| `temporal_order` | `[x_t, x_t+1, x_t+1-x_t]`, 25,134d | binary linear softmax | correct vs reversed order |
| `misalignment_detection` | motion plus visual pair, 7,343d | binary linear softmax | aligned vs shifted by 8 windows |

Diagram:

```text
docs/assets/task_architectures.png
```

## Neural Baseline

The suite can also run a lightweight PyTorch MLP baseline for every selected
task while preserving the NumPy baseline artifacts:

```bash
python scripts/episode_task_suite.py \
  --output-dir results/episode_task_suite \
  --include-neural
```

This requires `torch`; use `requirements-omni.txt` when the base environment
does not already include PyTorch.

The neural path reuses the same windows, features, chronological split, leakage
filters, and metrics as the minimal heads. It writes parallel artifacts under:

```text
results/episode_task_suite/neural_mlp/<task>/
```

Each neural task directory contains `metrics.json`, `history.json`, a
`model.pt` checkpoint, and the same prediction artifact shape used by the
corresponding minimal task (`predictions.csv` or `predictions.npz`). The suite
rollup adds a `neural_tasks` section to `summary_report.json`; visualization
generation adds neural-only and minimal-vs-neural score charts when those
metrics are present.

Useful knobs:

```bash
python scripts/episode_task_suite.py \
  --include-neural \
  --neural-epochs 80 \
  --neural-hidden-dim 128 \
  --neural-batch-size 128 \
  --neural-device auto
```

This neural baseline is intentionally small. It tests whether a nonlinear head
over the current handcrafted feature vector improves per-task behavior before
moving to heavier sequence or vision-language models.

## Qwen/Omni Neural Track

The Qwen3-Omni scripts remain a separate neural/VLM track under
`scripts/omni/`. They are better suited for action/subtask adapter checks, sensor-adapter
experiments, and LoRA fine-tuning than for the full 12-task matrix. A useful
comparison order is:

- current NumPy task suite
- lightweight `neural_mlp` task suite
- adapter-only smoke tests from `scripts/omni/qwen3_omni_adapter_smoke.py`
- Qwen3-Omni zero-shot or LoRA runs where GPU/model access is available

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

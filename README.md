# Ropedia Episode Task Suite

[![Website](https://img.shields.io/badge/site-GitHub%20Pages-1f63e9)](https://chaoyue0307.github.io/ropedia-episode-task-suite/)
[![HF Space](https://img.shields.io/badge/Hugging%20Face-Space-ffb000)](https://huggingface.co/spaces/cy0307/ropedia-episode-task-suite)
[![Dataset](https://img.shields.io/badge/dataset-Ropedia%20%2F%20Xperience--10M-008b9a)](https://github.com/Ropedia)
[![Scope](https://img.shields.io/badge/scope-single%20public%20sample-b65b04)](#scope)

An audit-first embodied-AI learning repo built around one public Ropedia /
Xperience-10M sample episode.

The project does one narrow thing carefully: it turns a raw multimodal episode
into:

- all-modality sliding-window features,
- motion-only and all-modality baseline models,
- 12 end-to-end episode-level tasks,
- metrics, predictions, model weights, manifests, charts, and a static website,
- a clear explanation of what a single episode can and cannot prove.

Start with the visual dashboard:

**https://chaoyue0307.github.io/ropedia-episode-task-suite/**

Hugging Face Space app:

**https://cy0307-ropedia-episode-task-suite.static.hf.space/**

## Read This Project In Three Layers

| Layer | What to inspect | Why it matters |
| --- | --- | --- |
| Data contract | `windows.csv`, `feature_manifest.json`, modality manifests | Confirms what each sample window contains before modeling |
| Minimal heads | softmax, ridge projection/regression, multi-label logistic heads | Keeps every input/output contract visible and debuggable |
| Evidence | metrics, predictions, confusion matrices, diagrams, dashboard | Makes the single-episode claims reviewable without rerunning first |

## Links

| Resource | Link |
| --- | --- |
| This GitHub repo | https://github.com/ChaoYue0307/ropedia-episode-task-suite |
| This project website | https://chaoyue0307.github.io/ropedia-episode-task-suite/ |
| This Hugging Face Space | https://huggingface.co/spaces/cy0307/ropedia-episode-task-suite |
| Live Hugging Face static app | https://cy0307-ropedia-episode-task-suite.static.hf.space/ |
| Derived artifacts on Hugging Face | https://huggingface.co/datasets/cy0307/ropedia-episode-task-suite-artifacts |
| Minimal baseline models on Hugging Face | https://huggingface.co/cy0307/ropedia-minimal-task-baselines |
| Hugging Face collection | https://huggingface.co/collections/cy0307/ropedia-episode-task-suite |
| Ropedia website | https://ropedia.com/dataset |
| Xperience-10M release page | https://ropedia.com/blog/20260316_xperience_10m |
| Ropedia GitHub organization | https://github.com/Ropedia |
| HOMIE Toolkit | https://github.com/Ropedia/HOMIE-toolkit |
| Xperience-10M Hugging Face dataset | https://huggingface.co/datasets/ropedia-ai/xperience-10m |
| Xperience-10M sample on Hugging Face | https://huggingface.co/datasets/ropedia-ai/xperience-10m-sample |
| Ropedia Hugging Face organization | https://huggingface.co/ropedia-ai |

![ChatGPT-image-backed 12-task infographic](docs/assets/task_suite_infographic.png)

The infographic uses a ChatGPT-image-generated text-free research background and
low-resolution modality thumbnails extracted from the public sample episode. The
task names, input/output summaries, and metrics are overlaid from
[`results/episode_task_suite/summary_report.json`](results/episode_task_suite/summary_report.json)
with [`scripts/render_task_suite_infographic.py`](scripts/render_task_suite_infographic.py),
so the published PNG is a presentation graphic with verified labels and metrics,
not a hallucinated metric sheet.

![Verified Pipeline](docs/assets/pipeline_diagram.svg)

![Minimal 12-task model architectures](docs/assets/task_architectures.svg)

## Scope

This is a learning, inspection, and pipeline-validation repo. It does **not**
claim cross-episode generalization because the public sample used here is one
episode. The correct next step for real model claims is to run the same suite
over many episodes and split train/test by held-out episode.

## What Is Inside

```text
scripts/
  train_min_action_model.py         # motion/IMU baseline
  train_all_modalities_model.py     # all-modality lightweight baseline
  episode_task_suite.py             # 12 end-to-end task definitions
  generate_visualizations.py        # refreshes SVG charts + summary JSON
  render_task_suite_infographic.py  # renders the ChatGPT-image-backed PNG

results/
  min_action_model/                 # motion-only action baseline artifacts
  min_subtask_model/                # motion-only subtask baseline artifacts
  min_all_modalities_action_model/  # all-modality action artifacts
  min_all_modalities_subtask_model/ # all-modality subtask artifacts
  episode_task_suite/               # 12-task suite metrics and predictions

docs/
  index.html                        # GitHub Pages dashboard
  data/summary_metrics.json         # website-readable metrics bundle
  assets/task_suite_infographic.png # 12-task presentation graphic
  assets/task_architectures.svg     # verified 12-task minimal architecture map
  assets/charts/*.svg               # regenerated visualizations

notes/
  min_action_model.md
  all_modalities_model.md
  episode_task_suite.md
```

Raw Ropedia data is **not** committed. Download it from the original source and
follow the dataset terms.

## Data Expected

The scripts expect a workspace with the Ropedia toolkit and the sample episode:

```text
<workspace>/
  HOMIE-toolkit/
  data/sample/xperience-10m-sample/
    annotation.hdf5
    fisheye_cam0.mp4
    fisheye_cam1.mp4
    fisheye_cam2.mp4
    fisheye_cam3.mp4
    stereo_left.mp4
    stereo_right.mp4
```

The public sample dataset identifier is:

```text
ropedia-ai/xperience-10m-sample
```

Hugging Face URL:

```text
https://huggingface.co/datasets/ropedia-ai/xperience-10m-sample
```

## Quickstart

From a workspace folder:

```bash
git clone https://github.com/Ropedia/HOMIE-toolkit.git
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r HOMIE-toolkit/requirements.txt huggingface_hub hf_xet
```

Download the sample:

```bash
hf download ropedia-ai/xperience-10m-sample \
  --repo-type dataset \
  --local-dir data/sample/xperience-10m-sample
```

Clone and run this repo:

```bash
git clone https://github.com/ChaoYue0307/ropedia-episode-task-suite.git
cd ropedia-episode-task-suite
python scripts/episode_task_suite.py --workspace /path/to/workspace
```

Run the smaller baselines:

```bash
python scripts/train_min_action_model.py --workspace /path/to/workspace
python scripts/train_all_modalities_model.py --workspace /path/to/workspace
```

Refresh charts and the website data bundle:

```bash
python scripts/generate_visualizations.py
```

## Minimal 12-Task Architectures

These are deliberately minimal baselines. They are useful because every
input/output contract is explicit, not because they are strong embodied-AI
models.

Shared setup:

```text
raw episode -> 20-frame windows, stride 5 -> 8,378-d all-modality vector
chronological split: first 70% train, last 30% test
scalers are fit on train windows only
```

There are three reusable head families:

| Head family | Used by | What it means |
| --- | --- | --- |
| Linear softmax classifier | `timeline_action`, `timeline_subtask`, `transition_detection`, `next_action`, `contact_prediction`, `temporal_order`, `misalignment_detection` | z-score features, then `XW+b`, softmax, cross-entropy, L2 |
| Dual ridge regression/projection | `hand_trajectory_forecast`, `caption_grounding`, `cross_modal_retrieval`, `modality_reconstruction` | z-score input/target, solve ridge regression with L2=10 |
| Multi-label logistic regression | `object_relevance` | z-score non-caption features, sigmoid object heads, threshold at 0.5 |

The task-specific heads are:

| Task | Input | Minimal head | Output |
| --- | --- | --- | --- |
| `timeline_action` | all modalities | linear softmax | current action class |
| `timeline_subtask` | all modalities | linear softmax | current subtask class |
| `transition_detection` | all modalities | linear softmax | steady vs action boundary |
| `next_action` | all modalities at `t` | linear softmax | action at `t+20` frames |
| `hand_trajectory_forecast` | all modalities at `t` | ridge regression | future 10-frame left/right hand joints |
| `contact_prediction` | non-contact and non-caption modalities | linear softmax | any body contact |
| `object_relevance` | non-caption modalities | multi-label logistic | relevant object set |
| `caption_grounding` | sensor windows projected to text space | ridge projection + cosine ranking | matching time window for text query |
| `cross_modal_retrieval` | motion/IMU/camera projected to visual space | ridge projection + cosine ranking | matching depth/video window |
| `modality_reconstruction` | motion/IMU/camera | ridge regression | depth/video feature vector |
| `temporal_order` | `[x_t, x_t+1, x_t+1-x_t]` | binary linear softmax | correct vs reversed order |
| `misalignment_detection` | motion plus visual pair | binary linear softmax | aligned vs shifted by 8 windows |

## Key Results

| Experiment | Main score | Accuracy | Notes |
| --- | ---: | ---: | --- |
| Motion-only action | 0.9688 macro-F1 | 0.9828 | Uses motion/IMU features only |
| All-modality action | 0.9791 macro-F1 | 0.9828 | 8,378-dimensional feature vector |
| Motion-only subtask | 0.9528 macro-F1 | 0.9759 | Strong within-episode subtask signal |
| All-modality subtask | 0.9308 macro-F1 | 0.9828 | High accuracy, lower class-balanced score |
| Cross-modal retrieval | 0.3764 top-5 | n/a | Motion/IMU/camera retrieves matching depth/video |
| Transition detection | 0.6552 macro-F1 | 0.9253 | Boundary F1 is 0.2143 |
| Hand trajectory forecast | 0.8223 MPJPE | n/a | Predicts future hand-joint trajectory |

The strongest single-episode self-supervised signal is cross-modal retrieval:
motion/IMU/camera features retrieve matching depth/video windows substantially
better than random.

## Reproducibility Audit

I re-ran the full pipeline from the local raw public sample into
`/private/tmp/ropedia-audit` and compared regenerated metrics with the committed
artifacts. The baseline metrics, 12 task metrics, feature manifest, and
available modality manifest matched exactly after float normalization.

See [`notes/reproducibility_audit.md`](notes/reproducibility_audit.md) for the
commands and verification evidence.

## Why Some Scores Are Low

The task suite intentionally uses a chronological split:

```text
first 70% of the episode -> train
last 30% of the episode  -> test
```

The test segment contains some action/subtask labels never seen during training.
Timeline and next-action classifiers therefore expose the core limitation of
single-episode learning instead of hiding it behind random splits.

## Modalities Used

The all-modality vector has 8,378 dimensions and includes:

- hand/body mocap joints and contact labels,
- camera translation and rotation,
- IMU acceleration and gyroscope traces,
- depth confidence features,
- six video streams,
- caption/object/interaction text features,
- SLAM point-cloud summary features,
- calibration parameters.

The exact feature block boundaries are stored in
[`results/episode_task_suite/feature_manifest.json`](results/episode_task_suite/feature_manifest.json).

## Data Notice

Ropedia / Xperience-10M data belongs to its original authors and is subject to
the dataset's original license and access terms. This repo contains code and
derived experiment artifacts only; it does not redistribute the raw videos or
raw annotation dataset.

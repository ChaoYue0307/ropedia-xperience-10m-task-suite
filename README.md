# Xperience-10M Episode Task Suite

[![Website](https://img.shields.io/badge/site-GitHub%20Pages-1f63e9)](https://chaoyue0307.github.io/ropedia-episode-task-suite/)
[![HF Space](https://img.shields.io/badge/Hugging%20Face-Space-ffb000)](https://huggingface.co/spaces/cy0307/ropedia-episode-task-suite)
[![Dataset](https://img.shields.io/badge/dataset-Xperience--10M%20by%20Ropedia-008b9a)](https://github.com/Ropedia)
[![Scope](https://img.shields.io/badge/scope-single%20public%20sample-b65b04)](#scope)

An audit-first embodied-AI learning repo built around one public
Xperience-10M sample episode released by Ropedia.

The project does one narrow thing carefully: it turns a raw multimodal episode
into:

- manifested sliding-window features over the currently extracted modalities,
- motion-only and current all-feature baseline models,
- 12 end-to-end episode-level tasks,
- lightweight neural MLP heads for the same 12 task contracts,
- metrics, predictions, model weights, manifests, charts, and a static website,
- a clear explanation of what a single episode can and cannot prove.

## Dataset Modality Coverage

The Xperience-10M sample is a 4D multimodal episode source spanning video,
audio, depth, pose, motion capture, inertial sensing, and language annotation.
This repo keeps that distinction explicit:

- the raw sample files include six MP4 video streams with AAC audio streams,
- `annotation.hdf5` includes depth, SLAM/camera pose, hand/body mocap, IMU, and
  language annotation,
- the current minimal 8,378-d baseline feature manifest includes video, depth,
  pose/SLAM, mocap, IMU, calibration, and language blocks,
- audio is documented in the figures but is not yet extracted as a model input
  feature block in this minimal baseline.

Start with the visual dashboard:

**[chaoyue0307.github.io/ropedia-episode-task-suite](https://chaoyue0307.github.io/ropedia-episode-task-suite/)**

Hugging Face Space app:

**[cy0307-ropedia-episode-task-suite.static.hf.space](https://cy0307-ropedia-episode-task-suite.static.hf.space/)**

## Read This Project In Three Layers

| Layer | What to inspect | Why it matters |
| --- | --- | --- |
| Data contract | `windows.csv`, `feature_manifest.json`, modality manifests | Confirms what each sample window contains before modeling |
| Minimal heads | softmax, ridge projection/regression, multi-label logistic heads | Keeps every input/output contract visible and debuggable |
| Neural heads | PyTorch MLP classifiers/regressors under `neural_mlp/` | Checks whether nonlinear heads improve each task without changing features |
| Evidence | metrics, predictions, confusion matrices, diagrams, dashboard | Makes the single-episode claims reviewable without rerunning first |

## Links

| Resource | Link |
| --- | --- |
| This GitHub repo | [github.com/ChaoYue0307/ropedia-episode-task-suite](https://github.com/ChaoYue0307/ropedia-episode-task-suite) |
| This project website | [chaoyue0307.github.io/ropedia-episode-task-suite](https://chaoyue0307.github.io/ropedia-episode-task-suite/) |
| This Hugging Face Space | [huggingface.co/spaces/cy0307/ropedia-episode-task-suite](https://huggingface.co/spaces/cy0307/ropedia-episode-task-suite) |
| Live Hugging Face static app | [cy0307-ropedia-episode-task-suite.static.hf.space](https://cy0307-ropedia-episode-task-suite.static.hf.space/) |
| Derived artifacts on Hugging Face | [huggingface.co/datasets/cy0307/ropedia-episode-task-suite-artifacts](https://huggingface.co/datasets/cy0307/ropedia-episode-task-suite-artifacts) |
| Minimal baseline models on Hugging Face | [huggingface.co/cy0307/ropedia-minimal-task-baselines](https://huggingface.co/cy0307/ropedia-minimal-task-baselines) |
| Hugging Face collection | [huggingface.co/collections/cy0307/ropedia-episode-task-suite](https://huggingface.co/collections/cy0307/ropedia-episode-task-suite) |
| Xperience-10M dataset website | [ropedia.com/dataset](https://ropedia.com/dataset) |
| Xperience-10M release page | [ropedia.com/blog/20260316_xperience_10m](https://ropedia.com/blog/20260316_xperience_10m) |
| Ropedia GitHub organization | [github.com/Ropedia](https://github.com/Ropedia) |
| HOMIE Toolkit | [github.com/Ropedia/HOMIE-toolkit](https://github.com/Ropedia/HOMIE-toolkit) |
| Xperience-10M Hugging Face dataset | [huggingface.co/datasets/ropedia-ai/xperience-10m](https://huggingface.co/datasets/ropedia-ai/xperience-10m) |
| Xperience-10M sample on Hugging Face | [huggingface.co/datasets/ropedia-ai/xperience-10m-sample](https://huggingface.co/datasets/ropedia-ai/xperience-10m-sample) |
| Ropedia Hugging Face organization | [huggingface.co/ropedia-ai](https://huggingface.co/ropedia-ai) |

![ChatGPT-image-backed Xperience-10M 12-task infographic](docs/assets/task_suite_infographic.png?v=xperience10m-nn)

The infographic uses a ChatGPT-image-generated text-free research background and
low-resolution modality thumbnails extracted from the public sample episode. The
task names, input/output summaries, and metrics are overlaid from
[`results/episode_task_suite/summary_report.json`](results/episode_task_suite/summary_report.json)
with [`scripts/render_task_suite_infographic.py`](scripts/render_task_suite_infographic.py),
so the published PNG is a presentation graphic with verified labels and metrics,
not a hallucinated metric sheet.

![Verified Pipeline](docs/assets/pipeline_diagram.png?v=xperience10m-nn)

![Minimal and neural 12-task model architectures](docs/assets/task_architectures.png?v=xperience10m-nn)

The pipeline and architecture figures use the same pattern: ChatGPT-image
provides text-free visual backgrounds, while
[`scripts/render_overview_figures.py`](scripts/render_overview_figures.py)
overlays exact labels, dimensions, and metrics from the committed result files.

## Scope

This is a learning, inspection, and pipeline-validation repo. It does **not**
claim cross-episode generalization because the public sample used here is one
episode. The correct next step for real model claims is to run the same suite
over many episodes and split train/test by held-out episode.

## What Is Inside

```text
scripts/
  train_min_action_model.py         # motion/IMU baseline
  train_all_modalities_model.py     # current all-feature lightweight baseline
  episode_task_suite.py             # 12 end-to-end task definitions
  neural_task_models.py             # optional PyTorch MLP heads for all 12 tasks
  generate_visualizations.py        # refreshes SVG charts + summary JSON
  render_task_suite_infographic.py  # renders the ChatGPT-image-backed PNG
  render_overview_figures.py        # renders polished pipeline/architecture PNGs
  omni/
    download_sample_modelscope.py   # mainland-China friendly sample download
    build_episode_manifest.py       # metadata-only multi-episode scanner
    plan_finetune_sample_budget.py  # H20 storage/sample-count planner
    qwen3_omni_adapter_smoke.py     # real-data Qwen3-Omni adapter smoke test

results/
  min_action_model/                 # motion-only action baseline artifacts
  min_subtask_model/                # motion-only subtask baseline artifacts
  min_all_modalities_action_model/  # current all-feature action artifacts
  min_all_modalities_subtask_model/ # current all-feature subtask artifacts
  episode_task_suite/               # 12-task suite metrics and predictions
    neural_mlp/                     # optional neural baseline artifacts per task
  omni_exploration/                 # H20/ModelScope smoke-test artifacts

docs/
  index.html                        # GitHub Pages dashboard
  data/summary_metrics.json         # website-readable metrics bundle
  assets/task_suite_infographic.png # 12-task presentation graphic
  assets/pipeline_diagram.png       # verified episode pipeline graphic
  assets/task_architectures.png     # verified 12-task minimal architecture map
  assets/charts/*.svg               # regenerated visualizations

notes/
  min_action_model.md
  all_modalities_model.md
  episode_task_suite.md
```

Raw Xperience-10M data is **not** committed. Download it from the official
Ropedia distribution and follow the dataset terms.

## Data Expected

The scripts expect a workspace with the Ropedia HOMIE toolkit and the
Xperience-10M sample episode:

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

On mainland-China servers, use ModelScope instead:

```bash
python scripts/omni/download_sample_modelscope.py \
  --output-dir data/sample/xperience-10m-sample \
  --mode minimal
```

`--mode minimal` downloads `annotation.hdf5`, `README.md`, and
`fisheye_cam0.mp4`. Use `--mode all-training` to add all six MP4 streams while
still skipping `visualization.rrd`.

Clone and run this repo:

```bash
git clone https://github.com/ChaoYue0307/ropedia-episode-task-suite.git
cd ropedia-episode-task-suite
python scripts/episode_task_suite.py --workspace /path/to/workspace
```

Run the same 12-task suite with lightweight neural heads:

```bash
pip install torch
python scripts/episode_task_suite.py \
  --workspace /path/to/workspace \
  --include-neural
```

Run the smaller baselines:

```bash
python scripts/train_min_action_model.py --workspace /path/to/workspace
python scripts/train_all_modalities_model.py --workspace /path/to/workspace
```

## Xperience-10M Fine-Tuning Exploration On H20

This repo now includes a concrete first step toward a Qwen3-Omni fine-tuning
pipeline over Xperience-10M. The important separation is:

- direct Qwen3-Omni inputs: RGB/fisheye video, embedded MP4 audio, and language
  prompts,
- adapter-required Xperience-10M sensor inputs: depth, pose/SLAM, hand/body
  mocap, contacts, and IMU.

The H20 smoke test validates the adapter-required side first, using real
Xperience-10M sample data from ModelScope and real action labels. It does not
download or fine-tune the 30B Qwen3-Omni weights yet.

```bash
python scripts/omni/build_episode_manifest.py \
  --data-root /home/cy/Ropedia/modelscope_data \
  --output outputs/omni_exploration/modelscope_manifest.json

python scripts/omni/qwen3_omni_adapter_smoke.py \
  --workspace /home/cy/Ropedia/ropedia-episode-task-suite \
  --episode-root /home/cy/Ropedia/modelscope_data/xperience-10m-sample \
  --target action \
  --window-frames 20 \
  --stride-frames 100 \
  --max-windows-per-episode 64 \
  --epochs 2 \
  --skip-video-features
```

Verified H20 run:

| Item | Value |
| --- | ---: |
| Server | 8 x NVIDIA H20, 96GB each |
| Free storage checked | about 1.5TB under `/home/cy` |
| Data source | ModelScope `ropedia-ai/xperience-10m-sample` |
| Downloaded minimal data | 1.93GB `annotation.hdf5` + 85.7MB `fisheye_cam0.mp4` |
| Smoke windows | 59 |
| Split | single-episode chronological |
| Feature dim | 4,262 |
| Adapter soft-token blocks | 11 |
| Qwen3-Omni weights loaded | no |
| Result | 0.0000 macro-F1, expected for this single-episode chronological smoke split |

The zero score is not treated as a model claim. It is a useful signal that this
split is not leaking labels across time: the train segment does not cover every
action that appears in the held-out segment. The next real step is to add more
episodes and split by held-out episode.

### Sample Count Decision

The local Mac sample is only one episode. For H20 fine-tuning, decide sample
count by storage and evaluation design, not by the local folder. The current H20
has about 1.5TB free under `/home/cy`; after reserving space for model weights,
checkpoints, caches, and logs, a realistic first budget is:

| Phase | Episodes/samples | Approx windows at stride 5 | Purpose |
| --- | ---: | ---: | --- |
| Smoke | 1-3 | 1k-3k | Verify loaders, token alignment, and task heads |
| Pilot | 16-32 | 18k-37k | First held-out-episode evaluation |
| Useful LoRA run | 64-128 | 74k-149k | Train sensor adapters plus selected Qwen3-Omni LoRA |
| Storage-heavy run | 256+ | 297k+ | Only after download layout and checkpoint size are stable |

For the next run, use **32 episodes** if ModelScope exposes enough files
cleanly. If download structure is simple and disk remains above 800GB free,
scale to **64 or 128 episodes**. Do not aim for 10k samples first; at the
observed sample-equivalent size, that would become a data-management project
before it is a modeling experiment.

Use the budget helper before downloading:

```bash
python scripts/omni/plan_finetune_sample_budget.py \
  --storage-root /home/cy \
  --target-free-after-download-gb 800 \
  --all-training-per-episode-gb 2.4 \
  --full-preview-per-episode-gb 5.1
```

Refresh charts and the website data bundle:

```bash
python scripts/generate_visualizations.py
python scripts/render_overview_figures.py
python scripts/render_task_suite_infographic.py
```

## Minimal 12-Task Architectures

These are deliberately minimal baselines. They are useful because every
input/output contract is explicit, not because they are strong embodied-AI
models.

Shared setup:

```text
raw episode -> 20-frame windows, stride 5 -> 8,378-d current feature vector
chronological split: first 70% train, last 30% test
scalers are fit on train windows only
```

There are four reusable head families:

| Head family | Used by | What it means |
| --- | --- | --- |
| Linear softmax classifier | `timeline_action`, `timeline_subtask`, `transition_detection`, `next_action`, `contact_prediction`, `temporal_order`, `misalignment_detection` | z-score features, then `XW+b`, softmax, cross-entropy, L2 |
| Dual ridge regression/projection | `hand_trajectory_forecast`, `modality_reconstruction` | z-score input/target, solve ridge regression with L2=10 |
| Ridge + cosine ranking | `caption_grounding`, `cross_modal_retrieval` | project one modality into another feature space, then rank candidates by cosine |
| Multi-label logistic regression | `object_relevance` | z-score non-caption features, sigmoid object heads, threshold at 0.5 |

The optional neural run keeps the same feature vectors, leakage filters,
chronological splits, and metrics, but replaces the task heads with small
PyTorch MLP classifiers or regressors. Its outputs live under
[`results/episode_task_suite/neural_mlp/`](results/episode_task_suite/neural_mlp/),
and the rollup is stored in the `neural_tasks` section of
[`results/episode_task_suite/summary_report.json`](results/episode_task_suite/summary_report.json).

The task-specific heads are:

| Task | Input | Minimal head | Output |
| --- | --- | --- | --- |
| `timeline_action` | all featurized modalities | linear softmax | current action class |
| `timeline_subtask` | all featurized modalities | linear softmax | current subtask class |
| `transition_detection` | all featurized modalities | linear softmax | steady vs action boundary |
| `next_action` | all featurized modalities at `t` | linear softmax | action at `t+20` frames |
| `hand_trajectory_forecast` | all featurized modalities at `t` | ridge regression | future 10-frame left/right hand joints |
| `contact_prediction` | non-contact and non-caption feature blocks | linear softmax | any body contact |
| `object_relevance` | non-caption feature blocks | multi-label logistic | relevant object set |
| `caption_grounding` | sensor windows projected to text space | ridge projection + cosine ranking | matching time window for text query |
| `cross_modal_retrieval` | motion/IMU/camera projected to visual space | ridge projection + cosine ranking | matching depth/video window |
| `modality_reconstruction` | motion/IMU/camera | ridge regression | depth/video feature vector |
| `temporal_order` | `[x_t, x_t+1, x_t+1-x_t]` | binary linear softmax | correct vs reversed order |
| `misalignment_detection` | motion plus visual pair | binary linear softmax | aligned vs shifted by 8 windows |

## Key Results

| Experiment | Main score | Accuracy | Notes |
| --- | ---: | ---: | --- |
| Motion-only action | 0.9688 macro-F1 | 0.9828 | Uses motion/IMU features only |
| Current all-feature action | 0.9791 macro-F1 | 0.9828 | 8,378-dimensional feature vector |
| Motion-only subtask | 0.9528 macro-F1 | 0.9759 | Strong within-episode subtask signal |
| Current all-feature subtask | 0.9308 macro-F1 | 0.9828 | High accuracy, lower class-balanced score |
| Cross-modal retrieval | 0.3764 top-5 | n/a | Motion/IMU/camera retrieves matching depth/video |
| Transition detection | 0.6552 macro-F1 | 0.9253 | Boundary F1 is 0.2143 |
| Hand trajectory forecast | 0.8223 MPJPE | n/a | Predicts future hand-joint trajectory |
| Neural MLP hand forecast | 0.1116 MPJPE | n/a | Same features/split, nonlinear regression head |
| Neural MLP temporal order | 0.8718 F1 | 0.8707 | Strong improvement on adjacent-window ordering |
| Neural MLP misalignment | 0.7335 F1 | 0.7312 | Detects shifted motion/visual pairs better than the linear head |

## Neural MLP Results

The neural baseline was run locally with `--include-neural` for all 12 tasks
using 80 epochs, hidden size 128, batch size 128, and CPU execution. It is not a
foundation model result; it is a controlled nonlinear-head comparison over the
same 8,378-d handcrafted window features.

| Task | Neural metric | Minimal metric | Readout |
| --- | ---: | ---: | --- |
| `timeline_action` | 0.0263 macro-F1 | 0.0500 macro-F1 | Still blocked by unseen future classes |
| `timeline_subtask` | 0.0175 macro-F1 | 0.0495 macro-F1 | Same single-episode split limitation |
| `transition_detection` | 0.6485 macro-F1 | 0.6552 macro-F1 | Similar to the linear baseline |
| `next_action` | 0.0235 macro-F1 | 0.0593 macro-F1 | Same unseen-label issue |
| `hand_trajectory_forecast` | 0.1116 MPJPE | 0.8223 MPJPE | Neural regression improves this target |
| `contact_prediction` | 1.0000 macro-F1 | 1.0000 macro-F1 | Degenerate one-class sample |
| `object_relevance` | 0.1798 micro-F1 | 0.1839 micro-F1 | Similar weak object signal |
| `caption_grounding` | 0.0178 MRR | 0.0172 MRR | Similar ranking behavior |
| `cross_modal_retrieval` | 0.1530 MRR | 0.2634 MRR | Linear ridge remains stronger here |
| `modality_reconstruction` | -0.0102 R2 | -0.0160 R2 | Small improvement but still weak |
| `temporal_order` | 0.8718 F1 | 0.5487 F1 | Neural head captures local temporal structure |
| `misalignment_detection` | 0.7335 F1 | 0.4866 F1 | Neural head improves alignment detection |

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

## Feature Blocks Used

The current feature vector has 8,378 dimensions and includes:

- hand/body mocap joints and contact labels,
- camera translation and rotation,
- IMU acceleration and gyroscope traces,
- depth confidence features,
- six video streams,
- caption/object/interaction text features,
- SLAM point-cloud summary features,
- calibration parameters.

It does not yet include an audio feature block.

The exact feature block boundaries are stored in
[`results/episode_task_suite/feature_manifest.json`](results/episode_task_suite/feature_manifest.json).

## Data Notice

Xperience-10M data belongs to its original authors and is subject to the
official Ropedia dataset license and access terms. This repo contains code and
derived experiment artifacts only; it does not redistribute the raw videos or
raw annotation dataset.

# Ropedia Xperience-10M Task Suite

[![Website](https://img.shields.io/badge/site-GitHub%20Pages-1f63e9)](https://chaoyue0307.github.io/ropedia-xperience-10m-task-suite/)
[![HF Space](https://img.shields.io/badge/Hugging%20Face-Space-ffb000)](https://huggingface.co/spaces/cy0307/ropedia-xperience-10m-task-suite)
[![Dataset](https://img.shields.io/badge/dataset-Xperience--10M%20by%20Ropedia-008b9a)](https://huggingface.co/datasets/ropedia-ai/xperience-10m)
[![Scope](https://img.shields.io/badge/scope-single%20public%20sample-b65b04)](#scope)
[![Citation](https://img.shields.io/badge/citation-CFF-7ae5c3)](CITATION.cff)
[![License](https://img.shields.io/badge/license-code%20MIT%20%2B%20data%20terms-a7f078)](LICENSE)

<p align="center">
  <img src="docs/assets/brand/xperience10m-logo-social-card.png" alt="Ropedia Xperience-10M Task Suite logo card" width="760">
</p>

An audit-first embodied-AI learning repo built around one public
Xperience-10M sample episode released by Ropedia.

The project does one narrow thing carefully: it turns a raw multimodal episode
into:

- manifested sliding-window features over the currently extracted modalities,
- motion-only and current all-feature baseline models,
- 12 end-to-end episode-level tasks,
- lightweight neural MLP heads for the same 12 task contracts,
- a generated four-direction research taxonomy matching the Ropedia job tracks,
- four additional direction-extension probes with minimal and neural baselines,
- junior-friendly walkthroughs for every task, with case study, input, process, and output,
- a next-milestone track for Qwen3-Omni fine-tuning and sensor-bridge evaluation,
- metrics, predictions, model weights, manifests, charts, and a static website,
- a clear explanation of what a single episode can and cannot prove.

## Evidence Contract

This repo is organized around an explicit proof boundary:

| Claim layer | Evidence | Boundary |
| --- | --- | --- |
| Official Xperience-10M description | `XPERIENCE10M_DATASET_CARD_ALIGNMENT.md`, `docs/data/xperience10m_dataset_card_alignment.json` | aligns public wording with the official gated dataset card, public sample card, and HF API metadata; does not mirror raw data |
| Source alignment audit | `SOURCE_ALIGNMENT_AUDIT.md`, `docs/data/source_alignment_audit.json`, `scripts/validate_source_alignment.py` | validates source facts and boundary wording across repo, website, and HF cards |
| Figure index | `FIGURE_INDEX.md`, `docs/data/figure_index.json`, `scripts/build_figure_index.py` | catalogs public figures, charts, modality thumbnails, dimensions, hashes, roles, and source scripts |
| Brand assets | `docs/assets/brand/`, `docs/favicon.png`, `docs/apple-touch-icon.png`, `scripts/build_brand_assets.py` | applies the ChatGPT-image-generated project logo across the website, README, HF cards, favicon, and social previews |
| Data windows | `results/episode_task_suite/windows.csv`, `shared_windows.npz`, `summary_report.json` | one public sample episode |
| Feature contract | `results/episode_task_suite/feature_manifest.json`, `available_modalities.json` | 8,378 current features; audio documented but not featurized |
| Evaluation protocol | `EVALUATION_PROTOCOL.md`, `docs/data/evaluation_protocol.json`, `scripts/build_evaluation_protocol.py` | defines windowing, chronological split, leakage controls, per-task metrics, and unsupported interpretations |
| 12-task suite | `scripts/episode_task_suite.py`, per-task `metrics.json`, predictions | chronological single-episode split |
| Neural heads | `scripts/neural_task_models.py`, `results/episode_task_suite/neural_mlp/` | compact MLP heads, not a foundation model |
| Research directions | `research_direction_taxonomy.json`, extension probe results | direct/proxy/diagnostic evidence, not full solutions |
| Qwen3-Omni | `results/omni_finetune/DATA_BLOCKER_REPORT.md`, `MULTI_EPISODE_ACCESS_STATUS.md` | readiness-only until 32 valid episodes are available |
| Scope claims guard | `scripts/validate_scope_claims.py`, `docs/data/scope_claims_audit.json` | historical `32ep` path strings are provenance, not 32-episode results |
| Mirror parity | `scripts/validate_mirror_parity.py`, `docs/data/mirror_parity.json` | prepared GitHub/HF mirrors carry matching data, figure, website HTML, and validator files |
| Publication hygiene | `scripts/validate_publication_package.py`, `docs/data/publication_audit.json` | public repo and HF bundles only; ignored local scratch files are excluded, and public cards must reference the current task-first figure |
| Quality gates | `QUALITY_GATES.md`, `docs/data/quality_gates.json`, `scripts/build_quality_gates.py` | one reviewer-facing checklist for automated gates and live post-publish checks |
| Artifact index | `scripts/build_artifact_index.py`, `docs/data/artifact_index.json` | selective source-of-truth catalog with existence, size, and stable-file hashes |
| Reviewer scorecard | `REVIEWER_SCORECARD.md`, `docs/data/reviewer_scorecard.json` | compact verified/data-gated/not-redistributed decision table for first-pass reviewers |
| Citation and metadata | `CITATION.cff`, `codemeta.json`, `docs/data/project_manifest.json`, `LICENSE` | code is MIT-scoped; raw-data use follows Xperience-10M terms |
| Reviewer path | `docs/data/reviewer_packet.json`, website reviewer section | audit guide only; no new experimental claim |

Read the full contract in [`EVIDENCE_CONTRACT.md`](EVIDENCE_CONTRACT.md), or
consume the machine-readable copy at
[`docs/data/evidence_contract.json`](docs/data/evidence_contract.json).
The current publication audit is at
[`docs/data/publication_audit.json`](docs/data/publication_audit.json).
The publication quality-gate summary is at
[`QUALITY_GATES.md`](QUALITY_GATES.md) and
[`docs/data/quality_gates.json`](docs/data/quality_gates.json).
The last live-publication verification report is at
[`docs/data/live_publication_status.json`](docs/data/live_publication_status.json).
The current prepared-mirror parity report is at
[`docs/data/mirror_parity.json`](docs/data/mirror_parity.json).
The current scope-claims audit is at
[`docs/data/scope_claims_audit.json`](docs/data/scope_claims_audit.json).
The generated evaluation protocol is at
[`EVALUATION_PROTOCOL.md`](EVALUATION_PROTOCOL.md) and
[`docs/data/evaluation_protocol.json`](docs/data/evaluation_protocol.json).
The source-of-truth artifact index is at
[`docs/data/artifact_index.json`](docs/data/artifact_index.json).
For a human-readable artifact map, use
[`ARTIFACT_GUIDE.md`](ARTIFACT_GUIDE.md).
For reproduction commands and expected outputs, use
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) and
[`docs/data/reproducibility_matrix.json`](docs/data/reproducibility_matrix.json).
Project citation and machine-readable metadata live in
[`CITATION.cff`](CITATION.cff), [`codemeta.json`](codemeta.json), and
[`docs/data/project_manifest.json`](docs/data/project_manifest.json).
The upstream dataset-card alignment note is
[`XPERIENCE10M_DATASET_CARD_ALIGNMENT.md`](XPERIENCE10M_DATASET_CARD_ALIGNMENT.md),
with a machine-readable copy at
[`docs/data/xperience10m_dataset_card_alignment.json`](docs/data/xperience10m_dataset_card_alignment.json).
The generated source-alignment audit is at
[`SOURCE_ALIGNMENT_AUDIT.md`](SOURCE_ALIGNMENT_AUDIT.md) and
[`docs/data/source_alignment_audit.json`](docs/data/source_alignment_audit.json).
The generated figure index is at
[`FIGURE_INDEX.md`](FIGURE_INDEX.md) and
[`docs/data/figure_index.json`](docs/data/figure_index.json).
The ChatGPT-image project logo is packaged by
[`scripts/build_brand_assets.py`](scripts/build_brand_assets.py), stored under
[`docs/assets/brand/`](docs/assets/brand/), and audited in
[`docs/data/brand_assets.json`](docs/data/brand_assets.json).

## Reviewer Scorecard

If you only have one minute, use
[`REVIEWER_SCORECARD.md`](REVIEWER_SCORECARD.md) and
[`docs/data/reviewer_scorecard.json`](docs/data/reviewer_scorecard.json).
They give the current decision boundary in one compact table:

| Area | Current decision |
| --- | --- |
| Public-sample pipeline | Verified on one public sample episode: 5,821 frames, 1,161 windows, 8,378 current features |
| 12-task suite | Verified minimal baselines with committed metrics, predictions, and manifests |
| Neural heads | Verified compact PyTorch MLP heads over the same task contracts and chronological splits |
| Official dataset wording | Verified against the public `ropedia-ai/xperience-10m` dataset card/API metadata |
| Source alignment audit | Verified source facts and source-boundary markers across repo, website, and HF cards |
| Evaluation protocol | Verified generated protocol for windowing, split policy, leakage controls, and per-task metrics |
| Website and HF mirrors | Verified by local integrity, mirror parity, and live-publication checks |
| Qwen3-Omni 32-episode pilot | Data-gated; prepared, but not a model-quality claim |
| Raw Xperience-10M data / full Qwen weights | Not redistributed |

## 90-Second Reviewer Path

If you are reviewing the project cold, open these in order:

| Step | Question | Primary artifacts | What should be true |
| --- | --- | --- | --- |
| 1 | What is actually claimed? | [`REVIEWER_SCORECARD.md`](REVIEWER_SCORECARD.md), [`docs/data/reviewer_scorecard.json`](docs/data/reviewer_scorecard.json), [`EVIDENCE_CONTRACT.md`](EVIDENCE_CONTRACT.md), [`ARTIFACT_GUIDE.md`](ARTIFACT_GUIDE.md), [`QUALITY_GATES.md`](QUALITY_GATES.md), [`docs/data/artifact_index.json`](docs/data/artifact_index.json), [`docs/data/figure_index.json`](docs/data/figure_index.json), [`docs/data/brand_assets.json`](docs/data/brand_assets.json), [`docs/data/live_publication_status.json`](docs/data/live_publication_status.json), [`docs/data/mirror_parity.json`](docs/data/mirror_parity.json), [`docs/data/publication_audit.json`](docs/data/publication_audit.json), [`docs/data/scope_claims_audit.json`](docs/data/scope_claims_audit.json) | Single-episode task engineering and hygiene are claimed; historical `32ep` identifiers are not treated as real 32-episode results, visual/brand assets are indexed, and quality gates plus prepared and live mirrors are checked. |
| 2 | What is the official upstream dataset? | [`XPERIENCE10M_DATASET_CARD_ALIGNMENT.md`](XPERIENCE10M_DATASET_CARD_ALIGNMENT.md), [`docs/data/xperience10m_dataset_card_alignment.json`](docs/data/xperience10m_dataset_card_alignment.json), [official HF dataset](https://huggingface.co/datasets/ropedia-ai/xperience-10m) | The full dataset is described as a gated large-scale 4D multimodal egocentric source; this repo validates only one public sample episode. |
| 3 | Are source facts consistently presented? | [`SOURCE_ALIGNMENT_AUDIT.md`](SOURCE_ALIGNMENT_AUDIT.md), [`docs/data/source_alignment_audit.json`](docs/data/source_alignment_audit.json), [`scripts/validate_source_alignment.py`](scripts/validate_source_alignment.py) | Repo, website, and HF cards preserve full-dataset, sample-card, API-listing, and project-boundary markers. |
| 4 | How exactly are tasks evaluated? | [`EVALUATION_PROTOCOL.md`](EVALUATION_PROTOCOL.md), [`docs/data/evaluation_protocol.json`](docs/data/evaluation_protocol.json), [`scripts/build_evaluation_protocol.py`](scripts/build_evaluation_protocol.py) | The window unit, chronological split, leakage controls, task metrics, and unsupported interpretations are explicit. |
| 5 | How do I reproduce it? | [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md), [`docs/data/reproducibility_matrix.json`](docs/data/reproducibility_matrix.json), [`notes/reproducibility_audit.md`](notes/reproducibility_audit.md) | Public commands, expected outputs, and exact-match audit evidence are explicit. |
| 6 | What is one model input? | [`windows.csv`](results/episode_task_suite/windows.csv), [`feature_manifest.json`](results/episode_task_suite/feature_manifest.json), [`available_modalities.json`](results/episode_task_suite/available_modalities.json) | The input is an aligned 8,378-d window vector with explicit feature-block boundaries. |
| 7 | Are the task results backed by files? | [`summary_report.json`](results/episode_task_suite/summary_report.json), [`neural_mlp/`](results/episode_task_suite/neural_mlp/), [`docs/data/summary_metrics.json`](docs/data/summary_metrics.json) | Each task has minimal and neural-head evidence over the same window contracts. |
| 8 | Is the website internally coherent? | [`docs/data/website_integrity.json`](docs/data/website_integrity.json), [`scripts/validate_website_integrity.py`](scripts/validate_website_integrity.py) | Local links, anchors, JSON data, and referenced images are checked before publishing. |
| 9 | What is still pending? | [`DATA_BLOCKER_REPORT.md`](results/omni_finetune/DATA_BLOCKER_REPORT.md), [`MULTI_EPISODE_ACCESS_STATUS.md`](results/omni_finetune/MULTI_EPISODE_ACCESS_STATUS.md), [`scripts/omni/discover_xperience10m_sources.py`](scripts/omni/discover_xperience10m_sources.py) | The 32-episode Qwen3-Omni run is prepared but not yet a real model-quality claim. |

The machine-readable reviewer packet is
[`docs/data/reviewer_packet.json`](docs/data/reviewer_packet.json).

## Artifact Index

[`docs/data/artifact_index.json`](docs/data/artifact_index.json) is the compact
audit map for the repo. It lists the core proof artifacts, whether each exists,
its size, and a SHA-256 hash for stable files. Volatile generated files, such as
the publication audit with a run timestamp, are marked so reviewers know they
are checked for presence and size rather than treated as fixed hashes.

[`ARTIFACT_GUIDE.md`](ARTIFACT_GUIDE.md) is the human-readable companion. It
groups the same proof layer into start-here files, data-contract files,
task-evidence files, platform mirrors, and scale-up boundary artifacts.

## Evaluation Protocol

[`EVALUATION_PROTOCOL.md`](EVALUATION_PROTOCOL.md) and
[`docs/data/evaluation_protocol.json`](docs/data/evaluation_protocol.json) are
generated from committed metric artifacts. They define:

- the 20-frame window unit, stride, feature dimension, and raw-data boundary,
- the chronological 70/30 single-episode split and its generalization limit,
- the per-task input, target, primary metric, minimal score, and neural score,
- leakage controls for future labels, target feature blocks, caption/object
  labels, and train-only normalization,
- unsupported interpretations, including cross-episode generalization,
  audio-visual learning, pixel-depth reconstruction, and real 32-episode
  Qwen3-Omni quality.

## Official Dataset Alignment

The official [`ropedia-ai/xperience-10m`](https://huggingface.co/datasets/ropedia-ai/xperience-10m)
card describes Xperience-10M as a large-scale gated egocentric multimodal
dataset for embodied AI, robotics, world models, and spatial intelligence. Its
public metadata lists video classification, image-to-text, depth estimation,
and robotics task categories; 3D, audio, and video modalities; English
language; `other` license; and manually reviewed non-commercial access.

At full scale, the official card describes about 10 million experience units,
about 10,000 hours, six RGB streams per episode, audio, stereo depth, camera
pose/SLAM, hand and full-body mocap, IMU, captions, metadata, and calibration.
The card also reports headline counts such as billions of RGB/depth/IMU records
and large caption/object annotations. The live HF page/API separately shows a
31.9 TB currently hosted file-size display; this is kept separate from the
card's about-1PB full-scale storage statement. This repo records those upstream facts in
[`XPERIENCE10M_DATASET_CARD_ALIGNMENT.md`](XPERIENCE10M_DATASET_CARD_ALIGNMENT.md)
and [`docs/data/xperience10m_dataset_card_alignment.json`](docs/data/xperience10m_dataset_card_alignment.json).

The current HF API snapshot for the gated dataset reports commit
`ce943cf271a758b60240084892d05cf6dc12dd90`, last modified
`2026-04-21T05:03:45.000Z`, manual gating, and a metadata file listing with
803 session folders and 12,103 episode folders carrying `annotation.hdf5`.
Those counts are upstream listing metadata only; they are not local downloads,
not redistributed files, and not evidence of model quality in this repo.

The public sample repo,
[`ropedia-ai/xperience-10m-sample`](https://huggingface.co/datasets/ropedia-ai/xperience-10m-sample),
is separately documented as `Xperience-10M-Sample` with sample metadata,
`cc-by-nc-4.0` license, HOMIE Toolkit usage, and Rerun 0.29.0 `.rrd`
visualization. This project preserves that distinction: the sample powers the
current 5,821-frame audit suite, while the full gated dataset remains the
future source for held-out multi-episode training.

This repo's current verified subset is much smaller and intentionally explicit:

- one public sample episode, 5,821 frames, and 1,161 aligned windows,
- raw sample files with six MP4 video streams and AAC audio streams,
- `annotation.hdf5` carrying depth, SLAM/camera pose, hand/body mocap, IMU,
  language/caption annotations, calibration, metadata, and timing records,
- an 8,378-d baseline feature vector using video-derived statistics, depth,
  pose/SLAM, mocap, IMU, calibration, and language-derived blocks,
- audio documented in figures and the modality atlas, but not yet extracted as
  a model input feature block.

The same alignment note also records what is not yet claimed: real
audio-visual learning, caption generation, pixel-depth estimation, SLAM
estimation, neural rendering, policy learning, cross-episode generalization,
and real 32-episode Qwen3-Omni model quality.
It also preserves the official responsible-use boundary: the open-source
dataset is limited in diversity and showcase/production quality, and it should
not be used for identity recognition, re-identification, biometric profiling,
surveillance, sensitive attribute inference, or safety-critical deployment
without appropriate safeguards.

Start with the visual dashboard:

**[chaoyue0307.github.io/ropedia-xperience-10m-task-suite](https://chaoyue0307.github.io/ropedia-xperience-10m-task-suite/)**

Hugging Face Space app:

**[cy0307-ropedia-xperience-10m-task-suite.static.hf.space](https://cy0307-ropedia-xperience-10m-task-suite.static.hf.space/)**

## Read This Project In Three Layers

| Layer | What to inspect | Why it matters |
| --- | --- | --- |
| Reviewer scorecard | `REVIEWER_SCORECARD.md`, `docs/data/reviewer_scorecard.json` | Gives a one-table current decision boundary before reading the full audit trail |
| Data contract | `windows.csv`, `feature_manifest.json`, modality manifests | Confirms what each sample window contains before modeling |
| Official dataset alignment | `XPERIENCE10M_DATASET_CARD_ALIGNMENT.md`, `docs/data/xperience10m_dataset_card_alignment.json` | Keeps public descriptions aligned with the official gated dataset card |
| Source alignment audit | `SOURCE_ALIGNMENT_AUDIT.md`, `docs/data/source_alignment_audit.json` | Verifies source facts and boundary markers across repo, website, and HF cards |
| Figure index | `FIGURE_INDEX.md`, `docs/data/figure_index.json` | Makes public figures, charts, modality thumbnails, dimensions, hashes, and source scripts auditable |
| Brand assets | `docs/data/brand_assets.json`, `docs/assets/brand/` | Makes the generated logo, favicon, README/HF card image, app icon, and social preview auditable |
| Evaluation protocol | `EVALUATION_PROTOCOL.md`, `docs/data/evaluation_protocol.json` | Defines the task unit, split, metrics, leakage controls, and unsupported interpretations |
| Minimal heads | softmax, ridge projection/regression, multi-label logistic heads | Keeps every input/output contract visible and debuggable |
| Neural heads | PyTorch MLP classifiers/regressors under `neural_mlp/` | Checks whether nonlinear heads improve each task without changing features |
| Evidence | metrics, predictions, confusion matrices, diagrams, dashboard | Makes the single-episode claims reviewable without rerunning first |
| Quality gates | `QUALITY_GATES.md`, `docs/data/quality_gates.json` | Shows the exact automated and post-publish checks required before presenting a release as current |
| Live publication status | `docs/data/live_publication_status.json` | Records the last live GitHub Pages, GitHub raw, and Hugging Face mirror verification |
| Publication audit | `docs/data/publication_audit.json` | Confirms public bundles contain no raw Xperience-10M data, Python caches, heavy archives, token strings, or stale public-card figure references |
| Artifact index | `docs/data/artifact_index.json` | Gives reviewers a compact source-of-truth catalog with stable hashes |
| Artifact guide | `ARTIFACT_GUIDE.md` | Groups the public evidence into reviewer-friendly layers |
| Reproducibility contract | `REPRODUCIBILITY.md`, `docs/data/reproducibility_matrix.json` | States public commands, expected outputs, exact-match audit evidence, and non-reproducible boundaries |
| Citation metadata | `CITATION.cff`, `codemeta.json`, `LICENSE` | Makes the repo easier to cite, index, and reuse without confusing code license and dataset terms |

## Links

| Resource | Link |
| --- | --- |
| This GitHub repo | [github.com/ChaoYue0307/ropedia-xperience-10m-task-suite](https://github.com/ChaoYue0307/ropedia-xperience-10m-task-suite) |
| This project website | [chaoyue0307.github.io/ropedia-xperience-10m-task-suite](https://chaoyue0307.github.io/ropedia-xperience-10m-task-suite/) |
| This Hugging Face Space | [huggingface.co/spaces/cy0307/ropedia-xperience-10m-task-suite](https://huggingface.co/spaces/cy0307/ropedia-xperience-10m-task-suite) |
| Live Hugging Face static app | [cy0307-ropedia-xperience-10m-task-suite.static.hf.space](https://cy0307-ropedia-xperience-10m-task-suite.static.hf.space/) |
| Derived artifacts on Hugging Face | [huggingface.co/datasets/cy0307/ropedia-xperience-10m-task-suite-artifacts](https://huggingface.co/datasets/cy0307/ropedia-xperience-10m-task-suite-artifacts) |
| Minimal and neural task baselines on Hugging Face | [huggingface.co/cy0307/ropedia-xperience-10m-task-baselines](https://huggingface.co/cy0307/ropedia-xperience-10m-task-baselines) |
| Hugging Face collection | [huggingface.co/collections/cy0307/ropedia-xperience-10m-task-suite](https://huggingface.co/collections/cy0307/ropedia-xperience-10m-task-suite) |
| Xperience-10M dataset website | [ropedia.com/dataset](https://ropedia.com/dataset) |
| Xperience-10M release page | [ropedia.com/blog/20260316_xperience_10m](https://ropedia.com/blog/20260316_xperience_10m) |
| Ropedia GitHub organization | [github.com/Ropedia](https://github.com/Ropedia) |
| HOMIE Toolkit | [github.com/Ropedia/HOMIE-toolkit](https://github.com/Ropedia/HOMIE-toolkit) |
| Xperience-10M Hugging Face dataset | [huggingface.co/datasets/ropedia-ai/xperience-10m](https://huggingface.co/datasets/ropedia-ai/xperience-10m) |
| Xperience-10M sample on Hugging Face | [huggingface.co/datasets/ropedia-ai/xperience-10m-sample](https://huggingface.co/datasets/ropedia-ai/xperience-10m-sample) |
| Ropedia Hugging Face organization | [huggingface.co/ropedia-ai](https://huggingface.co/ropedia-ai) |

## Citation, License, And Metadata

Use [`CITATION.cff`](CITATION.cff) when citing this project. The repository
also includes [`codemeta.json`](codemeta.json) for machine-readable software
metadata and [`docs/data/project_manifest.json`](docs/data/project_manifest.json)
for website/Hugging Face surface metadata.

The code files are MIT-licensed. Raw Xperience-10M data is not redistributed
here, and dataset use remains governed by the official Ropedia/Xperience-10M
terms. See [`LICENSE`](LICENSE) and [`DATA_NOTICE.md`](DATA_NOTICE.md).

![ChatGPT-image-backed Ropedia Xperience-10M 12-task infographic](docs/assets/task_suite_infographic.png?v=xperience10m-taskfirst-v12-modality-xl)

The infographic uses a ChatGPT-image-generated text-free research background,
but now puts the shared processing contract and all 12 task families before the
modality atlas. Public-sample modality thumbnails remain enlarged below the
task map. The task names, input/output summaries, and metrics are overlaid from
[`results/episode_task_suite/summary_report.json`](results/episode_task_suite/summary_report.json)
with [`scripts/render_task_suite_infographic.py`](scripts/render_task_suite_infographic.py),
so the published PNG is a presentation graphic with verified labels and metrics,
not a hallucinated metric sheet.

The website also includes a responsive native modality atlas backed by
[`docs/data/modality_atlas.json`](docs/data/modality_atlas.json) and
[`docs/assets/modalities/`](docs/assets/modalities/). Those assets are small
derived thumbnails from the public sample, not raw Xperience-10M files.

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
  research_direction_taxonomy.py    # maps 12 tasks to the four research tracks
  research_direction_extension_tasks.py # one extra data-backed probe per track
  task_walkthroughs.py              # beginner explanations for each task contract
  generate_visualizations.py        # refreshes SVG charts + summary JSON
  render_task_suite_infographic.py  # renders the ChatGPT-image-backed PNG
  export_modality_atlas_assets.py   # exports responsive modality-card assets
  render_overview_figures.py        # renders polished pipeline/architecture PNGs
  build_brand_assets.py             # derives logo sizes, favicon, social card
  build_artifact_index.py           # builds the source-of-truth reviewer index
  build_quality_gates.py            # builds reviewer-facing publication gates
  validate_mirror_parity.py         # checks prepared GitHub/HF mirror file parity
  validate_scope_claims.py          # checks Qwen3-Omni readiness/result claim boundaries
  validate_website_integrity.py     # checks local site links, anchors, JSON, images
  validate_publication_package.py   # checks public repo + HF bundle hygiene
  publish_hf_bundles.py             # uploads prepared HF Space/artifact/model bundles
  omni/
    download_sample_modelscope.py   # ModelScope sample download helper
    build_episode_manifest.py       # metadata-only multi-episode scanner
    plan_finetune_sample_budget.py  # storage/sample-count planner
    qwen3_omni_adapter_smoke.py     # real-data Qwen3-Omni adapter smoke test

results/
  min_action_model/                 # motion-only action baseline artifacts
  min_subtask_model/                # motion-only subtask baseline artifacts
  min_all_modalities_action_model/  # current all-feature action artifacts
  min_all_modalities_subtask_model/ # current all-feature subtask artifacts
  episode_task_suite/               # 12-task suite metrics and predictions
    neural_mlp/                     # optional neural baseline artifacts per task
    research_directions/            # four-track taxonomy, CSV, and summary
    research_direction_extensions/  # four extra direction probes + predictions
    task_walkthroughs/              # case-study walkthroughs for all 12 tasks
  omni_exploration/                 # ModelScope readiness-check artifacts

docs/
  index.html                        # GitHub Pages dashboard
  data/summary_metrics.json         # website-readable metrics bundle
  data/evidence_contract.json       # machine-readable proof boundary
  data/artifact_index.json          # compact proof-artifact catalog
  data/live_publication_status.json # live GitHub/HF publication verification
  data/quality_gates.json           # machine-readable publication gates
  data/publication_audit.json       # machine-readable publication hygiene check
  data/website_integrity.json       # machine-readable website integrity check
  data/project_manifest.json        # machine-readable public-surface metadata
  data/reviewer_packet.json         # machine-readable reviewer path and proof boundary
  data/research_directions.json     # four-track website data bundle
  data/research_direction_extensions.json # four extra probe data bundle
  data/task_walkthroughs.json       # beginner task explanation data bundle
  data/modality_atlas.json          # responsive modality-card data
  assets/brand/*.png                # project logo, favicon, social card
  assets/task_suite_infographic.png # 12-task presentation graphic
  assets/modalities/                # public-sample derived modality thumbnails
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

If Hugging Face access is unavailable in your environment, use ModelScope:

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
git clone https://github.com/ChaoYue0307/ropedia-xperience-10m-task-suite.git
cd ropedia-xperience-10m-task-suite
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

## Xperience-10M Fine-Tuning Exploration

This repo includes a first Qwen3-Omni fine-tuning path over Xperience-10M, but
the current evidence is still readiness evidence rather than model quality.
The useful distinction is:

- direct Qwen3-Omni inputs: RGB/fisheye video, embedded MP4 audio, and language
  prompts,
- adapter-required Xperience-10M sensor inputs: depth, pose/SLAM, hand/body
  mocap, contacts, and IMU.

The current scale-up artifacts prove that the export, manifest, sensor-feature,
LoRA, and evaluation scripts can run on the available sample episode. They do
not prove a real 32-episode result. A real pilot requires at least 32 valid
episodes, held-out episode splits, training metadata, predictions, metrics, and
a run report.

### Sample Count Decision

Do not treat "10M" as a reason to start with the entire dataset. The engineering
unit that matters first is diverse held-out episodes, not adjacent windows from
one session.

| Phase | Episodes/samples | Approx windows at stride 5 | Purpose |
| --- | ---: | ---: | --- |
| Readiness | 1-3 | 1k-3k | Verify loaders, token alignment, and task heads |
| Pilot | 16-32 | 18k-37k | First held-out-episode evaluation |
| Useful LoRA run | 64-128 | 74k-149k | Train sensor adapters plus selected Qwen3-Omni LoRA |
| Storage-heavy run | 256+ | 297k+ | Only after download layout and checkpoint size are stable |

Use the budget helper before downloading:

```bash
python scripts/omni/plan_finetune_sample_budget.py \
  --storage-root /path/to/storage \
  --target-free-after-download-gb 800 \
  --all-training-per-episode-gb 2.4 \
  --full-preview-per-episode-gb 5.1
```

### 32-Episode Readiness Gate

```bash
python scripts/omni/discover_xperience10m_sources.py \
  --workspace /path/to/ropedia-xperience-10m-task-suite \
  --data-root /path/to/xperience10m_data \
  --output results/omni_finetune/source_discovery.json \
  --report-output results/omni_finetune/DATA_BLOCKER_REPORT.md
```

Current status in this repo:

- local_valid_episodes: 1 (degraded-valid: annotation + fisheye_cam0.mp4)
- local_complete_episodes: 0
- ready_for_32_episode_pilot: false
- planned 32-episode pilot: stratified across 32 top-level session UUIDs
- full-dataset blocker: gated Xperience-10M access is still pending
- source_discovery: `results/omni_finetune/source_discovery.json`
- blocker_report: `results/omni_finetune/DATA_BLOCKER_REPORT.md`
- access_status: `results/omni_finetune/MULTI_EPISODE_ACCESS_STATUS.md`

Use this gate before scheduling any 32-episode full fine-tune run. The pilot
should use stratified selection, not the first 32 paths in repository order.
The current selection plan scans 64 top-level session UUIDs, filters for
complete leaf episodes, excludes `visualization.rrd`, applies a `0.25 GB`
minimum episode size, and selects 32 episodes from 32 different session UUIDs.

### Uploading the pilot Qwen3-Omni LoRA

A prepared upload package is available at `results/omni_finetune/hf_upload`.

```bash
python3 scripts/omni/upload_qwen3_omni_lora_to_hf.py \
  --repo-id cy0307/ropedia-qwen3-omni-lora-readiness \
  --source-dir results/omni_finetune/hf_upload \
  --message "Upload Xperience-10M Qwen3-Omni LoRA pilot"
```

This script requires a valid Hugging Face token via `HF_TOKEN` or `--token`.
Network availability to `huggingface.co` is required.

## Four Research Directions

The 12 tasks are now organized against the four Ropedia research directions in
a generated artifact, not only in prose:

- [`research_direction_taxonomy.json`](results/episode_task_suite/research_directions/research_direction_taxonomy.json)
- [`research_direction_task_map.csv`](results/episode_task_suite/research_directions/research_direction_task_map.csv)
- [`research_direction_summary.md`](results/episode_task_suite/research_directions/research_direction_summary.md)
- [`docs/data/research_directions.json`](docs/data/research_directions.json)

The taxonomy uses two current baselines for every task:

| Baseline | Role |
| --- | --- |
| Minimal interpretable heads | Softmax, logistic, ridge, and retrieval heads over the 8,378-d window feature vector. These expose the input/output contract cleanly. |
| Neural MLP heads | Small PyTorch MLP classifiers/regressors on the same features and splits. These check whether nonlinear heads help before moving to Qwen/Omni fine-tuning. |

Current direction-level coverage:

| Direction | Current status | Covered task evidence | What is not solved yet |
| --- | --- | --- | --- |
| A. Human Modeling & Motion Understanding | Partially implemented | `hand_trajectory_forecast` and `contact_prediction` are direct; `timeline_action` and `object_relevance` are proxies. Neural MLP improves hand forecasting from `0.8223` to `0.1116` MPJPE. | No full body/shape model, SMPL/MANO target, deformation prior, or multi-episode motion-generation evaluation yet. |
| B. 3D/4D Reconstruction & Neural Rendering | Proxy tasks only | `cross_modal_retrieval`, `modality_reconstruction`, and `misalignment_detection` test alignment/reconstruction prerequisites. | No NeRF, Gaussian Splatting, TSDF, mesh, novel-view synthesis, or calibrated 4D reconstruction model yet. |
| C. Egocentric Vision & Interaction | Strongest implemented track | 6 direct tasks: action, subtask, transition, next-action, object relevance, and caption grounding, plus alignment/order diagnostics. | Single-episode chronological split limits generalization; audio and stronger video-language backbones still need to be added. |
| D. Scene Reconstruction & World Modeling | Early proxy tasks | Subtask/next-action, object relevance, retrieval, reconstruction, temporal order, and misalignment provide state/world-model probes. | No persistent scene graph, object permanence task, long-term map, or held-out-episode world model yet. |

The important interpretation is that all four directions can be **started** from
the Xperience-10M sample modalities, but only direction C is strongly represented
by the current 12-task suite. Directions A, B, and D need additional targets and
multi-episode training before they become full research deliverables.

## Four Direction-Extension Probes

Beyond the original 12 core tasks, the repo now includes one extra data-backed
probe for each research direction. These probes are computed from the same
`shared_windows.npz`, `windows.csv`, and `feature_manifest.json` artifacts, so
the reported numbers are real sample-derived metrics, not placeholder results.

- [`research_direction_extension_results.json`](results/episode_task_suite/research_direction_extensions/research_direction_extension_results.json)
- [`research_direction_extension_summary.md`](results/episode_task_suite/research_direction_extensions/research_direction_extension_summary.md)
- [`docs/data/research_direction_extensions.json`](docs/data/research_direction_extensions.json)
- [`research_direction_extension_tasks.svg`](docs/assets/charts/research_direction_extension_tasks.svg)

![Four direction extension probes](docs/assets/charts/research_direction_extension_tasks.svg)

| Direction | New extension task | Input | Output | Minimal | Neural MLP | Why it matters |
| --- | --- | --- | --- | ---: | ---: | --- |
| A. Human Modeling & Motion Understanding | `body_motion_intensity` | non-mocap video/depth/pose/IMU/SLAM/language features | high vs low body/hand motion | `0.7827` macro-F1 | `0.7986` macro-F1 | Starts a human-motion-energy target without leaking mocap input. |
| B. 3D/4D Reconstruction & Neural Rendering | `multi_view_consistency_retrieval` | fisheye camera feature query | synchronized stereo-left view rank | `0.5534` MRR | `0.3469` MRR | Tests whether multi-view features preserve synchronized 4D scene identity. |
| C. Egocentric Vision & Interaction | `action_phase_progress` | non-caption multimodal window | progress inside current action segment | `0.3416` MAE | `0.3038` MAE | Adds a task-structure/intent-style target beyond class labels. |
| D. Scene Reconstruction & World Modeling | `ego_motion_forecast` | current sensors excluding camera translation and captions | future camera-translation delta | `0.1989` MAE | `0.0989` MAE | Starts a short-horizon world-model target over wearer motion. |

Run:

```bash
python scripts/research_direction_extension_tasks.py
```

These four probes make the four-direction mapping more concrete, but they are
still single-episode extension baselines. Full research claims still require
multi-episode training, held-out episode evaluation, and stronger task-specific
models.

## Task Walkthroughs For Juniors

Every task now has a beginner-facing explanation with:

- a concrete coffee-episode case study,
- exact input contract,
- middle process modules,
- output contract,
- minimal and neural metric,
- one important limitation.

Primary files:

- [`TASK_WALKTHROUGHS.md`](results/episode_task_suite/task_walkthroughs/TASK_WALKTHROUGHS.md)
- [`task_walkthroughs.json`](results/episode_task_suite/task_walkthroughs/task_walkthroughs.json)
- [`docs/data/task_walkthroughs.json`](docs/data/task_walkthroughs.json)

Compact map:

| Task | Case study | Input -> process -> output |
| --- | --- | --- |
| `timeline_action` | A pouring window should be named as the current action. | all-modality window -> action label builder + classifier -> action class |
| `timeline_subtask` | A fine action is grouped into a broader drink-preparation stage. | all-modality window -> subtask label builder + classifier -> subtask label |
| `transition_detection` | Detect the change from preparing to pouring. | window -> boundary builder + binary classifier -> boundary/steady |
| `next_action` | A preparing window predicts what happens 20 frames later. | current window -> future-label shift + classifier -> next action |
| `hand_trajectory_forecast` | A hand moving toward a cup becomes a future 3D hand path. | current window -> future mocap target + regressor -> hand trajectory |
| `contact_prediction` | Decide whether hand/body contact is happening. | non-contact features -> contact target + binary classifier -> contact label |
| `object_relevance` | Infer milk, cup, coffee, or related objects during pouring. | non-caption features -> multi-hot object target + sigmoid heads -> object set |
| `caption_grounding` | Query Pour milk into coffee and retrieve the matching moment. | text-like query + candidates -> projection + cosine ranker -> ranked windows |
| `cross_modal_retrieval` | Motion/IMU from pouring retrieves matching depth/video. | motion/IMU/camera -> projection + candidate index -> ranked depth/video windows |
| `modality_reconstruction` | Infer depth/video features from motion, IMU, and camera pose. | source modalities -> scaler + regressor -> target modality vector |
| `temporal_order` | Tell whether reaching then pouring was reversed. | adjacent window pair -> pair combiner + binary classifier -> correct/reversed |
| `misalignment_detection` | Catch motion paired with visual/depth features shifted in time. | motion side + visual side -> aligned/shifted pair builder + classifier -> aligned/shifted |

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

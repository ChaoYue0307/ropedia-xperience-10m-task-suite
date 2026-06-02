# Artifact Guide

This guide is the human-readable map for the public Ropedia Xperience-10M task
suite artifacts. It complements the machine-readable
[`docs/data/artifact_index.json`](docs/data/artifact_index.json).

The project separates these reading layers:

1. **Project status:** one compact table for first-pass current-state
   decisions.
2. **Project scope:** what is implemented now, what is setup-stage, and what
   remains gated by multi-episode data access.
3. **Official source alignment:** what the upstream Xperience-10M dataset card,
   public sample card, and HF API metadata say, and which parts this repo
   currently covers.
4. **Evaluation protocol:** windowing, split policy, per-task metrics, leakage
   controls, and current limitations.
5. **Visual evidence:** public figures, charts, modality thumbnails, dimensions,
   hashes, roles, and source scripts.
6. **Data contract:** how one public Xperience-10M sample episode becomes
   aligned model windows and feature blocks.
7. **Task evidence:** minimal and neural results for the 12 task contracts plus
   four research-direction extension probes.
8. **Reproducibility:** public commands, expected outputs, and exact-match
   evidence for the single-episode pipeline.
9. **Public project surface:** repo, website, and Hugging Face pages,
   accessibility semantics, links, and reader-facing copy.
10. **Scale-up readiness:** scripts and reports for the planned 32-episode
   Qwen3-Omni pilot, with the data-access requirement kept visible.

## Start Here

| Artifact | Why to open it first |
| --- | --- |
| [`PROJECT_STATUS.md`](PROJECT_STATUS.md) | Gives the fastest current-state table: implemented, data-gated, and outside current scope. |
| [`EVIDENCE_CONTRACT.md`](EVIDENCE_CONTRACT.md) | Defines the implemented scope, setup-stage artifacts, and multi-episode prerequisites. |
| [`QUALITY_GATES.md`](QUALITY_GATES.md) | Lists the automated release checks and post-publish verification used to keep the release current. |
| [`PUBLIC_SURFACE_QA.md`](PUBLIC_SURFACE_QA.md) | Describes whether repo, website, and Hugging Face cards read as one coherent research project surface. |
| [`EVALUATION_PROTOCOL.md`](EVALUATION_PROTOCOL.md) | Defines the task unit, chronological split, metrics, leakage controls, and current limitations. |
| [`XPERIENCE10M_DATASET_CARD_ALIGNMENT.md`](XPERIENCE10M_DATASET_CARD_ALIGNMENT.md) | Aligns this repo's public dataset wording with the official gated Xperience-10M card, sample card, and HF API metadata. |
| [`SOURCE_ALIGNMENT_AUDIT.md`](SOURCE_ALIGNMENT_AUDIT.md) | Verifies source-alignment markers across repo, website, and HF cards. |
| [`FIGURE_INDEX.md`](FIGURE_INDEX.md) | Catalogs public figures, charts, modality thumbnails, dimensions, hashes, roles, and source scripts. |
| [`docs/data/brand_assets.json`](docs/data/brand_assets.json) | Catalogs the generated logo system, favicon, app icon, social card, dimensions, hashes, and usage roles. |
| [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) | Defines public reproduction commands, expected outputs, and unreproducible boundaries. |
| [`docs/data/artifact_index.json`](docs/data/artifact_index.json) | Lists project-critical files with existence, size, and stable hashes. |
| [`docs/data/figure_index.json`](docs/data/figure_index.json) | Machine-readable visual asset index for website and HF mirrors. |
| [`docs/data/project_status.json`](docs/data/project_status.json) | Machine-readable copy of the project status table. |
| [`docs/data/xperience10m_dataset_card_alignment.json`](docs/data/xperience10m_dataset_card_alignment.json) | Machine-readable source-alignment summary, including gated metadata, sample license/tooling, and current project coverage. |
| [`docs/data/source_alignment_audit.json`](docs/data/source_alignment_audit.json) | Machine-readable source-fact and current-project marker report. |
| [`docs/data/evaluation_protocol.json`](docs/data/evaluation_protocol.json) | Machine-readable evaluation protocol generated from committed metrics. |
| [`docs/data/quality_gates.json`](docs/data/quality_gates.json) | Machine-readable release-check summary for website and HF mirrors. |
| [`docs/data/public_surface_qa.json`](docs/data/public_surface_qa.json) | Machine-readable public project-surface report for website, repo, and Hugging Face pages. |
| [`docs/data/live_publication_status.json`](docs/data/live_publication_status.json) | Last live GitHub/HF verification after upload. |
| [`docs/data/mirror_parity.json`](docs/data/mirror_parity.json) | Confirms prepared HF Space, artifact, and model mirrors match the repo for critical data, figures, website HTML, and validator scripts. |
| [`docs/data/publication_audit.json`](docs/data/publication_audit.json) | Records whether public bundles exclude raw data, Python caches, heavy archives, token strings, and stale public-card figure references. |
| [`docs/data/scope_claims_audit.json`](docs/data/scope_claims_audit.json) | Keeps historical `32ep` setup identifiers separate from completed held-out-episode results. |
| [`docs/data/task_surface_integrity.json`](docs/data/task_surface_integrity.json) | Confirms the public 12-task cards use readable task names, modality thumbnails, and the interactive walkthrough/player data contract. |
| [`docs/data/website_integrity.json`](docs/data/website_integrity.json) | Confirms local site links, anchors, JSON bundles, and referenced images resolve. |
| [`docs/data/project_packet.json`](docs/data/project_packet.json) | Gives the shortest machine-readable project route. |

## Official Source Alignment

| Artifact | What it proves |
| --- | --- |
| [`XPERIENCE10M_DATASET_CARD_ALIGNMENT.md`](XPERIENCE10M_DATASET_CARD_ALIGNMENT.md) | Human-readable summary of the official gated Xperience-10M dataset card, public sample card, API listing snapshot, scale, modalities, access boundary, intended uses, and limitations. |
| [`docs/data/xperience10m_dataset_card_alignment.json`](docs/data/xperience10m_dataset_card_alignment.json) | Machine-readable copy of the same alignment facts for website and HF mirrors. |
| [`SOURCE_ALIGNMENT_AUDIT.md`](SOURCE_ALIGNMENT_AUDIT.md) | Generated source-alignment report showing source facts, sample license/tooling, API-listing caveat, and current project scope. |
| [`docs/data/source_alignment_audit.json`](docs/data/source_alignment_audit.json) | Machine-readable source metadata, current-project marker, and HF card parity report. |
| [`scripts/validate_source_alignment.py`](scripts/validate_source_alignment.py) | Regenerates the source-alignment report from committed alignment facts and public card text. |

## Evaluation Protocol

| Artifact | What it proves |
| --- | --- |
| [`EVALUATION_PROTOCOL.md`](EVALUATION_PROTOCOL.md) | Human-readable task protocol: window unit, chronological split, input/target contracts, primary metrics, leakage controls, and current limitations. |
| [`docs/data/evaluation_protocol.json`](docs/data/evaluation_protocol.json) | Machine-readable protocol generated from committed task metrics. |
| [`scripts/build_evaluation_protocol.py`](scripts/build_evaluation_protocol.py) | Regenerates the protocol from `docs/data/summary_metrics.json` and source task artifacts. |

## Visual Evidence

| Artifact | What it proves |
| --- | --- |
| [`FIGURE_INDEX.md`](FIGURE_INDEX.md) | Human-readable catalog of public visual assets, dimensions, hashes, roles, and source scripts. |
| [`docs/data/figure_index.json`](docs/data/figure_index.json) | Machine-readable visual asset index mirrored to the website, artifact dataset, and model repo. |
| [`scripts/build_figure_index.py`](scripts/build_figure_index.py) | Regenerates visual-asset hashes, dimensions, and source-script provenance. |
| [`docs/data/brand_assets.json`](docs/data/brand_assets.json) | Machine-readable logo/brand manifest for the website, README, Hugging Face cards, favicon, app icon, and social preview. |
| [`docs/assets/brand/xperience10m-logo-social-card.png`](docs/assets/brand/xperience10m-logo-social-card.png) | Project logo card used by README and Hugging Face cards. |
| [`scripts/build_brand_assets.py`](scripts/build_brand_assets.py) | Regenerates deterministic logo derivatives, favicon variants, app icons, and the social card from the generated logo mark. |
| [`docs/assets/task_suite_infographic.png`](docs/assets/task_suite_infographic.png) | Primary 12-task suite map with sample modality thumbnails. |
| [`docs/assets/pipeline_diagram.png`](docs/assets/pipeline_diagram.png) | Episode-to-task pipeline overview. |
| [`docs/assets/task_architectures.png`](docs/assets/task_architectures.png) | Minimal and neural task-head architecture map. |

## Data Contract

| Artifact | What it proves |
| --- | --- |
| [`results/episode_task_suite/windows.csv`](results/episode_task_suite/windows.csv) | The sample episode is converted into 1,161 aligned 20-frame windows. |
| [`results/episode_task_suite/feature_manifest.json`](results/episode_task_suite/feature_manifest.json) | The current input vector has 8,378 dimensions with explicit feature-block boundaries. |
| [`results/episode_task_suite/available_modalities.json`](results/episode_task_suite/available_modalities.json) | The sample modality coverage is recorded, including the current audio-featurization boundary. |
| [`docs/data/modality_atlas.json`](docs/data/modality_atlas.json) | The responsive website modality cards and derived thumbnail assets are documented without redistributing raw data. |
| [`docs/assets/modalities/`](docs/assets/modalities/) | Small public-sample thumbnails used by the readable modality atlas. |

## Task Evidence

| Artifact | What it proves |
| --- | --- |
| [`results/episode_task_suite/summary_report.json`](results/episode_task_suite/summary_report.json) | The 12 task contracts, chronological split, and minimal/neural metrics. |
| [`results/episode_task_suite/neural_mlp/`](results/episode_task_suite/neural_mlp/) | Matching PyTorch MLP heads for the same task contracts and feature windows. |
| [`results/episode_task_suite/research_directions/`](results/episode_task_suite/research_directions/) | Mapping from the 12 tasks to the four Ropedia research directions. |
| [`results/episode_task_suite/research_direction_extensions/`](results/episode_task_suite/research_direction_extensions/) | Four additional coded probes, one per research direction. |
| [`results/episode_task_suite/task_walkthroughs/`](results/episode_task_suite/task_walkthroughs/) | Human-readable research names and case studies explaining input, process modules, output, metric, limitation, and the website task-player data. |
| [`scripts/validate_task_surface.py`](scripts/validate_task_surface.py) | Fails publication if public task cards drift back to raw artifact ids or lose their thumbnail/player wiring. |

## Reproducibility

| Artifact | What it proves |
| --- | --- |
| [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) | Public commands, expected outputs, and non-reproducible boundaries are explicit. |
| [`docs/data/reproducibility_matrix.json`](docs/data/reproducibility_matrix.json) | Machine-readable command matrix for website and HF mirrors. |
| [`notes/reproducibility_audit.md`](notes/reproducibility_audit.md) | The last exact metric rebuild reproduced the public-sample metrics and matched committed artifacts. |

## Platform Mirrors

| Surface | Purpose |
| --- | --- |
| [GitHub Pages dashboard](https://chaoyue0307.github.io/ropedia-xperience-10m-task-suite/) | Primary public website and visual research flow. |
| [Hugging Face Space](https://huggingface.co/spaces/cy0307/ropedia-xperience-10m-task-suite) | Static app mirror for HF users. |
| [HF artifact dataset](https://huggingface.co/datasets/cy0307/ropedia-xperience-10m-task-suite-artifacts) | Derived CSV/JSON/Markdown/figure artifacts without raw Xperience-10M data. |
| [HF baseline model repo](https://huggingface.co/cy0307/ropedia-xperience-10m-task-baselines) | Lightweight minimal and neural task-head model files. |
| [HF collection](https://huggingface.co/collections/cy0307/ropedia-xperience-10m-task-suite) | One grouped landing page for the Space, artifact dataset, and baseline model repo. |

| Public surface artifact | What it keeps aligned |
| --- | --- |
| [`PUBLIC_SURFACE_QA.md`](PUBLIC_SURFACE_QA.md) | Human-readable public project-surface report for repo, website, and Hugging Face cards. |
| [`docs/data/public_surface_qa.json`](docs/data/public_surface_qa.json) | Machine-readable report for SEO/social metadata, accessible tabs, public links, project links, and reader-facing copy. |
| [`scripts/build_public_surface_qa.py`](scripts/build_public_surface_qa.py) | Regenerates the public project-surface report before release. |

## Scale-Up Readiness

| Artifact | Current status |
| --- | --- |
| [`results/omni_finetune/DATA_BLOCKER_REPORT.md`](results/omni_finetune/DATA_BLOCKER_REPORT.md) | Documents the data-access requirement before the 32-episode Qwen3-Omni pilot can run. |
| [`results/omni_finetune/MULTI_EPISODE_ACCESS_STATUS.md`](results/omni_finetune/MULTI_EPISODE_ACCESS_STATUS.md) | Documents the public multi-episode access path and selected 32-episode pilot plan without private infrastructure details. |
| [`scripts/omni/discover_xperience10m_sources.py`](scripts/omni/discover_xperience10m_sources.py) | Discovery gate for valid multi-episode Xperience-10M sources. |
| [`scripts/omni/train_qwen3_omni_lora.py`](scripts/omni/train_qwen3_omni_lora.py) | Training entrypoint for the Qwen3-Omni LoRA pilot after the data gate passes. |

## What Is Not Included

The public repo and Hugging Face mirrors do not redistribute raw Xperience-10M
videos, raw `annotation.hdf5`, gated private dataset files, full Qwen weights,
or large full checkpoints. Dataset use remains governed by the official
Ropedia/Xperience-10M terms.

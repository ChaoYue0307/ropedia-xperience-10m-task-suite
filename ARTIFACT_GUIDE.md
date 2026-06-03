# Artifact Guide

This guide is the human-readable map for the public Ropedia Xperience-10M task
suite artifacts. It complements the machine-readable
[`docs/data/artifact_index.json`](docs/data/artifact_index.json).

The project separates these reading layers:

1. **Project status:** one compact table for first-pass current-state
   decisions.
2. **Project scope and roadmap:** what is implemented now, what is setup-stage,
   what remains gated by multi-episode data access, and how the staged research
   path progresses.
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
   audio contribution variants, and four research-direction
   extension probes.
8. **Reproducibility:** public commands, expected outputs, and exact-match
   evidence for the single-episode pipeline.
9. **Public project surface:** repo, website, and Hugging Face pages,
   accessibility semantics, links, and reader-facing copy.
10. **Multi-episode pilot status:** scripts and reports for the selected-episode
   Qwen3-Omni pilot, with the data-access requirement kept visible.
11. **Foundation-model selection:** Qwen3-Omni, Cosmos 3, GR00T, OpenVLA,
   openpi, Gemini Robotics, and lightweight policy candidates separated by
   task fit and current evidence level.

## Start Here

| Artifact | Why to open it first |
| --- | --- |
| [`PROJECT_STATUS.md`](PROJECT_STATUS.md) | Gives the fastest current-state table: implemented, in staging, and outside current scope. |
| [`RESEARCH_ROADMAP.md`](RESEARCH_ROADMAP.md) | Shows the staged path from public-sample task development to multi-episode data staging, Qwen3-Omni LoRA, robustness runs, and larger omni-model extensions. |
| [`FOUNDATION_MODEL_PLAN.md`](FOUNDATION_MODEL_PLAN.md) | Explains which foundation backbones fit which Xperience-10M objective: Qwen3-Omni first, Cosmos 3 for world modeling, and VLA/policy models after action-target conversion. |
| [`EVIDENCE_CONTRACT.md`](EVIDENCE_CONTRACT.md) | Defines the implemented scope, setup-stage artifacts, and multi-episode prerequisites. |
| [`QUALITY_GATES.md`](QUALITY_GATES.md) | Lists the automated release checks and post-publish verification used to keep the release current. |
| [`PUBLIC_SURFACE_QA.md`](PUBLIC_SURFACE_QA.md) | Describes whether repo, website, and Hugging Face cards read as one cohesive research project surface. |
| [`EVALUATION_PROTOCOL.md`](EVALUATION_PROTOCOL.md) | Defines the task unit, chronological split, metrics, leakage controls, and current limitations. |
| [`XPERIENCE10M_DATASET_CARD_ALIGNMENT.md`](XPERIENCE10M_DATASET_CARD_ALIGNMENT.md) | Aligns this repo's public dataset wording with the official gated Xperience-10M card, sample card, and HF API metadata. |
| [`SOURCE_ALIGNMENT_AUDIT.md`](SOURCE_ALIGNMENT_AUDIT.md) | Summarizes official dataset facts, sample-card facts, API-listing notes, and project coverage across repo, website, and HF cards. |
| [`FIGURE_INDEX.md`](FIGURE_INDEX.md) | Catalogs public figures, charts, modality thumbnails, dimensions, hashes, roles, and source scripts. |
| [`docs/data/brand_assets.json`](docs/data/brand_assets.json) | Catalogs the generated logo system, favicon, app icon, social card, dimensions, hashes, and usage roles. |
| [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) | Defines public reproduction commands, expected outputs, and unreproducible boundaries. |
| [`docs/data/artifact_index.json`](docs/data/artifact_index.json) | Lists project-critical files with existence, size, and stable hashes. |
| [`docs/data/figure_index.json`](docs/data/figure_index.json) | Machine-readable visual asset index for website and HF mirrors. |
| [`docs/data/project_status.json`](docs/data/project_status.json) | Machine-readable copy of the project status table. |
| [`docs/data/research_roadmap.json`](docs/data/research_roadmap.json) | Machine-readable roadmap for website and Hugging Face mirrors. |
| [`docs/data/foundation_model_plan.json`](docs/data/foundation_model_plan.json) | Machine-readable foundation-model selection matrix for website and Hugging Face mirrors. |
| [`docs/data/xperience10m_dataset_card_alignment.json`](docs/data/xperience10m_dataset_card_alignment.json) | Machine-readable source-alignment summary, including gated metadata, sample license/tooling, and current project coverage. |
| [`docs/data/source_alignment_audit.json`](docs/data/source_alignment_audit.json) | Machine-readable source metadata and HF card parity report. |
| [`docs/data/evaluation_protocol.json`](docs/data/evaluation_protocol.json) | Machine-readable evaluation protocol generated from committed metrics. |
| [`results/audio_ablation/AUDIO_ABLATION_SUMMARY.md`](results/audio_ablation/AUDIO_ABLATION_SUMMARY.md) | Shows measured current-audio and raw log-mel replacement deltas across the 12 task contracts. |
| [`docs/data/audio_ablation_summary.json`](docs/data/audio_ablation_summary.json) | Machine-readable audio ablation summary for website and HF mirrors. |
| [`docs/data/quality_gates.json`](docs/data/quality_gates.json) | Machine-readable release-check summary for website and HF mirrors. |
| [`docs/data/public_surface_qa.json`](docs/data/public_surface_qa.json) | Machine-readable public project-surface report for website, repo, and Hugging Face pages. |
| [`docs/data/live_publication_status.json`](docs/data/live_publication_status.json) | Last live GitHub/HF verification after upload. |
| [`docs/data/mirror_parity.json`](docs/data/mirror_parity.json) | Confirms prepared HF Space, artifact, and model mirrors match the repo for critical data, figures, website HTML, and validator scripts. |
| [`docs/data/publication_audit.json`](docs/data/publication_audit.json) | Summarizes public bundle contents and exclusions for raw data, Python caches, heavy archives, token strings, and public-card figure references. |
| [`docs/data/scope_claims_audit.json`](docs/data/scope_claims_audit.json) | Separates setup identifiers from completed held-out-episode results. |
| [`docs/data/task_surface_integrity.json`](docs/data/task_surface_integrity.json) | Confirms the public 12-task cards use readable task names, modality thumbnails, and the interactive walkthrough/player data contract. |
| [`docs/data/website_integrity.json`](docs/data/website_integrity.json) | Confirms local site links, anchors, JSON bundles, and referenced images resolve. |
| [`RENDERED_SITE_CHECK.md`](RENDERED_SITE_CHECK.md) and [`docs/data/rendered_site_check.json`](docs/data/rendered_site_check.json) | Records the latest browser-level page load, tab navigation, walkthrough deep link, player interaction, and console-health check. |
| [`docs/data/project_packet.json`](docs/data/project_packet.json) | Gives the shortest machine-readable project route. |

## Official Source Alignment

| Artifact | What it shows |
| --- | --- |
| [`XPERIENCE10M_DATASET_CARD_ALIGNMENT.md`](XPERIENCE10M_DATASET_CARD_ALIGNMENT.md) | Human-readable summary of the official gated Xperience-10M dataset card, public sample card, API listing snapshot, scale, modalities, access terms, intended uses, and limitations. |
| [`docs/data/xperience10m_dataset_card_alignment.json`](docs/data/xperience10m_dataset_card_alignment.json) | Machine-readable copy of the same alignment facts for website and HF mirrors. |
| [`SOURCE_ALIGNMENT_AUDIT.md`](SOURCE_ALIGNMENT_AUDIT.md) | Generated source-alignment report showing source facts, sample license/tooling, API-listing notes, and current project scope. |
| [`docs/data/source_alignment_audit.json`](docs/data/source_alignment_audit.json) | Machine-readable source metadata and HF card parity report. |
| [`scripts/validate_source_alignment.py`](scripts/validate_source_alignment.py) | Regenerates the source-alignment report from committed alignment facts and public card text. |

## Evaluation Protocol

| Artifact | What it shows |
| --- | --- |
| [`EVALUATION_PROTOCOL.md`](EVALUATION_PROTOCOL.md) | Human-readable task protocol: window unit, chronological split, input/target contracts, primary metrics, leakage controls, and current limitations. |
| [`docs/data/evaluation_protocol.json`](docs/data/evaluation_protocol.json) | Machine-readable protocol generated from committed task metrics. |
| [`scripts/build_evaluation_protocol.py`](scripts/build_evaluation_protocol.py) | Regenerates the protocol from `docs/data/summary_metrics.json` and source task artifacts. |

## Visual Evidence

| Artifact | What it shows |
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

| Artifact | What it shows |
| --- | --- |
| [`results/episode_task_suite/windows.csv`](results/episode_task_suite/windows.csv) | The sample episode is converted into 1,161 aligned 20-frame windows. |
| [`results/episode_task_suite/feature_manifest.json`](results/episode_task_suite/feature_manifest.json) | The current input vector has 8,546 dimensions with explicit modality-group boundaries, including a 168-d audio group. |
| [`results/episode_task_suite/available_modalities.json`](results/episode_task_suite/available_modalities.json) | The sample modality coverage is recorded, including the current audio-featurization status. |
| [`results/audio_ablation/raw_logmel_fisheye_cam0_sr16000_mels64_fft512_hop160.npz`](results/audio_ablation/raw_logmel_fisheye_cam0_sr16000_mels64_fft512_hop160.npz) | Derived 588-d raw log-mel window features decoded from the local public-sample MP4 audio stream; raw audio itself is not redistributed. |
| [`docs/data/modality_atlas.json`](docs/data/modality_atlas.json) | The responsive website modality cards and derived thumbnail assets are documented without redistributing raw data. |
| [`docs/assets/modalities/`](docs/assets/modalities/) | Small public-sample thumbnails used by the readable modality atlas. |

## Task Evidence

| Artifact | What it shows |
| --- | --- |
| [`results/episode_task_suite/summary_report.json`](results/episode_task_suite/summary_report.json) | The 12 task contracts, chronological split, and minimal/neural metrics. |
| [`results/episode_task_suite/neural_mlp/`](results/episode_task_suite/neural_mlp/) | Matching PyTorch MLP heads for the same task contracts and feature windows. |
| [`results/episode_task_suite/research_directions/`](results/episode_task_suite/research_directions/) | Mapping from the 12 tasks to the four Ropedia research directions. |
| [`results/episode_task_suite/research_direction_extensions/`](results/episode_task_suite/research_direction_extensions/) | Four additional coded probes, one per research direction. |
| [`results/episode_task_suite/task_walkthroughs/`](results/episode_task_suite/task_walkthroughs/) | Human-readable research names and case studies explaining input, process modules, output, metric, limitation, and the website task-player data. |
| [`results/audio_ablation/audio_ablation_metrics.csv`](results/audio_ablation/audio_ablation_metrics.csv) | All 72 measured audio rows: 12 tasks times six variants, including no-audio, audio-only, alternate-audio-only, representation replacement, and all-input variants. |
| [`results/audio_ablation/audio_delta_summary.csv`](results/audio_ablation/audio_delta_summary.csv) | Compact per-task audio delta table for quick manual inspection. |
| [`scripts/audio_ablation_and_raw_upgrade.py`](scripts/audio_ablation_and_raw_upgrade.py) | Regenerates audio contribution results from real task-suite artifacts plus the local public-sample MP4. |
| [`scripts/validate_task_surface.py`](scripts/validate_task_surface.py) | Fails publication if public task cards drift back to raw artifact ids or lose their thumbnail/player wiring. |

## Reproducibility

| Artifact | What it shows |
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
| [`results/omni_finetune/DATA_ACCESS_STATUS.md`](results/omni_finetune/DATA_ACCESS_STATUS.md) | Summarizes the staging requirement before the held-out Qwen3-Omni pilot can report metrics. |
| [`results/omni_finetune/MULTI_EPISODE_ACCESS_STATUS.md`](results/omni_finetune/MULTI_EPISODE_ACCESS_STATUS.md) | Documents the public multi-episode access path, selected relay plan, and data requirements. |
| [`scripts/omni/discover_xperience10m_sources.py`](scripts/omni/discover_xperience10m_sources.py) | Discovery gate for valid multi-episode Xperience-10M sources. |
| [`scripts/omni/train_qwen3_omni_lora.py`](scripts/omni/train_qwen3_omni_lora.py) | Training entrypoint for the Qwen3-Omni LoRA pilot after the data gate passes. |
| [`FOUNDATION_MODEL_PLAN.md`](FOUNDATION_MODEL_PLAN.md) | Adds the post-data-gate backbone selection plan: Qwen3-Omni first, Cosmos 3 for world modeling, and OpenVLA/openpi/GR00T for policy/action branches. |
| [`docs/data/foundation_model_plan.json`](docs/data/foundation_model_plan.json) | Machine-readable model-family registry with source links, entry conditions, and evaluation additions. |

## What Is Not Included

The public repo and Hugging Face mirrors do not redistribute raw Xperience-10M
videos, raw `annotation.hdf5`, gated private dataset files, full Qwen weights,
or large full checkpoints. Dataset use remains governed by the official
Ropedia/Xperience-10M terms.

# Artifact Guide

This guide is the human-readable map for the public Ropedia Xperience-10M task
suite artifacts. It is organized around what a reader usually wants to do:
understand the project, inspect the sample episode, compare baselines, read the
task results, follow the Qwen3-Omni scale-up path, and understand the longer
Xperience-native pretraining goal.

## Start Here

| Artifact | Why to open it first |
| --- | --- |
| [`PROJECT_STATUS.md`](PROJECT_STATUS.md) | Gives the fastest current-state table: implemented, being improved, and outside current scope. |
| [`RESEARCH_ROADMAP.md`](RESEARCH_ROADMAP.md) | Shows the roadmap from public-sample task development to multi-episode data preparation, Qwen3-Omni LoRA, robustness runs, model branches, and the future native-pretraining goal. |
| [`FOUNDATION_MODEL_PLAN.md`](FOUNDATION_MODEL_PLAN.md) | Explains which foundation backbones fit which Xperience-10M objective: Qwen3-Omni first, Cosmos 3 for world modeling, and VLA/policy models after action-target conversion. |
| [`OMNI_MODEL_EXTENSION_CONTRACT.md`](OMNI_MODEL_EXTENSION_CONTRACT.md) | Defines the shared manifest, split, evaluation, packaging, and public-safety contract that future Qwen, Cosmos-style, and VLA/policy branches must satisfy. |
| [`ADDITIONAL_DEVELOPMENT_DIRECTIONS.md`](ADDITIONAL_DEVELOPMENT_DIRECTIONS.md) | Records concrete non-backbone development tracks: taxonomy, benchmark protocol, representation learning, skill graphs, affordances, 3D/4D memory, QA, and policy transfer. |
| [`XPERIENCE_EMBODIED_FOUNDATION_MODEL_PRETRAINING.md`](XPERIENCE_EMBODIED_FOUNDATION_MODEL_PRETRAINING.md) | Describes the future full-corpus Xperience Embodied Foundation Model goal, including modules, objectives, staged scale-up, hardware ranges, and evaluation. |
| [`EVALUATION_PROTOCOL.md`](EVALUATION_PROTOCOL.md) | Defines the task unit, chronological split, metrics, leakage controls, and current limitations. |
| [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) | Defines public reproduction commands, expected outputs, and unreproducible boundaries. |
| [`results/audio_ablation/AUDIO_ABLATION_SUMMARY.md`](results/audio_ablation/AUDIO_ABLATION_SUMMARY.md) | Shows measured current-audio and raw log-mel replacement deltas across the 12 task contracts. |
| [`docs/single_episode_explorer.html`](docs/single_episode_explorer.html) | Gives a static window-level explorer for the public sample episode. |
| [`XPERIENCE10M_DATASET_CARD_ALIGNMENT.md`](XPERIENCE10M_DATASET_CARD_ALIGNMENT.md) | Optional detail for readers who need official dataset and access-term context. |

## Dataset Context

| Artifact | What it shows |
| --- | --- |
| [`XPERIENCE10M_DATASET_CARD_ALIGNMENT.md`](XPERIENCE10M_DATASET_CARD_ALIGNMENT.md) | Human-readable summary of the official gated Xperience-10M dataset, public sample, modalities, access terms, intended uses, and limitations. |
| [`docs/data/xperience10m_dataset_card_alignment.json`](docs/data/xperience10m_dataset_card_alignment.json) | Machine-readable dataset-context bundle for the website and Hub pages. |
| [`SOURCE_ALIGNMENT_AUDIT.md`](SOURCE_ALIGNMENT_AUDIT.md) | Supporting provenance note for maintainers who want to inspect how public dataset descriptions were checked. |
| [`docs/data/source_alignment_audit.json`](docs/data/source_alignment_audit.json) | Machine-readable provenance record for generated project pages. |
| [`scripts/validate_source_alignment.py`](scripts/validate_source_alignment.py) | Maintenance script for refreshing the dataset-context note. |

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
| [`docs/data/reproducibility_matrix.json`](docs/data/reproducibility_matrix.json) | Machine-readable command matrix for the website and Hub pages. |
| [`notes/reproducibility_audit.md`](notes/reproducibility_audit.md) | The last exact metric rebuild reproduced the public-sample metrics and matched committed artifacts. |

## Public Pages

| Surface | Purpose |
| --- | --- |
| [GitHub Pages dashboard](https://chaoyue0307.github.io/ropedia-xperience-10m-task-suite/) | Primary public website and visual research flow. |
| [GitHub Container package](https://github.com/ChaoYue0307/ropedia-xperience-10m-task-suite/pkgs/container/ropedia-xperience-10m-task-suite) | Static dashboard image for local browsing with Docker. |
| [Hugging Face Space](https://huggingface.co/spaces/cy0307/ropedia-xperience-10m-task-suite) | Static app mirror for HF users. |
| [HF artifact dataset](https://huggingface.co/datasets/cy0307/ropedia-xperience-10m-task-suite-artifacts) | Derived CSV/JSON/Markdown/figure artifacts without raw Xperience-10M data. |
| [HF baseline model repo](https://huggingface.co/cy0307/ropedia-xperience-10m-task-baselines) | Lightweight minimal and neural task-head model files. |
| [HF collection](https://huggingface.co/collections/cy0307/ropedia-xperience-10m-task-suite) | One grouped landing page for the Space, artifact dataset, and baseline model repo. |

The public pages are meant to be the normal reader path. Supporting maintenance
checks remain in the repo, but they are not required for understanding the
research project.

## Scale-Up Readiness

| Artifact | Current status |
| --- | --- |
| [`results/omni_finetune/DATA_ACCESS_STATUS.md`](results/omni_finetune/DATA_ACCESS_STATUS.md) | Summarizes the data-readiness checks required before a held-out Qwen3-Omni pilot can report metrics. |
| [`results/omni_finetune/MULTI_EPISODE_ACCESS_STATUS.md`](results/omni_finetune/MULTI_EPISODE_ACCESS_STATUS.md) | Documents the public multi-episode access path, selected 128-episode pilot plan, and data requirements. |
| [`docs/data/omni_finetune_verified_result.json`](docs/data/omni_finetune_verified_result.json) | Compact verified summary for the final selected-episode Qwen3-Omni diagnostic result, including split counts, held-out metrics, quality-target status, and adapter repo. |
| [`results/omni_finetune/verified_public/`](results/omni_finetune/verified_public/) | Public-safe verified held-out result packages. These include metrics, predictions, reports, manifests, training metadata, validation summaries, and audit files, but not raw data or weights. |
| [`results/omni_finetune/verified_public/xperience10m_qwen3_omni_128ep_structured_json_v2_reuse_full8gpu_lora_eval_test_full/`](results/omni_finetune/verified_public/xperience10m_qwen3_omni_128ep_structured_json_v2_reuse_full8gpu_lora_eval_test_full/) | Final verified Qwen3-Omni public package with 448 held-out predictions, 99.78% JSON validity, metrics, reports, training metadata, validation summaries, and package audit. |
| [`https://huggingface.co/cy0307/ropedia-qwen3-omni-lora-128ep`](https://huggingface.co/cy0307/ropedia-qwen3-omni-lora-128ep) | Public LoRA adapter weight repository for the final 128-episode Qwen3-Omni diagnostic run; raw Xperience-10M data and base Qwen weights remain excluded. |
| [`results/omni_finetune/multi_episode_128_task_baselines/BASELINE_ALIGNMENT_REPORT.md`](results/omni_finetune/multi_episode_128_task_baselines/BASELINE_ALIGNMENT_REPORT.md) | Same-split 128-episode simple and neural metadata baselines for the 12 task ids, aligned to the 96/16/16 Qwen3-Omni split and explicit about raw-feature-only tasks. |
| [`results/omni_finetune/multi_episode_128_task_baselines/summary_report.json`](results/omni_finetune/multi_episode_128_task_baselines/summary_report.json) | Machine-readable split counts, run configuration, simple metrics, neural metrics, and unsupported raw-feature markers for the aligned 128-episode baseline suite. |
| [`scripts/omni/run_128_task_baselines.py`](scripts/omni/run_128_task_baselines.py) | Runner for the aligned 128-episode metadata/text baselines; it consumes the derived Qwen JSONL export locally but does not publish raw data, Qwen weights, or LoRA weights. |
| [`scripts/omni/discover_xperience10m_sources.py`](scripts/omni/discover_xperience10m_sources.py) | Discovery gate for valid multi-episode Xperience-10M sources. |
| [`scripts/omni/train_qwen3_omni_lora.py`](scripts/omni/train_qwen3_omni_lora.py) | Training entrypoint for the Qwen3-Omni LoRA pilot after the data gate passes. |
| [`scripts/omni/run_128_fullsplit_parallel_export_8gpu.sh`](scripts/omni/run_128_fullsplit_parallel_export_8gpu.sh) | Full 96/16/16 launcher with parallel export, 8-process LoRA training, validation-sample monitoring, held-out test evaluation, and quality-target reporting. |
| [`scripts/omni/merge_qwen3_omni_eval_shards.py`](scripts/omni/merge_qwen3_omni_eval_shards.py) | Recomputes held-out metrics from deterministic Qwen eval shards and checks missing or duplicate prediction ids. |
| [`scripts/omni/package_verified_omni_result.py`](scripts/omni/package_verified_omni_result.py) | Creates a contract-driven public-safe package from validated held-out fine-tuning outputs without raw data, base weights, adapter/checkpoint weights, full checkpoints, or large archives. |
| [`scripts/omni/audit_verified_omni_package.py`](scripts/omni/audit_verified_omni_package.py) | Audits a verified package before README, website, or Hugging Face updates by checking validation status, required files, primary metrics, held-out evidence, and forbidden file types. |
| [`scripts/omni/analyze_qwen3_omni_errors.py`](scripts/omni/analyze_qwen3_omni_errors.py) | Computes public-safe held-out error-analysis tables from the verified Qwen3-Omni prediction package. |
| [`scripts/omni/watch_verified_omni_package.py`](scripts/omni/watch_verified_omni_package.py) | Waits for a passing held-out eval validation and then runs the verified public-safe packager automatically. |
| [`OMNI_MODEL_EXTENSION_CONTRACT.md`](OMNI_MODEL_EXTENSION_CONTRACT.md) | Human-readable contract for adding new model families while preserving the same episode split, held-out evaluation, packaging gate, and public-safety boundary. |
| [`configs/omni_backbones/`](configs/omni_backbones/) | Backbone registry for implemented Qwen3-Omni LoRA plus planned Cosmos-style world-model and VLA/policy branches. |
| [`scripts/omni/backbone_registry.py`](scripts/omni/backbone_registry.py) | Validates each backbone contract, required metrics, required files, split policy, and forbidden public package categories. |
| [`scripts/omni/export_model_neutral_window_index.py`](scripts/omni/export_model_neutral_window_index.py) | Converts Qwen JSONL records into a model-neutral window index that future Cosmos-style and policy/VLA exporters can consume. |
| [`scripts/omni/smoke_test_backbone_packaging.py`](scripts/omni/smoke_test_backbone_packaging.py) | Runs synthetic package-contract checks for every configured backbone, including Qwen3-Omni, Cosmos-style world modeling, and VLA/policy branches. |
| [`scripts/omni/scaffold_omni_backbone.py`](scripts/omni/scaffold_omni_backbone.py) | Creates a validated planned-backbone config from an existing contract template so new model branches inherit split, artifact, and publication rules. |
| [`FOUNDATION_MODEL_PLAN.md`](FOUNDATION_MODEL_PLAN.md) | Adds the post-data-gate backbone selection plan: Qwen3-Omni first, Cosmos 3 for world modeling, and OpenVLA/openpi/GR00T for policy/action branches. |
| [`docs/data/foundation_model_plan.json`](docs/data/foundation_model_plan.json) | Machine-readable model-family registry with source links, entry conditions, and evaluation additions. |
| [`ADDITIONAL_DEVELOPMENT_DIRECTIONS.md`](ADDITIONAL_DEVELOPMENT_DIRECTIONS.md) | Concise reader-facing plan for non-backbone tracks that can be built from Xperience-10M data. |
| [`docs/data/additional_development_directions.json`](docs/data/additional_development_directions.json) | Machine-readable copy of the additional directions for website and Hugging Face surfaces. |
| [`XPERIENCE_EMBODIED_FOUNDATION_MODEL_PRETRAINING.md`](XPERIENCE_EMBODIED_FOUNDATION_MODEL_PRETRAINING.md) | Future full-corpus Xperience-native pretraining plan; not a current model result. |

## What Is Not Included

The public repo and Hugging Face mirrors do not redistribute raw Xperience-10M
videos, raw `annotation.hdf5`, gated private dataset files, full Qwen weights,
or large full checkpoints. Dataset use remains governed by the official
Ropedia/Xperience-10M terms.

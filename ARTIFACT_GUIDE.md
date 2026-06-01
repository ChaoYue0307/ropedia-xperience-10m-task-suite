# Artifact Guide

This guide is the human-readable map for the public Ropedia Xperience-10M task
suite artifacts. It complements the machine-readable
[`docs/data/artifact_index.json`](docs/data/artifact_index.json).

The project intentionally separates four layers:

1. **Proof boundary:** what is claimed, what is smoke-only, and what remains
   gated by data access.
2. **Data contract:** how one public Xperience-10M sample episode becomes
   aligned model windows and feature blocks.
3. **Task evidence:** minimal and neural results for the 12 task contracts plus
   four research-direction extension probes.
4. **Reproducibility:** public commands, expected outputs, and exact-match audit
   evidence for the single-episode pipeline.
5. **Scale-up status:** scripts and reports for the planned 32-episode
   Qwen3-Omni pilot, without claiming those results before data access lands.

## Start Here

| Artifact | Why to open it first |
| --- | --- |
| [`EVIDENCE_CONTRACT.md`](EVIDENCE_CONTRACT.md) | Defines which claims are verified and which are explicitly not claimed. |
| [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) | Defines public reproduction commands, expected outputs, and unreproducible boundaries. |
| [`docs/data/artifact_index.json`](docs/data/artifact_index.json) | Lists reviewer-critical files with existence, size, and stable hashes. |
| [`docs/data/publication_audit.json`](docs/data/publication_audit.json) | Confirms public bundles exclude raw data, Python caches, heavy archives, and token strings. |
| [`docs/data/reviewer_packet.json`](docs/data/reviewer_packet.json) | Gives the shortest machine-readable reviewer route. |

## Data Contract

| Artifact | What it proves |
| --- | --- |
| [`results/episode_task_suite/windows.csv`](results/episode_task_suite/windows.csv) | The sample episode is converted into 1,161 aligned 20-frame windows. |
| [`results/episode_task_suite/feature_manifest.json`](results/episode_task_suite/feature_manifest.json) | The current input vector has 8,378 dimensions with explicit feature-block boundaries. |
| [`results/episode_task_suite/available_modalities.json`](results/episode_task_suite/available_modalities.json) | The sample modality coverage is recorded, including the current audio-featurization boundary. |

## Task Evidence

| Artifact | What it proves |
| --- | --- |
| [`results/episode_task_suite/summary_report.json`](results/episode_task_suite/summary_report.json) | The 12 task contracts, chronological split, and minimal/neural metrics. |
| [`results/episode_task_suite/neural_mlp/`](results/episode_task_suite/neural_mlp/) | Matching PyTorch MLP heads for the same task contracts and feature windows. |
| [`results/episode_task_suite/research_directions/`](results/episode_task_suite/research_directions/) | Mapping from the 12 tasks to the four Ropedia research directions. |
| [`results/episode_task_suite/research_direction_extensions/`](results/episode_task_suite/research_direction_extensions/) | Four additional coded probes, one per research direction. |
| [`results/episode_task_suite/task_walkthroughs/`](results/episode_task_suite/task_walkthroughs/) | Junior-friendly case studies explaining input, process modules, output, metric, and limitation. |

## Reproducibility

| Artifact | What it proves |
| --- | --- |
| [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) | Public commands, expected outputs, and non-reproducible boundaries are explicit. |
| [`docs/data/reproducibility_matrix.json`](docs/data/reproducibility_matrix.json) | Machine-readable command matrix for website and HF mirrors. |
| [`notes/reproducibility_audit.md`](notes/reproducibility_audit.md) | The last exact metric audit rebuilt the public-sample metrics and matched committed artifacts. |

## Platform Mirrors

| Surface | Purpose |
| --- | --- |
| [GitHub Pages dashboard](https://chaoyue0307.github.io/ropedia-xperience-10m-task-suite/) | Primary public website and visual reviewer flow. |
| [Hugging Face Space](https://huggingface.co/spaces/cy0307/ropedia-xperience-10m-task-suite) | Static app mirror for HF users. |
| [HF artifact dataset](https://huggingface.co/datasets/cy0307/ropedia-xperience-10m-task-suite-artifacts) | Derived CSV/JSON/Markdown/figure artifacts without raw Xperience-10M data. |
| [HF baseline model repo](https://huggingface.co/cy0307/ropedia-xperience-10m-task-baselines) | Lightweight minimal and neural task-head model files. |
| [HF collection](https://huggingface.co/collections/cy0307/ropedia-xperience-10m-task-suite) | One grouped landing page for the Space, artifact dataset, and baseline model repo. |

## Scale-Up Boundary

| Artifact | Current status |
| --- | --- |
| [`results/omni_finetune/DATA_BLOCKER_REPORT.md`](results/omni_finetune/DATA_BLOCKER_REPORT.md) | Documents why no real 32-episode Qwen3-Omni result is claimed yet. |
| [`results/omni_finetune/A100_HF_RELAY_STATUS.md`](results/omni_finetune/A100_HF_RELAY_STATUS.md) | Documents the pending A100-to-H20 relay and selected 32-session pilot plan. |
| [`scripts/omni/discover_xperience10m_sources.py`](scripts/omni/discover_xperience10m_sources.py) | Discovery gate for valid multi-episode Xperience-10M sources. |
| [`scripts/omni/train_qwen3_omni_lora.py`](scripts/omni/train_qwen3_omni_lora.py) | Training entrypoint for the Qwen3-Omni LoRA pilot after the data gate passes. |

## What Is Not Included

The public repo and Hugging Face mirrors do not redistribute raw Xperience-10M
videos, raw `annotation.hdf5`, gated private dataset files, full Qwen weights,
or large full checkpoints. Dataset use remains governed by the official
Ropedia/Xperience-10M terms.

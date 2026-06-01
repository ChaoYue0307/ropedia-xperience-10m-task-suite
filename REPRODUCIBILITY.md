# Reproducibility Contract

This file defines what can be reproduced from the public repo and the official
Xperience-10M sample, what each command should produce, and which claims remain
outside the current public data boundary.

## Scope

| Layer | Reproducible now | Boundary |
| --- | --- | --- |
| Sample download | Yes, from `ropedia-ai/xperience-10m-sample` or ModelScope sample mirror | Sample card lists `cc-by-nc-4.0`; raw data is not redistributed in this repo. |
| Minimal baselines | Yes | One public sample episode, chronological split. |
| 12-task suite | Yes | Uses the current 8,378-d feature contract; audio is documented but not featurized. |
| Neural MLP heads | Yes, when `torch` is installed | Compact task heads only, not a foundation model. |
| Website figures and charts | Yes | Generated from committed metrics and sample thumbnails. |
| Publication audit | Yes | Checks public repo and prepared HF bundles. |
| 32-episode Qwen3-Omni LoRA pilot | Not yet | Gated by full Xperience-10M access and held-out-episode evaluation. |

## Environment

Use Python 3.12 when possible. The current public scripts depend on the HOMIE
toolkit environment plus lightweight plotting and Hub tooling.

```bash
git clone https://github.com/Ropedia/HOMIE-toolkit.git
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r HOMIE-toolkit/requirements.txt huggingface_hub hf_xet
pip install -r ropedia-xperience-10m-task-suite/requirements.txt
pip install torch
```

## Data

Download the public sample from Hugging Face:

```bash
hf download ropedia-ai/xperience-10m-sample \
  --repo-type dataset \
  --local-dir data/sample/xperience-10m-sample
```

On mainland-China servers, use the included ModelScope helper:

```bash
python scripts/omni/download_sample_modelscope.py \
  --output-dir data/sample/xperience-10m-sample \
  --mode all-training
```

`--mode all-training` downloads `annotation.hdf5` and the six MP4 streams while
skipping `visualization.rrd`.

The sample card points to HOMIE Toolkit for inspecting videos and annotations.
When `visualization.rrd` is downloaded for human inspection, open it with Rerun
0.29.0. The `.rrd` viewer artifact is not used by the training/evaluation
scripts and is excluded from public publication bundles.

## Core Commands

Run these from the repo root after setting `WORKSPACE` to the folder that owns
`data/sample/xperience-10m-sample`.

```bash
export WORKSPACE=/path/to/workspace

python scripts/train_min_action_model.py --workspace "$WORKSPACE"
python scripts/train_all_modalities_model.py --workspace "$WORKSPACE"

python scripts/episode_task_suite.py \
  --workspace "$WORKSPACE" \
  --include-neural

python scripts/research_direction_taxonomy.py
python scripts/research_direction_extension_tasks.py
python scripts/task_walkthroughs.py
python scripts/build_evaluation_protocol.py
python scripts/generate_visualizations.py
python scripts/render_overview_figures.py
python scripts/render_task_suite_infographic.py
python scripts/export_modality_atlas_assets.py
python scripts/validate_website_integrity.py
python scripts/validate_scope_claims.py
python scripts/build_artifact_index.py
python scripts/validate_mirror_parity.py
python scripts/validate_publication_package.py
```

## Expected Public Outputs

| Command group | Expected artifacts |
| --- | --- |
| Minimal baselines | `results/min_action_model/`, `results/min_all_modalities_action_model/`, metrics and model weights |
| 12-task suite | `results/episode_task_suite/summary_report.json`, per-task `metrics.json`, predictions, confusion matrices |
| Neural heads | `results/episode_task_suite/neural_mlp/**/metrics.json`, histories, model checkpoints |
| Research directions | `results/episode_task_suite/research_directions/`, `docs/data/research_directions.json` |
| Direction probes | `results/episode_task_suite/research_direction_extensions/`, `docs/data/research_direction_extensions.json` |
| Walkthroughs | `results/episode_task_suite/task_walkthroughs/`, `docs/data/task_walkthroughs.json` |
| Evaluation protocol | `EVALUATION_PROTOCOL.md`, `docs/data/evaluation_protocol.json` |
| Figures | `docs/assets/*.png`, `docs/assets/charts/*.svg` |
| Modality atlas | `docs/data/modality_atlas.json`, `docs/assets/modalities/*` |
| Website integrity | `docs/data/website_integrity.json` |
| Publication checks | `docs/data/artifact_index.json`, `docs/data/mirror_parity.json`, `docs/data/publication_audit.json`, `docs/data/scope_claims_audit.json` |

## Exact-Match Audit

The last full metric reproducibility audit was run on **2026-05-30
Asia/Singapore** from a fresh output directory outside the repo. It rebuilt the
minimal baselines, all-modality baselines, and the 12-task suite from the local
public sample. The regenerated metrics matched the committed artifacts after
float normalization.

Evidence:

- [`notes/reproducibility_audit.md`](notes/reproducibility_audit.md)
- [`docs/data/reproducibility_matrix.json`](docs/data/reproducibility_matrix.json)

## Non-Reproducible From This Public Repo Alone

The following require gated data, large model weights, or private compute
state, so this repo does not claim they are publicly reproducible yet:

- a real 32-episode Qwen3-Omni LoRA run,
- held-out episode metrics for Qwen3-Omni,
- full Xperience-10M-scale pretraining,
- raw Xperience-10M video or annotation redistribution,
- full Qwen weights or large full checkpoints.

Before interpreting any Qwen3-Omni result, read
[`docs/data/scope_claims_audit.json`](docs/data/scope_claims_audit.json),
[`results/omni_finetune/DATA_BLOCKER_REPORT.md`](results/omni_finetune/DATA_BLOCKER_REPORT.md)
and
[`results/omni_finetune/A100_HF_RELAY_STATUS.md`](results/omni_finetune/A100_HF_RELAY_STATUS.md).

# Reproducibility Contract

This file defines what can be reproduced from the public repo and the official
Xperience-10M sample, what each command should produce, and which results remain
outside the current public data scope.

## Scope

| Layer | Reproducible now | Current scope |
| --- | --- | --- |
| Sample download | Yes, from `ropedia-ai/xperience-10m-sample` or ModelScope sample mirror | Sample card lists `cc-by-nc-4.0`; raw data is not redistributed in this repo. |
| Minimal baselines | Yes | One public sample episode, chronological split. |
| 12-task suite | Yes | Uses the current 8,546-d synchronized multimodal feature contract. |
| Neural MLP heads | Yes, when `torch` is installed | Compact task heads only, not a foundation model. |
| Website figures and charts | Yes | Generated from committed metrics and sample thumbnails. |
| Public bundle contents | Yes | Covers public repo and prepared HF bundles. |
| Multi-episode Qwen3-Omni LoRA pilot | Yes, as a public-safe verified result package | The selected 96/16/16 episode split produced verified held-out packages; the latest v6 package records 34,269 exported multiscale windows and 4,032 held-out predictions. Public readers can inspect the package, but rerunning requires gated Xperience data and base-model weights. |
| Owner-side staged Qwen3-Omni v6 reproduction | Yes, on the private staged GPU host only | The staged host has the exported media cache, path-rewritten JSONL, Qwen3-Omni base-model cache, v6 adapter, HF mirrors, and a one-sample smoke with `exit_code=0` on 2026-06-14. |

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

If Hugging Face access is unavailable in your environment, use the included
ModelScope helper:

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
python scripts/validate_source_alignment.py
python scripts/build_evaluation_protocol.py
python scripts/generate_visualizations.py
python scripts/render_overview_figures.py
python scripts/render_task_suite_infographic.py
python scripts/export_modality_atlas_assets.py
python scripts/build_brand_assets.py
python scripts/build_figure_index.py
python scripts/validate_website_integrity.py
python scripts/validate_task_surface.py
python scripts/validate_scope_claims.py
python scripts/build_artifact_index.py
python scripts/validate_mirror_parity.py
python scripts/validate_publication_package.py
```

## Owner-Side Staged Qwen3-Omni v6 Reproduction

This section is for the private staged GPU host, not for public reruns from the
GitHub repo alone. It preserves the verified result path after the original
training host is released.

Expected private staging layout:

| Item | Staged path |
| --- | --- |
| Staging root | `/mnt/kgc/chaoyue/ropedia-h20-side` |
| Repo | `<staged-repo-root>` |
| Qwen3-Omni base model | `/mnt/kgc/chaoyue/ropedia-h20-side/modelscope_models/Qwen__Qwen3-Omni-30B-A3B-Instruct` |
| v6 adapter | `checkpoints/xperience10m_qwen3_omni_128ep_multiscale_cap96_v6_rank64_lr5e5_full8gpu_lora/adapter_lora` |
| Staged eval JSONL | `results/omni_finetune/xperience10m_qwen3_omni_128ep_multiscale_cap96_v5_full8gpu_lora_dataset/dataset_a100_eval.jsonl` |
| Private handoff manifest | `/mnt/kgc/chaoyue/ropedia-h20-side/STAGING_MANIFEST_20260614.md` |

The staged JSONL has the same 34,269 rows as the original export JSONL, with
exported media paths rewritten from the training-host repo root to the private
staging root. Raw upstream Xperience-10M source files are not required for this
train/eval cache reproduction and were not copied because the selected raw
source tree is about 278 GB.

Run this from the staged repo:

```bash
cd <staged-repo-root>
CUDA_VISIBLE_DEVICES=0,1,2,3 \
RUN_ID=a100_repro_qwen_v6_eval_smoke1_manual \
SAMPLE_LIMIT=1 \
MAX_NEW_TOKENS=1 \
scripts/omni/run_private_gpu_qwen3_v6_repro_smoke.sh
```

The launcher first applies/checks the narrow Transformers Qwen3-Omni
video-feature compatibility patch. The expected compatible installed source
hash is `da5feea4afc11767db3ca7eedb85ac129c66605643dadc6272c4288b03be7d25`;
the known incompatible pre-patch hash is
`2aa5752c32965dbaeee230a016afbbbb30d459a46a12c88c1d6f712e12ba95ad`.

Verified staged-GPU smoke evidence from 2026-06-14:

| Field | Value |
| --- | --- |
| Run id | `a100_repro_qwen_v6_eval_smoke1_h20compat_tok1_20260614` |
| Exit code | `0` |
| Samples | `1` |
| JSON validity | `1.0` |
| Transition accuracy | `1.0` |
| Contact accuracy | `1.0` |
| Object micro-F1 | `0.28571428571428575` |
| Metrics path | `results/omni_finetune/a100_repro_qwen_v6_eval_smoke1_h20compat_tok1_20260614/metrics.json` |

## Expected Public Outputs

| Command group | Expected artifacts |
| --- | --- |
| Minimal baselines | `results/min_action_model/`, `results/min_all_modalities_action_model/`, metrics and model weights |
| 12-task suite | `results/episode_task_suite/summary_report.json`, per-task `metrics.json`, predictions, confusion matrices |
| Neural heads | `results/episode_task_suite/neural_mlp/**/metrics.json`, histories, model checkpoints |
| Research directions | `results/episode_task_suite/research_directions/`, `docs/data/research_directions.json` |
| Direction probes | `results/episode_task_suite/research_direction_extensions/`, `docs/data/research_direction_extensions.json` |
| Walkthroughs | `results/episode_task_suite/task_walkthroughs/`, `docs/data/task_walkthroughs.json` |
| Task surface integrity | `docs/data/task_surface_integrity.json` |
| Source alignment | `SOURCE_ALIGNMENT_AUDIT.md`, `docs/data/source_alignment_audit.json` |
| Evaluation protocol | `EVALUATION_PROTOCOL.md`, `docs/data/evaluation_protocol.json` |
| Figures | `docs/assets/*.png`, `docs/assets/charts/*.svg` |
| Brand assets | `docs/assets/brand/*.png`, `docs/favicon.png`, `docs/apple-touch-icon.png`, `docs/data/brand_assets.json` |
| Figure index | `FIGURE_INDEX.md`, `docs/data/figure_index.json` |
| Modality atlas | `docs/data/modality_atlas.json`, `docs/assets/modalities/*` |
| Website integrity | `docs/data/website_integrity.json` |
| Release reports | `docs/data/artifact_index.json`, `docs/data/mirror_parity.json`, `docs/data/publication_audit.json`, `docs/data/scope_claims_audit.json` |

## Exact-Match Reproduction Record

The last full metric reproduction run was completed on **2026-05-30
Asia/Singapore** from a fresh output directory outside the repo. It rebuilt the
minimal baselines, all-modality baselines, and the 12-task suite from the local
public sample. The regenerated metrics matched the committed artifacts after
float normalization.

Evidence:

- [`notes/reproducibility_audit.md`](notes/reproducibility_audit.md)
- [`docs/data/reproducibility_matrix.json`](docs/data/reproducibility_matrix.json)

## Non-Reproducible From This Public Repo Alone

The following require gated data, large model weights, or private compute
state, so this repo does not provide public reproduction for:

- rerunning the multi-episode Qwen3-Omni LoRA pilot from raw gated data,
- full Xperience-10M-scale pretraining,
- raw Xperience-10M video or annotation redistribution,
- full Qwen weights or large full checkpoints.

Before interpreting any Qwen3-Omni result, read
[`docs/data/scope_claims_audit.json`](docs/data/scope_claims_audit.json),
[`results/omni_finetune/DATA_ACCESS_STATUS.md`](results/omni_finetune/DATA_ACCESS_STATUS.md)
and
[`results/omni_finetune/MULTI_EPISODE_ACCESS_STATUS.md`](results/omni_finetune/MULTI_EPISODE_ACCESS_STATUS.md).

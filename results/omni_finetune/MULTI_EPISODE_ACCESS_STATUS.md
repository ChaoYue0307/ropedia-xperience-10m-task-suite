# Multi-Episode Access Status

Current status: access to the gated full `ropedia-ai/xperience-10m` dataset is
granted, and a metadata-only Hugging Face audit has been completed. A selected
128-episode pilot has produced a verified diagnostic Qwen3-Omni LoRA package
with held-out evaluation. The result is useful as a pipeline and error-analysis
baseline, not as a strong final model.

This file records the public data-access status and pilot requirements. It does
not include local-machine aliases, private paths, SSH hosts, or token locations.

## Selection Plan

| Item | Value |
| --- | ---: |
| Dataset | `ropedia-ai/xperience-10m` |
| Minimum pilot gate | 32 complete leaf episodes |
| Strategy | stratified round-robin across top-level session UUIDs |
| Metadata-audited visible complete episodes | 12,102 |
| Metadata-audited complete sessions | 802 |
| Current selected pilot | 128 source-balanced episodes |
| Recommended split | 96 train / 16 val / 16 test |
| Recommended estimated download | 277.71 GiB excluding `visualization.rrd` |
| Representative 32-episode estimate | ~70.5 GiB at median episode size |
| Smallest one-per-session 32-episode estimate | 35.35 GiB |
| Excluded file | `visualization.rrd` |

## Current Stage

The current Qwen3-Omni artifacts include a verified validation-monitored
diagnostic held-out run: 96/16/16 selected train/val/test episodes, 3,808
exported windows, 2,848 train examples, 512 validation examples, and 448
held-out test predictions from 14 exported test episodes. Training used eight
distributed accelerator processes for one epoch with LoRA rank 16 and recorded a
final train loss of 0.4130 plus a validation loss of 0.0331. The result verifies
the multi-episode pipeline and gives a real error-analysis baseline; it is still
not a strong final model.

A stronger model-quality pilot should be claimed only after:

- selected valid episodes are available locally,
- the manifest builder confirms complete held-out episode splits,
- training finishes with recorded metadata and progress logs,
- evaluation runs on held-out test episodes,
- predictions, metrics, confusion matrices, and a run report are committed.
- JSON validity and action/subtask metrics improve beyond the current
  diagnostic baseline.

Current diagnostic metrics:

- JSON validity: 87.50%
- action macro-F1: 0.0027
- subtask accuracy: 0.0067
- transition accuracy: 0.8504
- next-action accuracy: 0.0246
- contact accuracy: 0.6451
- object micro-F1: 0.2230

The public data access summary is:

`results/omni_finetune/DATA_ACCESS_STATUS.md`

The current metadata-only full dataset audit is:

`results/omni_finetune/FULL_DATASET_METADATA_AUDIT.md`

The current 128-episode source-balanced download plan is:

`results/omni_finetune/XPERIENCE10M_128_EPISODE_SELECTION.md`

The current verified diagnostic package is:

`docs/data/omni_finetune_verified_result.json`

`results/omni_finetune/verified_public/`

The older machine-generated source discovery report remains a pre-access local
planning record:

`results/omni_finetune/DATA_BLOCKER_REPORT.md`

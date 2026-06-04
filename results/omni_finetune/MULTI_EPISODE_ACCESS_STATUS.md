# Multi-Episode Access Status

Current status: access to the gated full `ropedia-ai/xperience-10m` dataset is
granted, and a metadata-only Hugging Face audit has been completed. A selected
128-episode pilot has been defined, but the multi-episode fine-tune has not
completed preprocessing, training, or held-out evaluation yet.

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
| Current selected pilot | 128 metadata-balanced episodes |
| Recommended split | 96 train / 16 val / 16 test |
| Recommended estimated download | 277.71 GiB excluding `visualization.rrd` |
| Representative 32-episode estimate | ~70.5 GiB at median episode size |
| Smallest one-per-session 32-episode estimate | 35.35 GiB |
| Excluded file | `visualization.rrd` |

## Current Stage

The current Qwen3-Omni artifacts come from the locally available sample data.
The held-out model-quality run starts after selected complete episodes are
available, validated for modality coverage, reviewed for content balance, and
preprocessed into train/validation/test examples. This is still a preparation
stage, not a model-quality result.

A real held-out pilot can be claimed only after:

- selected valid episodes are available locally,
- the manifest builder confirms complete held-out episode splits,
- training finishes with recorded metadata and progress logs,
- evaluation runs on held-out test episodes,
- predictions, metrics, confusion matrices, and a run report are committed.

The public data access summary is:

`results/omni_finetune/DATA_ACCESS_STATUS.md`

The current metadata-only full dataset audit is:

`results/omni_finetune/FULL_DATASET_METADATA_AUDIT.md`

The current 128-episode metadata-balanced download plan is:

`results/omni_finetune/XPERIENCE10M_128_EPISODE_SELECTION.md`

The older machine-generated source discovery report remains a pre-access local
planning record:

`results/omni_finetune/DATA_BLOCKER_REPORT.md`

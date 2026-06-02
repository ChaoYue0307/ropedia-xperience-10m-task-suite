# Xperience-10M Multi-Episode Data Access Status

This file summarizes what is needed before the Qwen3-Omni pilot becomes a real
held-out multi-episode experiment.

## Current State

| Item | Status |
| --- | --- |
| Target pilot size | 32 valid Xperience-10M episodes |
| Current public local sample | 1 episode |
| Full dataset access | Pending gated-dataset approval |
| Current Qwen3-Omni artifacts | Setup-stage sample run, not held-out multi-episode model metrics |
| Public raw-data redistribution | Not included |

## Episode Requirement

A valid training episode needs `annotation.hdf5` and at least
`fisheye_cam0.mp4`. A complete omni-model episode preferably includes all six
MP4 streams. `visualization.rrd` is a viewer artifact and is excluded from
training downloads.

The 32-episode pilot should only be reported after:

- at least 32 valid episodes are staged,
- train/test splits are separated by episode,
- manifest files record missing views and feature coverage,
- training finishes with metadata and progress logs,
- evaluation runs on held-out test episodes,
- predictions, metrics, confusion matrices, and a run report are committed.

## Discovery Snapshot

| Source | Valid episodes available to the current public project state |
| --- | ---: |
| Local public sample | 1 |
| ModelScope discovery | 0 |
| Hugging Face discovery | 0 |

These counts describe the current staged/project-visible data, not the full
scale of Xperience-10M.

## Related Files

- `results/omni_finetune/MULTI_EPISODE_ACCESS_STATUS.md`
- `results/omni_finetune/source_discovery.json`
- `scripts/omni/discover_xperience10m_sources.py`
- `scripts/omni/build_episode_manifest.py`

# Xperience-10M Multi-Episode Data Access Status

This file summarizes what is needed before the Qwen3-Omni pilot becomes a real
held-out multi-episode experiment.

## Current State

| Item | Status |
| --- | --- |
| Minimum pilot gate | 32 valid Xperience-10M episodes |
| Current public local sample | 1 episode |
| Full dataset access | Granted; metadata-only Hugging Face audit completed |
| Current full-dataset metadata snapshot | 12,102 complete visible HF episodes across 802 complete sessions |
| Current selected pilot | 128 metadata-balanced episodes, 96/16/16 train/val/test |
| Current multi-episode data state | Selected episodes are being prepared; preprocessing and held-out evaluation are not complete yet |
| Current Qwen3-Omni artifacts | Setup-stage sample run, not held-out multi-episode model metrics |
| Public raw-data redistribution | Not included |

The selected 128-episode pilot is the next model-quality milestone. It should
not be described as a completed fine-tune or evaluated model until the selected
episodes are available, checked for modality coverage, preprocessed into
train/validation/test examples, trained, and evaluated on held-out sessions.

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
| Hugging Face gated metadata audit | 12,102 complete visible episodes |

The Hugging Face count is a metadata-only availability result. It does not mean
that the raw files have been downloaded, staged, or used for multi-episode
training yet.

## Related Files

- `results/omni_finetune/MULTI_EPISODE_ACCESS_STATUS.md`
- `results/omni_finetune/FULL_DATASET_METADATA_AUDIT.md`
- `results/omni_finetune/full_dataset_metadata_audit.json`
- `results/omni_finetune/XPERIENCE10M_128_EPISODE_SELECTION.md`
- `results/omni_finetune/XPERIENCE10M_128_DATA_PREPARATION_AND_FINETUNE_PLAN.md`
- `results/omni_finetune/xperience10m_128_episode_selection.json`
- `results/omni_finetune/xperience10m_128_episode_download_files.txt`
- `results/omni_finetune/source_discovery.json`
- `scripts/omni/discover_xperience10m_sources.py`
- `scripts/omni/analyze_xperience10m_hf_metadata.py`
- `scripts/omni/select_xperience10m_pilot_episodes.py`
- `scripts/omni/relay_xperience10m_selection.py`
- `scripts/omni/parallel_chunk_transfer.py`
- `scripts/omni/audit_staged_xperience10m_content.py`
- `scripts/omni/build_episode_manifest.py`

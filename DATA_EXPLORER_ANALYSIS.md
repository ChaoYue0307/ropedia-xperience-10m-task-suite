# Ropedia Xperience-10M Data Explorer Analysis

Generated: 2026-06-23T09:35:08Z

This report summarizes three data scopes without mixing them: the official public sample episode, the selected 128-episode public-safe feature surface, and authenticated metadata for the full gated Hugging Face dataset.

## Scope Summary

| Scope | Episodes | Rows / windows | Storage view | Notes |
|---|---:|---:|---:|---|
| Public sample | 1 | 1,161 | 4.76 GiB | Raw sample files are playable or source-linked. |
| Selected 128 | 128 | 34,269 | 277.71 GiB | Public-safe matrices and window manifests, not raw redistribution. |
| Full HF dataset | 12,103 episode-like folders | 3,098,112 projected rows at 256/episode | 24.63 TiB | Gated upstream file metadata only. |

## Public Sample

- 5,821 frames at about 20.00 fps.
- 1,161 aligned 20-frame windows with 5-frame stride.
- 8,546 model-input dimensions across 7 modality groups.
- 35 action segments and 34 object labels in the derived explorer.

## Selected 128 Episodes

- Split: train 96, val 16, test 16 episodes.
- Size bands: short 32, lower_mid 32, upper_mid 32, long 32.
- Qwen3-Omni v6 multiscale export: 34,269 rows.
- Dense multiscale compact export: 106,095 rows.

## Full Gated Dataset Metadata

- Repo: `ropedia-ai/xperience-10m` at `ce943cf271a758b60240084892d05cf6dc12dd90`.
- 85,257 files excluding `.gitattributes`.
- 12,102 complete episode folders (99.9917%).
- 72,612 MP4 files and 12,103 `annotation.hdf5` files.

## Generated Charts

- Scope ladder: `assets/charts/data_explorer_scope_ladder.svg`
- Public sample feature dimensions: `assets/charts/data_explorer_sample_feature_modalities.svg`
- Public sample action distribution: `assets/charts/data_explorer_sample_action_distribution.svg`
- Selected-128 split rows: `assets/charts/data_explorer_selected128_split_rows.svg`
- Full dataset file composition: `assets/charts/data_explorer_full_file_composition.svg`

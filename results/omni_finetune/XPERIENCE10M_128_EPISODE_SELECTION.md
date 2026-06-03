# Xperience-10M 128-Episode Metadata-Balanced Selection

This is a download plan, not a trained model result. It uses Hugging Face file metadata only and downloads no raw episode data.

## Why This Selection

- Use only complete episodes: `annotation.hdf5` plus six MP4 streams.
- Exclude `visualization.rrd` from the training download plan.
- Avoid tiny annotation outliers that are likely one-segment examples.
- Use one episode per top-level session to reduce leakage and overfitting to one capture session.
- Balance across four annotation-size bands as a proxy for duration/content richness before category labels are available.
- Split by session into train/val/test.

## Selection Summary

| Measure | Value |
| --- | --- |
| Selected episodes | 128 |
| Unique sessions | 128 |
| Split counts | {"test": 16, "train": 96, "val": 16} |
| Size-band counts | {"long": 32, "lower_mid": 32, "short": 32, "upper_mid": 32} |
| Estimated training download, no RRD | 277.71 GiB |
| Estimated annotation bytes | 226.53 GiB |
| Estimated windows at 256/episode | 32768 |
| Session leakage train/val | 0 |
| Session leakage train/test | 0 |
| Session leakage val/test | 0 |

## Filters

| Rule | Value |
| --- | --- |
| Available complete episodes | 12102 |
| Candidates after filters | 11478 |
| Minimum annotation size | 992.76 MiB |
| Minimum training size | 1.22 GiB |
| Rejected counts | {"annotation_too_small": 606, "training_too_small": 18} |

## Split x Size Band

| Split | short | lower_mid | upper_mid | long |
| --- | --- | --- | --- | --- |
| train | 24 | 24 | 24 | 24 |
| val | 4 | 4 | 4 | 4 |
| test | 4 | 4 | 4 | 4 |

## Important Limitation

HF metadata does not expose semantic content categories. This selection is the best first-pass balance before downloading. After the selected annotations are staged, parse `Main Task`, `Sub Task`, `Current Action`, objects, and interaction text; then swap episodes if one content cluster dominates.

## Output Files

- JSON: `results/omni_finetune/xperience10m_128_episode_selection.json`
- CSV: `results/omni_finetune/xperience10m_128_episode_selection.csv`
- Download file list: `results/omni_finetune/xperience10m_128_episode_download_files.txt`

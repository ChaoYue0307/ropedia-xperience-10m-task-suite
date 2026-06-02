# Xperience-10M Fine-Tune Readiness

Target episodes: 32
Ready for 32-episode pilot: False
Selected source: none

## Source counts
- local (degraded-valid): 1 / 1
- modelscope (degraded-valid): 0 / 0
- huggingface (degraded-valid): 0 / 0

## Current data status
- The staged data is below the 32-episode pilot target. Need 32 degraded-valid episodes; local data currently has 1.
- The current local run demonstrates the training stack on one episode only.
- ModelScope probe unavailable or reported no matching episode files.
- Hugging Face probe unavailable or reported no matching episode files.

## Interpretation
- Degraded-valid means: annotation.hdf5 and fisheye_cam0.mp4 both exist.
- Complete means all six MP4 views are present with annotation.
- A 32-episode pilot starts after this script selects a source with 32+ degraded-valid episodes.

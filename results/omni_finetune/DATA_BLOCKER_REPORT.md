# Xperience-10M Fine-Tune Readiness

Target episodes: 32
Ready for 32-episode pilot: False
Selected source: none

## Source counts
- local (degraded-valid): 1 / 1
- modelscope (degraded-valid): 0 / 0
- huggingface (degraded-valid): 0 / 0

## Blockers
- Not enough degraded-valid episodes for a 32-episode pilot. Need 32, local has 1.
- Current H20 path remains one-episode proof-of-stack only.
- ModelScope probe unavailable or reported no matching episode files.
- Hugging Face probe unavailable or reported no matching episode files.

## Interpretation
- Degraded-valid means: annotation.hdf5 and fisheye_cam0.mp4 both exist.
- Complete means all six MP4 views are present with annotation.
- A 32-episode pilot must not be claimed unless this script selects a source with 32+ degraded-valid episodes.

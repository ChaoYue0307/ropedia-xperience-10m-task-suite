# Data Notice

This repository does not redistribute raw Xperience-10M data.

The full official dataset lives at
[`ropedia-ai/xperience-10m`](https://huggingface.co/datasets/ropedia-ai/xperience-10m)
and is manually gated for approved non-commercial use. This repo records the
official dataset-card description in
[`XPERIENCE10M_DATASET_CARD_ALIGNMENT.md`](XPERIENCE10M_DATASET_CARD_ALIGNMENT.md)
and [`docs/data/xperience10m_dataset_card_alignment.json`](docs/data/xperience10m_dataset_card_alignment.json).
That alignment file separates the official card's about-1PB full-scale storage
statement from the live Hugging Face page/API's 31.9 TB currently hosted
file-size display.
The public Hugging Face API may expose episode-path metadata for the gated repo;
this repo treats that as source-discovery metadata only, not as local data
possession or a permission to redistribute raw files.

To reproduce the experiments, download the public sample from Hugging Face:

```bash
hf download ropedia-ai/xperience-10m-sample \
  --repo-type dataset \
  --local-dir data/sample/xperience-10m-sample
```

Expected files:

```text
annotation.hdf5
fisheye_cam0.mp4
fisheye_cam1.mp4
fisheye_cam2.mp4
fisheye_cam3.mp4
stereo_left.mp4
stereo_right.mp4
visualization.rrd   # optional viewer artifact when available; not used for training
```

The sample card lists `cc-by-nc-4.0` and points to HOMIE Toolkit for inspecting
the videos/annotations and Rerun 0.29.0 for `.rrd` visualization. Use of the
full gated dataset remains governed by the official Xperience-10M terms. The
official card also describes the open-source dataset as limited in diversity
and showcase/production quality, so downstream work still needs robust
evaluation and safeguards. The code license in `LICENSE` does not grant rights
to raw Xperience-10M data.

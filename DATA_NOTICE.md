# Data Notice

This repository does not redistribute raw Xperience-10M data.

The full official dataset lives at
[`ropedia-ai/xperience-10m`](https://huggingface.co/datasets/ropedia-ai/xperience-10m)
and is manually gated for approved non-commercial use. This repo records the
official dataset-card description in
[`XPERIENCE10M_DATASET_CARD_ALIGNMENT.md`](XPERIENCE10M_DATASET_CARD_ALIGNMENT.md)
and [`docs/data/xperience10m_dataset_card_alignment.json`](docs/data/xperience10m_dataset_card_alignment.json).

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
```

Use of the dataset is governed by the original Xperience-10M dataset terms.
The code license in `LICENSE` does not grant rights to raw Xperience-10M data.

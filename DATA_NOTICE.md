# Data Notice

This repository does not redistribute raw Xperience-10M data.

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

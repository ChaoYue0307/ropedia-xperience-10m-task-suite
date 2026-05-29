# All-Modality Minimal Model

Script:

```text
scripts/train_all_modalities_model.py
```

This extends the first minimal model by using every major sample modality in a lightweight way.

## Modalities Used

Dynamic sensor/action modalities:

- `hand_mocap/left_joints_3d`
- `hand_mocap/right_joints_3d`
- `full_body_mocap/keypoints`
- `full_body_mocap/contacts`
- `slam/trans_xyz`
- `slam/quat_wxyz` converted by the toolkit into camera rotation matrices
- `imu/accel_xyz`
- `imu/gyro_xyz`
- `depth/depth`
- `depth/confidence`
- `fisheye_cam0.mp4`
- `fisheye_cam1.mp4`
- `fisheye_cam2.mp4`
- `fisheye_cam3.mp4`
- `stereo_left.mp4`
- `stereo_right.mp4`

Static/context modalities:

- `slam/point_cloud`
- `calibration/*`
- caption objects
- caption interaction text

By default, the script does **not** include `action_label`, `Sub Task`, or action-description text as input, because those are too close to the prediction target. You can force that with `--include-label-text`, but that should be treated as a leakage/debug run, not a fair action-recognition experiment.

## Feature Design

The model is still intentionally small:

```text
raw modality -> per-frame or static handcrafted features -> window temporal statistics -> softmax classifier
```

For each 20-frame window:

- Motion signals use mean/std/min/max/delta/velocity statistics.
- Depth uses global depth stats plus a small normalized depth grid and confidence grid.
- Each video stream uses color stats, color histograms, a small grayscale grid, and simple edge stats.
- Text uses a hashed bag-of-words vector from objects and interaction text.
- Point cloud and calibration are included as static episode-level features.

Current feature blocks:

```text
hand_left_joints:                  441
hand_right_joints:                 441
body_joints:                      1092
body_contacts:                     147
camera_translation:                 21
camera_rotation_matrix:             63
imu_accel_gyro:                     42
depth_confidence:                  980
video_fisheye_cam0:                686
video_fisheye_cam1:                686
video_fisheye_cam2:                686
video_fisheye_cam3:                686
video_stereo_left:                 686
video_stereo_right:                686
caption_objects_interaction_text:  896
slam_point_cloud:                   22
calibration:                       117
total:                            8378
```

## Run Commands

Action prediction:

```bash
cd /path/to/Ropedia
source .venv/bin/activate
python scripts/train_all_modalities_model.py
```

Subtask prediction:

```bash
python scripts/train_all_modalities_model.py --target subtask
```

The first run builds reusable caches in:

```text
outputs/feature_cache/
```

## Current Results

Action-label model:

```text
outputs/min_all_modalities_action_model/
accuracy:          0.9828
balanced_accuracy: 0.9801
macro_f1:          0.9791
weighted_f1:       0.9828
majority_baseline: 0.1375
classes:           18
feature_dim:       8378
test_windows:      291
```

Subtask-label model:

```text
outputs/min_all_modalities_subtask_model/
accuracy:          0.9828
balanced_accuracy: 0.9505
macro_f1:          0.9308
weighted_f1:       0.9838
majority_baseline: 0.1448
classes:           14
feature_dim:       8378
test_windows:      290
```

## How To Interpret This

This proves that the full sample can be converted into a complete supervised learning pipeline on this Mac.

It does **not** prove real generalization, because the public sample is one episode and the split is random windows from that same episode. Neighboring windows are correlated.

For a serious embodied-AI experiment:

```text
many episodes
-> cache features per episode
-> split by episode or task instance
-> train on some episodes
-> test on unseen episodes
```

The next useful upgrade is not a bigger classifier. It is a better split and more episodes.

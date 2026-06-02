# Reproduction Record

Run date: 2026-05-30 Asia/Singapore.

Purpose: show that the committed Ropedia Xperience-10M Task Suite artifacts are
real outputs from the scripts and can be reproduced from the public sample.

## Raw Inputs Checked

The run used the local public sample episode:

```text
data/sample/xperience-10m-sample/
  annotation.hdf5
  fisheye_cam0.mp4
  fisheye_cam1.mp4
  fisheye_cam2.mp4
  fisheye_cam3.mp4
  stereo_left.mp4
  stereo_right.mp4
```

`annotation.hdf5` contains 5,821 aligned frames with depth, hand mocap, body
mocap, IMU, SLAM, calibration, and caption metadata. The video feature cache was
rebuilt from all six video files during the run.

## Commands Re-run

All reproduction outputs were written outside the repo:

```bash
REPRO=/path/to/ignored-scratch-workspace
WORKSPACE=/path/to/Ropedia
ANN=$WORKSPACE/data/sample/xperience-10m-sample/annotation.hdf5
PY=$WORKSPACE/.venv/bin/python

$PY -B scripts/train_min_action_model.py \
  --workspace $WORKSPACE \
  --annotation $ANN \
  --output-dir $REPRO/min_action_model \
  --target action

$PY -B scripts/train_min_action_model.py \
  --workspace $WORKSPACE \
  --annotation $ANN \
  --output-dir $REPRO/min_subtask_model \
  --target subtask

$PY -B scripts/train_all_modalities_model.py \
  --workspace $WORKSPACE \
  --annotation $ANN \
  --output-dir $REPRO/min_all_modalities_action_model \
  --cache-dir $REPRO/cache \
  --target action

$PY -B scripts/train_all_modalities_model.py \
  --workspace $WORKSPACE \
  --annotation $ANN \
  --output-dir $REPRO/min_all_modalities_subtask_model \
  --cache-dir $REPRO/cache \
  --target subtask

$PY -B scripts/episode_task_suite.py \
  --workspace $WORKSPACE \
  --annotation $ANN \
  --output-dir $REPRO/episode_task_suite \
  --cache-dir $REPRO/cache
```

## Exact Match Checks

The regenerated files matched the committed files:

```text
min_action_model/metrics.json: MATCH
min_subtask_model/metrics.json: MATCH
min_all_modalities_action_model/metrics.json: MATCH
min_all_modalities_subtask_model/metrics.json: MATCH
episode_task_suite/summary_report.json: MATCH
episode_task_suite/feature_manifest.json: MATCH
episode_task_suite/available_modalities.json: MATCH
```

Every per-task `metrics.json` also matched:

```text
caption_grounding/metrics.json: MATCH
contact_prediction/metrics.json: MATCH
cross_modal_retrieval/metrics.json: MATCH
hand_trajectory_forecast/metrics.json: MATCH
misalignment_detection/metrics.json: MATCH
modality_reconstruction/metrics.json: MATCH
next_action/metrics.json: MATCH
object_relevance/metrics.json: MATCH
temporal_order/metrics.json: MATCH
timeline_action/metrics.json: MATCH
timeline_subtask/metrics.json: MATCH
transition_detection/metrics.json: MATCH
```

## Fresh Cache Evidence

The all-modality run rebuilt a fresh feature cache:

```text
depth_n5821_grid8.npz: shape=(5821, 140), nonzero=809107
video_fisheye_cam0_n5821_img32_grid8_hist8.npz: shape=(5821, 98), nonzero=570458
video_fisheye_cam1_n5821_img32_grid8_hist8.npz: shape=(5821, 98), nonzero=570400
video_fisheye_cam2_n5821_img32_grid8_hist8.npz: shape=(5821, 98), nonzero=570458
video_fisheye_cam3_n5821_img32_grid8_hist8.npz: shape=(5821, 98), nonzero=568723
video_stereo_left_n5821_img32_grid8_hist8.npz: shape=(5821, 98), nonzero=570249
video_stereo_right_n5821_img32_grid8_hist8.npz: shape=(5821, 98), nonzero=570430
```

This confirms the committed metrics are reproducible from the raw sample and
that the all-modality pipeline reads real depth/video files instead of using
empty placeholder features.

## Caveats

The scripts contain a zero-feature fallback if a video file is missing. That is
not the path used in this run: all six videos existed and produced nonzero
features. The repo remains a single-episode learning and pipeline-validation
project, not evidence of cross-episode generalization.

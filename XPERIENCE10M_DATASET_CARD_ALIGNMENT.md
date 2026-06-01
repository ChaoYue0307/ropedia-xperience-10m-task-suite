# Xperience-10M Official Dataset Card Alignment

This file records the public description of the official
[`ropedia-ai/xperience-10m`](https://huggingface.co/datasets/ropedia-ai/xperience-10m)
dataset card and how this repo uses only one public sample episode from that
larger source. It is a description-alignment artifact, not a raw-data mirror.

Checked on: 2026-06-01.

## Official Dataset Scope

The official Xperience-10M dataset is described by Ropedia as a large-scale
egocentric multimodal dataset for embodied AI, robotics, world models, and
spatial intelligence. The dataset card frames it as human-experience data with
roughly 10 million interaction/experience units and about 10,000 hours of
synchronized first-person recording.

The official card metadata lists these task and modality categories:

- task categories: video classification, image-to-text, depth estimation, robotics
- modalities: 3D, audio, video
- language: English
- license field: `other`
- size category: `1M<n<10M`
- access: manually gated, reviewed access for approved non-commercial use

The current public Hugging Face API metadata reports the dataset repo as
`gated: manual` and notes that an external DocuSign agreement may be required
before approval.

## Official Modalities

The official dataset card describes the full dataset as synchronized 4D
multimodal egocentric data spanning:

- six RGB video streams: four fisheye views and two rectified stereo views
- audio embedded in the video streams
- stereo depth and depth confidence
- camera pose, SLAM trajectory, and point-cloud information
- two-hand motion capture, including hand joints and MANO-related data
- full-body motion capture, keypoints, contacts, and body orientation data
- inertial sensing from accelerometer and gyroscope streams
- hierarchical language/caption annotations
- metadata and calibration records

## Official Scale Statistics

The official dataset card describes Xperience-10M at full scale with these
headline counts:

| Quantity | Official-card scale |
| --- | --- |
| Human experience / interaction units | about 10 million |
| Recording duration | about 10,000 hours |
| RGB frames | about 2.88 billion |
| Depth frames | about 720 million |
| Camera-pose records | about 576 million |
| Motion-capture frames | about 576 million |
| IMU records | about 7.2 billion |
| Caption sentences | about 16 million |
| Caption words | about 200 million |
| Vocabulary size | about 6,000 words |
| Object annotations | about 350,000 objects |
| Trajectory distance | about 39,000 km |
| Total storage described by the card | about 1 PB |

The public Hugging Face page may show a smaller currently listed file-size
summary for the gated repo. This project keeps those concepts separate: the
official card scale describes the dataset design, while this repo validates
only the files that are actually available to the project.

## Episode File Layout

The official gated file listing and the public sample use episode folders with
this practical layout:

```text
<session_uuid>/
  ep<episode_id>/
    fisheye_cam0.mp4
    fisheye_cam1.mp4
    fisheye_cam2.mp4
    fisheye_cam3.mp4
    stereo_left.mp4
    stereo_right.mp4
    annotation.hdf5
    visualization.rrd        # optional viewer artifact; excluded from training downloads
```

For this repo, a valid training/evaluation episode requires `annotation.hdf5`.
Full-omni mode prefers all six MP4 streams. Degraded mode may use
`fisheye_cam0.mp4` plus the annotation file, but must record missing views in
the manifest.

## Annotation File Content

The official card describes the HDF5 annotation file as carrying aligned
multimodal records. The relevant groups include:

- calibration: camera intrinsics/extrinsics for fisheye and stereo cameras
- SLAM/camera pose: quaternions, translations, frame names, and point cloud
- depth: depth map, confidence, scale, min/max, and validity metadata
- hand motion capture: left/right hand joints, translations, and MANO-related records
- full-body motion capture: body keypoints, contacts, transforms, and body rotations
- IMU: timestamps, accelerometer, gyroscope, and keyframe metadata
- video timing: timestamps, frame numbers, and video duration
- language/caption annotations and metadata

This repo's current 8,378-d feature vector uses video-derived statistics,
depth, pose/SLAM, calibration, mocap, IMU, and language-derived blocks. Audio
is documented and visualized, but it is not yet extracted into the current
baseline feature vector.

## Intended Research Uses

The official dataset card supports research directions such as:

- egocentric video/action understanding
- task and subtask recognition
- temporal action localization and human-object interaction analysis
- object grounding and caption/language grounding
- audio-visual learning and multimodal pretraining
- embodied reasoning, world-model learning, and robotics imitation learning
- depth estimation, visual odometry, camera trajectory, SLAM, and scene reconstruction
- hand/body pose, human motion understanding, and sensor fusion

This repo currently implements a single-episode audit suite that starts several
of those directions, but it does not solve the full official task list. The 12
current tasks cover action/subtask labels, next-action prediction, transition
and temporal diagnostics, hand trajectory forecasting, contact prediction,
object relevance, caption grounding, cross-modal retrieval, modality
reconstruction, and misalignment detection. Missing or only-proxy coverage
includes real audio-visual modeling, full caption generation, depth-pixel
estimation, full SLAM estimation, neural rendering, policy learning, and
cross-episode generalization.

## Responsible-Use Boundary

The official dataset is gated and intended for approved non-commercial research
use. This repo therefore does not redistribute raw MP4 files, raw
`annotation.hdf5`, private gated data, raw `visualization.rrd`, or any full
Qwen weights. Public assets here are derived metrics, small thumbnails,
manifests, scripts, charts, and lightweight baseline artifacts.

The official card also makes clear that the data is not meant for identity
recognition, re-identification, biometric profiling, surveillance, sensitive
attribute inference, or safety-critical deployment without appropriate
safeguards.

## Limitations To Preserve In This Project

When describing Xperience-10M in this repo, keep these limitations visible:

- one public sample episode cannot prove cross-environment generalization
- full-dataset claims require gated access, many episodes, and held-out episode splits
- motion capture, SLAM, depth, captions, and other annotations can contain noise
- language annotations are not exhaustive descriptions of every scene state
- large-scale training requires substantial storage, preprocessing, and compute
- the current feature vector does not include an extracted audio feature block

## Current Project Alignment

| Official dataset card concept | Current repo status |
| --- | --- |
| Full Xperience-10M is large, gated, and multi-episode | Acknowledged; not redistributed |
| Public sample includes video/audio/depth/pose/mocap/IMU/language | Represented in the modality atlas |
| Episode layout uses six MP4 streams and `annotation.hdf5` | Used by sample inspection and pilot-readiness scripts |
| Audio exists in MP4 streams | Documented and visualized, not featurized |
| 4D reconstruction/world modeling are intended research directions | Represented by proxy/diagnostic tasks only |
| Real model quality requires held-out multi-episode evaluation | Not claimed yet; 32-episode pilot remains gated |

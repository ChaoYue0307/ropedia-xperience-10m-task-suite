# Xperience-10M Annotation Record Probe

Minimal-cost probe. Downloaded only `annotation.hdf5`; no MP4 or `visualization.rrd` files were downloaded.

- Repo: `ropedia-ai/xperience-10m`
- Probe count: 3
- Raw annotation cache: outside the published repo
- Local files only: `False`

## 9cecac72-8874-4b97-9541-18d4858f8e43/ep10/annotation.hdf5

- Downloaded annotation size: 6.38 MiB (6,687,192 bytes)
- HDF5 top-level keys: `calibration, caption, depth, full_body_mocap, hand_mocap, imu, metadata, slam, video`
- HDF5 dataset count: 65
- Largest first-dimension dataset: `imu/accel_xyz` with first dimension `190`

### Caption JSON Summary

| Measure | Value |
| --- | --- |
| Parse status | ok |
| JSON bytes | 1,178 |
| Segment count | 1 |
| Current-action count | 1 |
| Object-frame count | 1 |
| Interaction-frame count | 1 |
| Sampled-frame count | 1 |
| Unique subtasks | 1 |
| Unique action labels | 1 |
| Unique objects | 3 |
| Action labels | ["Arrange items in bin"] |
| Objects | ["cardboard box", "hand", "plastic storage bin"] |

### Top Groups

| Group | Dataset count | Max first dimension | First-dim histogram top values |
| --- | --- | --- | --- |
| calibration | 23 | 4 | {"4": 14} |
| caption | 1 | 0 | {} |
| depth | 5 | 20 | {"20": 2} |
| full_body_mocap | 9 | 20 | {"20": 9} |
| hand_mocap | 10 | 20 | {"20": 10} |
| imu | 4 | 190 | {"190": 3, "20": 1} |
| metadata | 6 | 0 | {} |
| slam | 4 | 47 | {"20": 3, "47": 1} |
| video | 3 | 20 | {"20": 2} |

### Caption / Action / Interaction Related Datasets

| Dataset | Shape | Dtype | First dim | Sample values |
| --- | --- | --- | --- | --- |
| caption | [] | object | None | ["{\"config\": {\"segment_sec\": 20, \"sample_fps\": 0.5, \"total_tokens\": 2047, \"Main Task\": \"Packing items into a plastic bin. The person is placing va... |

## cdc1ae12-a460-48ac-a892-7d314095c4b1/ep23/annotation.hdf5

- Downloaded annotation size: 6.38 MiB (6,687,256 bytes)
- HDF5 top-level keys: `calibration, caption, depth, full_body_mocap, hand_mocap, imu, metadata, slam, video`
- HDF5 dataset count: 65
- Largest first-dimension dataset: `imu/accel_xyz` with first dimension `188`

### Caption JSON Summary

| Measure | Value |
| --- | --- |
| Parse status | ok |
| JSON bytes | 1,051 |
| Segment count | 1 |
| Current-action count | 1 |
| Object-frame count | 1 |
| Interaction-frame count | 1 |
| Sampled-frame count | 1 |
| Unique subtasks | 1 |
| Unique action labels | 1 |
| Unique objects | 4 |
| Action labels | ["Pulling up sock"] |
| Objects | ["bathroom floor", "feet", "sock", "toilet"] |

### Top Groups

| Group | Dataset count | Max first dimension | First-dim histogram top values |
| --- | --- | --- | --- |
| calibration | 23 | 4 | {"4": 14} |
| caption | 1 | 0 | {} |
| depth | 5 | 20 | {"20": 2} |
| full_body_mocap | 9 | 20 | {"20": 9} |
| hand_mocap | 10 | 20 | {"20": 10} |
| imu | 4 | 188 | {"188": 3, "20": 1} |
| metadata | 6 | 0 | {} |
| slam | 4 | 128 | {"20": 3, "128": 1} |
| video | 3 | 20 | {"20": 2} |

### Caption / Action / Interaction Related Datasets

| Dataset | Shape | Dtype | First dim | Sample values |
| --- | --- | --- | --- | --- |
| caption | [] | object | None | ["{\"config\": {\"segment_sec\": 20, \"sample_fps\": 0.5, \"total_tokens\": 2035, \"Main Task\": \"Putting on socks. The person is standing in a bathroom and... |

## 10282b64-a955-461e-9ef9-a1ddf8dc619a/ep5/annotation.hdf5

- Downloaded annotation size: 6.40 MiB (6,706,448 bytes)
- HDF5 top-level keys: `calibration, caption, depth, full_body_mocap, hand_mocap, imu, metadata, slam, video`
- HDF5 dataset count: 65
- Largest first-dimension dataset: `slam/point_cloud` with first dimension `837`

### Caption JSON Summary

| Measure | Value |
| --- | --- |
| Parse status | ok |
| JSON bytes | 1,299 |
| Segment count | 1 |
| Current-action count | 1 |
| Object-frame count | 1 |
| Interaction-frame count | 1 |
| Sampled-frame count | 1 |
| Unique subtasks | 1 |
| Unique action labels | 1 |
| Unique objects | 4 |
| Action labels | ["Walk down retail aisle"] |
| Objects | ["person seated", "product packaging", "retail shelf", "shopping bags"] |

### Top Groups

| Group | Dataset count | Max first dimension | First-dim histogram top values |
| --- | --- | --- | --- |
| calibration | 23 | 4 | {"4": 14} |
| caption | 1 | 0 | {} |
| depth | 5 | 20 | {"20": 2} |
| full_body_mocap | 9 | 20 | {"20": 9} |
| hand_mocap | 10 | 20 | {"20": 10} |
| imu | 4 | 190 | {"190": 3, "20": 1} |
| metadata | 6 | 0 | {} |
| slam | 4 | 837 | {"20": 3, "837": 1} |
| video | 3 | 20 | {"20": 2} |

### Caption / Action / Interaction Related Datasets

| Dataset | Shape | Dtype | First dim | Sample values |
| --- | --- | --- | --- | --- |
| caption | [] | object | None | ["{\"config\": {\"segment_sec\": 20, \"sample_fps\": 0.5, \"total_tokens\": 2060, \"Main Task\": \"walking through a retail store. The video shows a first-pe... |

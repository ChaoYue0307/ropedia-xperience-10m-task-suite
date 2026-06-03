# Xperience-10M HF Metadata Audit

Metadata-only analysis of the gated Hugging Face dataset. No MP4, HDF5, RRD, or model files were downloaded.

## Access and Source

- Repo: `ropedia-ai/xperience-10m`
- Repo SHA: `ce943cf271a758b60240084892d05cf6dc12dd90`
- Last modified: `2026-04-21T05:03:45+00:00`
- Gated mode: `manual`
- Pretty name: `Xperience-10M`
- License field: `other`
- HF size category: `1M<n<10M`
- Tags: `egocentric, first-person, multimodal, 3d, 4d, embodied-ai, robotics, human-motion, mocap, imu, audio, depth, captions, video`

## Current Hub File Metadata

| Measure | Value |
| --- | --- |
| Files listed by API | 85,257 |
| Total bytes from file metadata | 25.52 TiB (28,057,584,187,079 bytes) |
| Bytes excluding visualization.rrd | 24.63 TiB (27,083,292,060,675 bytes) |
| visualization.rrd bytes | 907.38 GiB (974,292,126,404 bytes) |
| Top-level session folders | 804 |
| Episode-like folders | 12,103 |

## File Composition

| File type | Count |
| --- | --- |
| .hdf5 | 12,103 |
| .md | 1 |
| .mp4 | 72,612 |
| .rrd | 541 |

## Episode Completeness

| Measure | Value |
| --- | --- |
| annotation.hdf5 files | 12,103 |
| MP4 files | 72,612 |
| visualization.rrd files | 541 |
| Complete episodes: annotation + all six MP4 views | 12,102 (99.9917%) |
| Degraded-valid episodes: annotation + fisheye_cam0 | 12,102 (99.9917%) |
| Sessions with complete episodes | 802 |
| Video-count histogram per episode | {"0": 1, "6": 12102} |

## Episode Size Distribution

| Statistic | Training bytes per complete episode, excluding visualization.rrd |
| --- | --- |
| Min | 7.78 MiB |
| P25 | 2.13 GiB |
| Median | 2.20 GiB |
| P75 | 2.25 GiB |
| Mean | 2.08 GiB |
| Max | 2.53 GiB |

## Annotation File Size Distribution

| Statistic | annotation.hdf5 size |
| --- | --- |
| Min | 6.38 MiB |
| P25 | 1.74 GiB |
| Median | 1.83 GiB |
| P75 | 1.85 GiB |
| Mean | 1.70 GiB |
| Max | 1.86 GiB |

## Pilot Scale Estimates

| Pilot | Episodes | Max windows at 256/episode | Storage estimate |
| --- | --- | --- | --- |
| 32-episode smallest one-per-session | 32 | 8192 | 35.35 GiB |
| 32-episode median-sized estimate | 32 | 8192 | 70.51 GiB |
| 32-episode mean-sized estimate | 32 | 8192 | 66.69 GiB |
| 100-episode pilot | 100 | 25600 | roughly 220.34 GiB at median episode size |
| 500-episode pilot | 500 | 128000 | roughly 1.08 TiB at median episode size |
| All complete visible HF episodes | 12102 | 3098112 | 24.63 TiB |

## Incomplete Episode Records

[
  {
    "episode_path": "dc3f4139-f499-4de7-b057-e25b7dfb2d83/ep1",
    "episode_id": "ep1",
    "top_level_session": "dc3f4139-f499-4de7-b057-e25b7dfb2d83",
    "file_count": 1,
    "total_bytes": 1418232696,
    "training_bytes_excluding_visualization_rrd": 1418232696,
    "has_annotation": true,
    "has_fisheye_cam0": false,
    "video_count": 0,
    "has_all_six_videos": false,
    "is_degraded_valid": false,
    "is_complete": false,
    "has_visualization_rrd": false,
    "missing_required_files": [
      "fisheye_cam0.mp4",
      "fisheye_cam1.mp4",
      "fisheye_cam2.mp4",
      "fisheye_cam3.mp4",
      "stereo_left.mp4",
      "stereo_right.mp4"
    ]
  }
]

## Download and Compute Recommendation

- This metadata audit can run on any machine with Hugging Face access.
- If the training host cannot reach Hugging Face, download on an HF-reachable relay host, then transfer staged episode folders to the training host.
- For training downloads, include `annotation.hdf5` plus the six MP4 streams; exclude `visualization.rrd` unless Rerun visualization is specifically needed.
- For the first real training pilot, prefer 32 complete episodes from different top-level sessions and avoid selecting only the tiny outlier episodes.
- The training host is used after staged data exists: manifest validation, preprocessing, LoRA training, and held-out evaluation.

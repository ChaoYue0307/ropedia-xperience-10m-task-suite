#!/usr/bin/env python3
"""Build a lightweight manifest for local Ropedia/Xperience episode folders.

The manifest is intentionally metadata-only. It lets us decide how many
episodes fit on target storage before downloading or copying large media.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import h5py


VIDEO_NAMES = [
    "fisheye_cam0.mp4",
    "fisheye_cam1.mp4",
    "fisheye_cam2.mp4",
    "fisheye_cam3.mp4",
    "stereo_left.mp4",
    "stereo_right.mp4",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan Ropedia/Xperience episode folders.")
    workspace_default = Path(__file__).resolve().parents[2]
    parser.add_argument("--workspace", type=Path, default=workspace_default)
    parser.add_argument(
        "--data-root",
        type=Path,
        action="append",
        required=True,
        help="Root to scan. May be passed multiple times.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/omni_exploration/episode_manifest.json"),
    )
    parser.add_argument("--max-episodes", type=int, default=0, help="0 means no cap.")
    parser.add_argument("--window-frames", type=int, default=20)
    parser.add_argument("--stride-frames", type=int, default=20)
    parser.add_argument("--min-label-fraction", type=float, default=0.6)
    parser.add_argument("--train-fraction", type=float, default=0.80)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--split-seed", type=int, default=7)
    return parser.parse_args()


def add_toolkit_to_path(workspace: Path) -> None:
    toolkit = workspace / "HOMIE-toolkit"
    if not toolkit.exists():
        raise FileNotFoundError(f"HOMIE-toolkit not found: {toolkit}")
    if str(toolkit) not in sys.path:
        sys.path.insert(0, str(toolkit))


def size_or_zero(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def decode_frame_name(value) -> str:
    raw = value
    if hasattr(raw, "tobytes"):
        raw = raw.tobytes()
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace").strip("\x00")
    return str(raw)


def infer_frame_names(annotation: Path) -> list[str]:
    with h5py.File(annotation, "r") as f:
        if "slam/frame_names" in f:
            ds = f["slam/frame_names"]
            return [decode_frame_name(ds[i]) for i in range(ds.shape[0])]
        for key in ("hand_mocap/left_joints_3d", "depth/depth", "full_body_mocap/keypoints"):
            if key in f:
                return [f"frame_{idx:06d}.jpg" for idx in range(f[key].shape[0])]
    return []


def hdf5_presence(annotation: Path) -> dict:
    checks = {
        "calibration": "calibration",
        "slam_pose": "slam/quat_wxyz",
        "slam_point_cloud": "slam/point_cloud",
        "depth": "depth/depth",
        "depth_confidence": "depth/confidence",
        "hand_mocap": "hand_mocap/left_joints_3d",
        "body_mocap": "full_body_mocap/keypoints",
        "contacts": "full_body_mocap/contacts",
        "imu": "imu/accel_xyz",
        "caption": "caption",
        "captions": "captions",
    }
    with h5py.File(annotation, "r") as f:
        return {name: key in f for name, key in checks.items()}


def frame_label(info: dict, target: str) -> str:
    key = "theme" if target == "subtask" else "action_label"
    label = str(info.get(key, "")).strip()
    if not label or label.upper() == "N/A":
        return ""
    return label


def majority_label(labels: list[str], min_fraction: float) -> tuple[str, float]:
    labels = [label for label in labels if label]
    if not labels:
        return "", 0.0
    label, count = Counter(labels).most_common(1)[0]
    fraction = count / len(labels)
    if fraction < min_fraction:
        return "", fraction
    return label, fraction


def label_metadata(annotation: Path, frame_names: list[str], args: argparse.Namespace) -> dict:
    from utils.caption_utils import load_caption_data_from_annotation_hdf5

    main_task, frame_info, segment_boundaries, _task_to_id = load_caption_data_from_annotation_hdf5(
        annotation,
        str(annotation.parent),
        frame_names,
    )
    if frame_info is None:
        return {
            "main_task": "",
            "segments": 0,
            "frame_labels": {"action": {}, "subtask": {}},
            "window_labels": {"action": {}, "subtask": {}},
            "num_labeled_windows": {"action": 0, "subtask": 0},
        }

    frame_counts = {"action": Counter(), "subtask": Counter()}
    for idx in range(len(frame_names)):
        info = frame_info.get(idx, {})
        for target in frame_counts:
            label = frame_label(info, target)
            if label:
                frame_counts[target][label] += 1

    window_counts = {"action": Counter(), "subtask": Counter()}
    for target in window_counts:
        for start in range(0, len(frame_names) - args.window_frames + 1, args.stride_frames):
            end = start + args.window_frames
            labels = [frame_label(frame_info.get(i, {}), target) for i in range(start, end)]
            label, _frac = majority_label(labels, args.min_label_fraction)
            if label:
                window_counts[target][label] += 1

    return {
        "main_task": main_task,
        "segments": len(segment_boundaries),
        "frame_labels": {target: dict(counts.most_common()) for target, counts in frame_counts.items()},
        "window_labels": {target: dict(counts.most_common()) for target, counts in window_counts.items()},
        "num_labeled_windows": {target: int(sum(counts.values())) for target, counts in window_counts.items()},
    }


def assign_splits(episodes: list[dict], args: argparse.Namespace) -> None:
    if not episodes:
        return
    total = args.train_fraction + args.val_fraction + args.test_fraction
    if total <= 0:
        raise ValueError("Split fractions must sum to a positive value.")
    train_fraction = args.train_fraction / total
    val_fraction = args.val_fraction / total

    order = list(range(len(episodes)))
    rng = random.Random(args.split_seed)
    rng.shuffle(order)
    n = len(order)
    n_train = int(round(n * train_fraction))
    n_val = int(round(n * val_fraction))
    if n >= 3:
        n_train = max(1, min(n_train, n - 2))
        n_val = max(1, min(n_val, n - n_train - 1))
    elif n == 2:
        n_train, n_val = 1, 0
    else:
        n_train, n_val = 1, 0

    split_by_idx = {}
    for pos, idx in enumerate(order):
        if pos < n_train:
            split = "train"
        elif pos < n_train + n_val:
            split = "val"
        else:
            split = "test"
        split_by_idx[idx] = split

    for idx, episode in enumerate(episodes):
        episode["split"] = split_by_idx[idx]


def inspect_episode(annotation: Path, args: argparse.Namespace) -> dict:
    episode_dir = annotation.parent
    files = [{"name": "annotation.hdf5", "bytes": size_or_zero(annotation), "exists": annotation.exists()}]
    for name in VIDEO_NAMES:
        path = episode_dir / name
        files.append({"name": name, "bytes": size_or_zero(path), "exists": path.exists()})
    rrd = episode_dir / "visualization.rrd"
    files.append({"name": "visualization.rrd", "bytes": size_or_zero(rrd), "exists": rrd.exists()})
    total_bytes = sum(item["bytes"] for item in files)
    train_bytes = sum(item["bytes"] for item in files if item["name"] != "visualization.rrd")
    frame_names = infer_frame_names(annotation)
    hdf5_modalities = hdf5_presence(annotation)
    labels = label_metadata(annotation, frame_names, args)
    videos = [
        {
            "name": name,
            "path": str(episode_dir / name),
            "bytes": size_or_zero(episode_dir / name),
            "exists": (episode_dir / name).exists(),
        }
        for name in VIDEO_NAMES
    ]
    return {
        "episode_id": episode_dir.name,
        "path": str(episode_dir),
        "annotation": str(annotation),
        "frame_count": len(frame_names),
        "main_task": labels["main_task"],
        "files": files,
        "videos": videos,
        "hdf5_modalities": hdf5_modalities,
        "label_stats": labels,
        "total_bytes": total_bytes,
        "train_minimal_bytes": train_bytes,
        "has_annotation": annotation.exists(),
        "has_any_video": any((episode_dir / name).exists() for name in VIDEO_NAMES),
        "has_all_videos": all((episode_dir / name).exists() for name in VIDEO_NAMES),
        "has_rrd": rrd.exists(),
    }


def main() -> int:
    args = parse_args()
    args.workspace = args.workspace.expanduser().resolve()
    add_toolkit_to_path(args.workspace)
    annotations: list[Path] = []
    for root in args.data_root:
        annotations.extend(sorted(root.expanduser().resolve().rglob("annotation.hdf5")))
    if args.max_episodes > 0:
        annotations = annotations[: args.max_episodes]

    episodes = [inspect_episode(path, args) for path in annotations]
    assign_splits(episodes, args)
    split_counts = Counter(ep["split"] for ep in episodes)
    summary = {
        "num_episodes": len(episodes),
        "total_bytes": sum(ep["total_bytes"] for ep in episodes),
        "train_minimal_bytes": sum(ep["train_minimal_bytes"] for ep in episodes),
        "split_counts": dict(split_counts),
        "split_fractions": {
            "train": args.train_fraction,
            "val": args.val_fraction,
            "test": args.test_fraction,
            "seed": args.split_seed,
        },
        "windowing": {
            "window_frames": args.window_frames,
            "stride_frames": args.stride_frames,
            "min_label_fraction": args.min_label_fraction,
        },
        "notes": [
            "train_minimal_bytes excludes visualization.rrd because model training does not need it.",
            "This file is metadata-only; it does not copy or download raw data.",
            "Splits are assigned by whole episode to avoid window leakage.",
        ],
    }
    payload = {"summary": summary, "episodes": episodes}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

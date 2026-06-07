#!/usr/bin/env python3
"""Augment exported Xperience windows with Cosmos3 camera-pose action targets.

This does not invent robot-control labels. It converts frame-aligned SLAM poses
from `annotation.hdf5` into the Cosmos3-supported `camera_pose` action domain:
9D per-transition vectors with translation delta, rotation delta as a rotation
vector, and absolute displacement from the window start. The target is a
continuous egocentric-motion proxy suitable for a first Cosmos3 action-packer
smoke run; it is intentionally separate from the semantic JSON QA target.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from qwen3_omni_dataset_utils import load_jsonl, write_jsonl


RAW_ACTION_DIM = 9
DOMAIN_NAME = "camera_pose"


def parse_args() -> argparse.Namespace:
    workspace_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--resolution-tier", type=int, default=480, choices=[256, 480, 704, 720])
    parser.add_argument("--view-point", default="ego_view")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def read_pose_cache(annotation_path: Path) -> dict[str, np.ndarray]:
    with h5py.File(annotation_path, "r") as h5:
        slam = h5["slam"]
        trans = np.asarray(slam["trans_xyz"], dtype=np.float64)
        quat = np.asarray(slam["quat_wxyz"], dtype=np.float64)
        frame_numbers = np.asarray(h5["video"]["frame_number"], dtype=np.int64)
    return {"trans": trans, "quat": normalize_quat_array(quat), "frame_numbers": frame_numbers}


def normalize_quat_array(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    norm[norm <= 1e-12] = 1.0
    quat = quat / norm
    # Keep quaternion sign continuous enough for simple deltas.
    for idx in range(1, len(quat)):
        if np.dot(quat[idx - 1], quat[idx]) < 0:
            quat[idx] *= -1.0
    return quat


def quat_inverse(q: np.ndarray) -> np.ndarray:
    return np.asarray([q[0], -q[1], -q[2], -q[3]], dtype=np.float64) / max(float(np.dot(q, q)), 1e-12)


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.asarray(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    if q[0] < 0:
        q = -q
    w = float(np.clip(q[0], -1.0, 1.0))
    xyz = q[1:]
    sin_half = float(np.linalg.norm(xyz))
    if sin_half < 1e-8:
        return 2.0 * xyz
    angle = 2.0 * math.atan2(sin_half, w)
    if angle > math.pi:
        angle -= 2.0 * math.pi
    return xyz / sin_half * angle


def nearest_index(frame_numbers: np.ndarray, frame: int) -> int:
    if frame <= int(frame_numbers[0]):
        return 0
    if frame >= int(frame_numbers[-1]):
        return len(frame_numbers) - 1
    return int(np.searchsorted(frame_numbers, frame, side="left"))


def sampled_frame_pairs(start_frame: int, end_frame: int, chunk_size: int) -> list[tuple[int, int]]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    if end_frame <= start_frame:
        end_frame = start_frame + chunk_size
    points = np.linspace(start_frame, end_frame, chunk_size + 1)
    frames = [int(round(value)) for value in points]
    pairs: list[tuple[int, int]] = []
    for left, right in zip(frames[:-1], frames[1:]):
        if right <= left:
            right = left + 1
        pairs.append((left, right))
    return pairs


def camera_pose_actions(pose: dict[str, np.ndarray], start_frame: int, end_frame: int, chunk_size: int) -> list[list[float]]:
    trans = pose["trans"]
    quat = pose["quat"]
    frame_numbers = pose["frame_numbers"]
    start_idx = nearest_index(frame_numbers, start_frame)
    origin = trans[start_idx]
    rows: list[list[float]] = []
    for left_frame, right_frame in sampled_frame_pairs(start_frame, end_frame, chunk_size):
        li = nearest_index(frame_numbers, left_frame)
        ri = nearest_index(frame_numbers, right_frame)
        delta_t = trans[ri] - trans[li]
        delta_q = quat_multiply(quat[ri], quat_inverse(quat[li]))
        delta_r = quat_to_rotvec(delta_q)
        displacement = trans[ri] - origin
        row = np.concatenate([delta_t, delta_r, displacement]).astype(np.float32)
        if row.shape[0] != RAW_ACTION_DIM:
            raise AssertionError(row.shape)
        rows.append([float(value) for value in row])
    return rows


def media_condition(row: dict[str, Any]) -> dict[str, Any]:
    media = row.get("media") if isinstance(row.get("media"), dict) else {}
    return {
        "mosaic_video_path": media.get("mosaic_video_path"),
        "video_paths": media.get("video_paths") if isinstance(media.get("video_paths"), list) else [],
        "context_start_frame": media.get("context_start_frame"),
        "context_end_frame": media.get("context_end_frame"),
    }


def augment_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pose_cache: dict[str, dict[str, np.ndarray]] = {}
    counters = Counter()
    issues: list[dict[str, Any]] = []
    augmented: list[dict[str, Any]] = []
    selected = rows[: args.max_records] if args.max_records > 0 else rows

    for idx, row in enumerate(selected):
        counters["rows_seen"] += 1
        episode_path_raw = row.get("episode_path")
        window = row.get("center_window") if isinstance(row.get("center_window"), dict) else {}
        if not episode_path_raw or "start_frame" not in window or "end_frame" not in window:
            counters["rows_skipped_missing_source_fields"] += 1
            issues.append({"row_index": idx, "id": row.get("id"), "reason": "missing episode_path or center_window"})
            if args.strict:
                raise ValueError(issues[-1])
            continue
        annotation_path = Path(str(episode_path_raw)) / "annotation.hdf5"
        if not annotation_path.exists():
            counters["rows_skipped_missing_annotation"] += 1
            issues.append({"row_index": idx, "id": row.get("id"), "reason": f"missing {annotation_path}"})
            if args.strict:
                raise FileNotFoundError(annotation_path)
            continue
        key = str(annotation_path)
        if key not in pose_cache:
            pose_cache[key] = read_pose_cache(annotation_path)
        start_frame = int(window["start_frame"])
        end_frame = int(window["end_frame"])
        try:
            raw_actions = camera_pose_actions(pose_cache[key], start_frame, end_frame, args.chunk_size)
        except Exception as exc:
            counters["rows_skipped_action_build_error"] += 1
            issues.append({"row_index": idx, "id": row.get("id"), "reason": repr(exc)})
            if args.strict:
                raise
            continue

        copied = dict(row)
        copied["cosmos_action_target"] = {
            "mode": "forward_dynamics",
            "domain_name": DOMAIN_NAME,
            "chunk_size": args.chunk_size,
            "raw_action_dim": RAW_ACTION_DIM,
            "raw_actions": raw_actions,
            "resolution_tier": args.resolution_tier,
            "view_point": args.view_point,
            "source": {
                "kind": "slam_camera_pose_delta_proxy_v1",
                "annotation_hdf5": str(annotation_path),
                "frame_range": {"start_frame": start_frame, "end_frame": end_frame},
                "fields": [
                    "slam/trans_xyz delta",
                    "slam/quat_wxyz delta as rotation vector",
                    "slam/trans_xyz displacement from window start",
                ],
                "units": "translation in annotation coordinate units; rotation in radians",
            },
            "conditioning": media_condition(row),
        }
        augmented.append(copied)
        counters["rows_augmented"] += 1

    manifest = {
        "status": "pass" if counters["rows_augmented"] else "fail",
        "input_dataset_jsonl": str(args.dataset_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "domain_name": DOMAIN_NAME,
        "raw_action_dim": RAW_ACTION_DIM,
        "chunk_size": args.chunk_size,
        "resolution_tier": args.resolution_tier,
        "view_point": args.view_point,
        "target_kind": "slam_camera_pose_delta_proxy_v1",
        "counts": dict(counters),
        "episode_annotation_files_read": len(pose_cache),
        "issues": issues[:100],
        "limitations": [
            "This is an egocentric camera-motion proxy, not a robot gripper or human hand-control action.",
            "Use it for Cosmos3 action-packer and one-episode overfit smoke tests before claiming model-quality improvement.",
            "Fit any normalization on train episodes only before a full publishable Cosmos adapter run.",
        ],
    }
    return augmented, manifest


def main() -> int:
    args = parse_args()
    rows = load_jsonl(args.dataset_jsonl)
    augmented, manifest = augment_rows(rows, args)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_jsonl, augmented)
    args.output_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0 if manifest["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

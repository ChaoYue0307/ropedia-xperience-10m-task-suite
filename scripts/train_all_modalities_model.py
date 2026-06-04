#!/usr/bin/env python3
"""
All-modality lightweight baseline for an Xperience-10M episode.

This intentionally stays small enough for a MacBook:
  - no deep video training
  - no CUDA
  - no PyTorch dependency

Each modality is compressed into window-level statistics, then the same
Numpy softmax classifier from train_min_action_model.py is used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
from collections import Counter, OrderedDict
from pathlib import Path

import cv2
import h5py
import numpy as np

from train_min_action_model import (
    add_toolkit_to_path,
    center_by_body_root,
    compute_metrics,
    encode_labels,
    fit_scaler,
    frame_label,
    majority_label,
    predict,
    portable_path,
    safe_window,
    save_artifacts,
    stratified_split,
    temporal_stats,
    train_softmax_classifier,
)


VIDEO_FILES = OrderedDict([
    ("fisheye_cam0", "fisheye_cam0.mp4"),
    ("fisheye_cam1", "fisheye_cam1.mp4"),
    ("fisheye_cam2", "fisheye_cam2.mp4"),
    ("fisheye_cam3", "fisheye_cam3.mp4"),
    ("stereo_left", "stereo_left.mp4"),
    ("stereo_right", "stereo_right.mp4"),
])

PRIMARY_AUDIO_VIDEO = "fisheye_cam0"


def parse_args() -> argparse.Namespace:
    workspace_default = Path(__file__).resolve().parents[1]
    annotation_default = workspace_default / "data/sample/xperience-10m-sample/annotation.hdf5"

    parser = argparse.ArgumentParser(description="Train a lightweight all-modality Ropedia classifier.")
    parser.add_argument("--workspace", type=Path, default=workspace_default, help="Ropedia workspace root.")
    parser.add_argument("--annotation", type=Path, default=annotation_default, help="Path to annotation.hdf5.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output artifact directory.")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Feature cache directory.")
    parser.add_argument("--target", choices=["action", "subtask"], default="action", help="Prediction target.")
    parser.add_argument("--window-frames", type=int, default=20, help="Frames per training window.")
    parser.add_argument("--stride-frames", type=int, default=5, help="Stride between windows.")
    parser.add_argument("--min-label-fraction", type=float, default=0.6, help="Minimum majority-label fraction.")
    parser.add_argument("--test-fraction", type=float, default=0.25, help="Stratified test fraction.")
    parser.add_argument("--epochs", type=int, default=800, help="Training epochs.")
    parser.add_argument("--learning-rate", type=float, default=0.12, help="Softmax learning rate.")
    parser.add_argument("--l2", type=float, default=2e-3, help="L2 weight decay.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--no-class-weights", action="store_true", help="Disable inverse-frequency class weighting.")
    parser.add_argument("--force-rebuild-cache", action="store_true", help="Recompute cached depth/video features.")
    parser.add_argument("--video-image-size", type=int, default=32, help="Resize video frames before visual features.")
    parser.add_argument("--video-grid-size", type=int, default=8, help="Small grayscale grid per video frame.")
    parser.add_argument("--video-hist-bins", type=int, default=8, help="Color histogram bins per channel.")
    parser.add_argument("--depth-grid-size", type=int, default=8, help="Small depth/confidence grid per frame.")
    parser.add_argument("--text-hash-dim", type=int, default=128, help="Hashed bag-of-words dimension.")
    parser.add_argument("--audio-source", choices=list(VIDEO_FILES), default=PRIMARY_AUDIO_VIDEO, help="MP4 stream used for audio features.")
    parser.add_argument("--audio-sample-rate", type=int, default=16000, help="Audio sample rate for extracted audio features.")
    parser.add_argument("--audio-band-count", type=int, default=16, help="Number of log-spaced spectral energy bands per frame.")
    parser.add_argument(
        "--include-label-text",
        action="store_true",
        help="Also include action/subtask/action-description text as input. This leaks target semantics.",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        name = "min_all_modalities_action_model" if args.target == "action" else "min_all_modalities_subtask_model"
        args.output_dir = args.workspace / "outputs" / name
    if args.cache_dir is None:
        args.cache_dir = args.workspace / "outputs/feature_cache"
    return args


def numeric_array(value) -> np.ndarray | None:
    try:
        arr = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if arr.size == 0:
        return None
    return np.nan_to_num(arr.reshape(-1), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def calibration_features(calib_data: dict | None) -> np.ndarray:
    if not calib_data:
        return np.zeros(0, dtype=np.float32)
    chunks: list[np.ndarray] = []
    for cam_id in sorted(calib_data):
        cam = calib_data.get(cam_id, {})
        if not isinstance(cam, dict):
            continue
        for key in sorted(cam):
            arr = numeric_array(cam.get(key))
            if arr is not None:
                chunks.append(arr)
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32)


def point_cloud_features(points: np.ndarray | None) -> np.ndarray:
    if points is None:
        return np.zeros(0, dtype=np.float32)
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0:
        return np.zeros(0, dtype=np.float32)
    pts = np.nan_to_num(pts, nan=0.0, posinf=0.0, neginf=0.0)
    stats = [
        pts.mean(axis=0),
        pts.std(axis=0),
        pts.min(axis=0),
        pts.max(axis=0),
        np.percentile(pts, 10, axis=0),
        np.percentile(pts, 50, axis=0),
        np.percentile(pts, 90, axis=0),
        np.asarray([np.log1p(len(pts))], dtype=np.float32),
    ]
    return np.concatenate(stats).astype(np.float32)


def video_frame_features(frame: np.ndarray, image_size: int, grid_size: int, hist_bins: int) -> np.ndarray:
    small = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = rgb.reshape(-1, 3).mean(axis=0)
    std = rgb.reshape(-1, 3).std(axis=0)

    hists = []
    for channel in range(3):
        hist, _ = np.histogram(rgb[:, :, channel], bins=hist_bins, range=(0.0, 1.0))
        hist = hist.astype(np.float32)
        hist /= max(float(hist.sum()), 1.0)
        hists.append(hist)

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    grid = cv2.resize(gray, (grid_size, grid_size), interpolation=cv2.INTER_AREA).reshape(-1)
    gy, gx = np.gradient(gray)
    edge = np.asarray([np.abs(gx).mean(), np.abs(gy).mean(), np.abs(gx).std(), np.abs(gy).std()], dtype=np.float32)

    return np.concatenate([mean, std, *hists, grid, edge]).astype(np.float32)


def read_video_feature_cache(
    path: Path,
    n_frames: int,
    cache_dir: Path,
    image_size: int,
    grid_size: int,
    hist_bins: int,
    force: bool,
) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"video_{path.stem}_n{n_frames}_img{image_size}_grid{grid_size}_hist{hist_bins}.npz"
    if cache_path.exists() and not force:
        return np.load(cache_path)["features"].astype(np.float32)

    dummy_dim = 6 + 3 * hist_bins + grid_size * grid_size + 4
    features = np.zeros((n_frames, dummy_dim), dtype=np.float32)
    if not path.exists():
        np.savez_compressed(cache_path, features=features)
        return features

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        np.savez_compressed(cache_path, features=features)
        return features

    last = np.zeros(dummy_dim, dtype=np.float32)
    for idx in range(n_frames):
        ok, frame = cap.read()
        if ok:
            last = video_frame_features(frame, image_size, grid_size, hist_bins)
        features[idx] = last
        if idx and idx % 1000 == 0:
            print(f"    {path.name}: {idx}/{n_frames} frames")
    cap.release()
    np.savez_compressed(cache_path, features=features)
    return features


def depth_frame_features(depth: np.ndarray, confidence: np.ndarray | None, depth_min: float, depth_max: float, grid_size: int) -> np.ndarray:
    d = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(d) & (d > 0)
    if valid.any():
        vals = d[valid]
        d_stats = np.asarray([
            vals.mean(),
            vals.std(),
            vals.min(),
            vals.max(),
            np.percentile(vals, 10),
            np.percentile(vals, 50),
            np.percentile(vals, 90),
            valid.mean(),
        ], dtype=np.float32)
    else:
        d_stats = np.zeros(8, dtype=np.float32)

    denom = max(depth_max - depth_min, 1e-6)
    d_norm = np.clip((np.nan_to_num(d, nan=0.0) - depth_min) / denom, 0.0, 1.0)
    d_grid = cv2.resize(d_norm, (grid_size, grid_size), interpolation=cv2.INTER_AREA).reshape(-1).astype(np.float32)

    if confidence is None:
        c_stats = np.zeros(4, dtype=np.float32)
        c_grid = np.zeros(grid_size * grid_size, dtype=np.float32)
    else:
        c = np.asarray(confidence, dtype=np.float32)
        c_scale = 255.0 if c.max(initial=0) > 1.0 else 1.0
        c = np.clip(c / c_scale, 0.0, 1.0)
        c_stats = np.asarray([c.mean(), c.std(), c.min(initial=0), c.max(initial=0)], dtype=np.float32)
        c_grid = cv2.resize(c, (grid_size, grid_size), interpolation=cv2.INTER_AREA).reshape(-1).astype(np.float32)

    return np.concatenate([d_stats, d_grid, c_stats, c_grid]).astype(np.float32)


def read_depth_feature_cache(annotation: Path, n_frames: int, cache_dir: Path, grid_size: int, force: bool) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"depth_n{n_frames}_grid{grid_size}.npz"
    if cache_path.exists() and not force:
        return np.load(cache_path)["features"].astype(np.float32)

    feature_dim = 8 + grid_size * grid_size + 4 + grid_size * grid_size
    features = np.zeros((n_frames, feature_dim), dtype=np.float32)
    with h5py.File(annotation, "r") as f:
        if "depth/depth" not in f:
            np.savez_compressed(cache_path, features=features)
            return features
        depth_ds = f["depth/depth"]
        conf_ds = f["depth/confidence"] if "depth/confidence" in f else None
        depth_min = float(np.asarray(f["depth/depth_min"][()]).flat[0]) if "depth/depth_min" in f else 0.0
        depth_max = float(np.asarray(f["depth/depth_max"][()]).flat[0]) if "depth/depth_max" in f else 4.0
        limit = min(n_frames, depth_ds.shape[0])
        for idx in range(limit):
            confidence = conf_ds[idx] if conf_ds is not None else None
            features[idx] = depth_frame_features(depth_ds[idx], confidence, depth_min, depth_max, grid_size)
            if idx and idx % 1000 == 0:
                print(f"    depth: {idx}/{limit} frames")
    np.savez_compressed(cache_path, features=features)
    return features


def video_fps(path: Path) -> float | None:
    if not path.exists():
        return None
    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) if cap.isOpened() else 0.0
    cap.release()
    return fps if np.isfinite(fps) and fps > 0 else None


def decode_audio_mono(path: Path, sample_rate: int) -> np.ndarray:
    if not path.exists() or shutil.which("ffmpeg") is None:
        return np.zeros(0, dtype=np.float32)
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "-",
    ]
    try:
        proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return np.zeros(0, dtype=np.float32)
    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    if audio.size == 0:
        return np.zeros(0, dtype=np.float32)
    return np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def audio_segment_features(segment: np.ndarray, sample_rate: int, band_count: int) -> np.ndarray:
    segment = np.asarray(segment, dtype=np.float32).reshape(-1)
    if segment.size == 0:
        return np.zeros(8 + band_count, dtype=np.float32)

    segment = np.nan_to_num(segment, nan=0.0, posinf=0.0, neginf=0.0)
    rms = float(np.sqrt(np.mean(segment * segment)))
    mean_abs = float(np.mean(np.abs(segment)))
    peak = float(np.max(np.abs(segment)))
    zcr = float(np.mean(segment[1:] * segment[:-1] < 0.0)) if segment.size > 1 else 0.0

    windowed = segment * np.hanning(segment.size).astype(np.float32)
    spectrum = np.fft.rfft(windowed)
    power = (np.abs(spectrum) ** 2).astype(np.float64)
    freqs = np.fft.rfftfreq(segment.size, d=1.0 / float(sample_rate))
    total = float(power.sum())
    nyquist = max(sample_rate / 2.0, 1.0)
    if total <= 1e-12:
        spectral = [0.0, 0.0, 0.0]
        band_energy = np.zeros(band_count, dtype=np.float32)
    else:
        centroid = float((freqs * power).sum() / total)
        bandwidth = float(np.sqrt((((freqs - centroid) ** 2) * power).sum() / total))
        cumulative = np.cumsum(power)
        rolloff = float(freqs[int(np.searchsorted(cumulative, 0.85 * total, side="left"))])
        spectral = [centroid / nyquist, bandwidth / nyquist, rolloff / nyquist]

        edges = np.geomspace(50.0, nyquist, band_count + 1)
        band_vals = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (freqs >= lo) & (freqs < hi)
            band_vals.append(float(power[mask].sum()) if np.any(mask) else 0.0)
        band_energy = np.log1p(np.asarray(band_vals, dtype=np.float32))
        norm = np.linalg.norm(band_energy)
        if norm > 0:
            band_energy = band_energy / norm

    log_energy = float(np.log1p(total / max(segment.size, 1)))
    return np.asarray([rms, mean_abs, peak, zcr, log_energy, *spectral, *band_energy], dtype=np.float32)


def read_audio_feature_cache(
    path: Path,
    n_frames: int,
    cache_dir: Path,
    sample_rate: int,
    band_count: int,
    force: bool,
) -> tuple[np.ndarray, dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"audio_{path.stem}_n{n_frames}_sr{sample_rate}_bands{band_count}.npz"
    if cache_path.exists() and not force:
        data = np.load(cache_path, allow_pickle=True)
        meta = json.loads(str(data["metadata"].item())) if "metadata" in data else {}
        return data["features"].astype(np.float32), meta

    dim = 8 + band_count
    features = np.zeros((n_frames, dim), dtype=np.float32)
    fps = video_fps(path)
    audio = decode_audio_mono(path, sample_rate)
    has_audio = bool(audio.size > 0)
    if has_audio:
        if fps is None:
            fps = n_frames / max(audio.size / float(sample_rate), 1e-6)
        for frame_idx in range(n_frames):
            start_sample = int(round((frame_idx / fps) * sample_rate))
            end_sample = int(round(((frame_idx + 1) / fps) * sample_rate))
            start_sample = max(0, min(start_sample, audio.size))
            end_sample = max(start_sample + 1, min(end_sample, audio.size))
            features[frame_idx] = audio_segment_features(audio[start_sample:end_sample], sample_rate, band_count)
            if frame_idx and frame_idx % 1000 == 0:
                print(f"    audio/{path.name}: {frame_idx}/{n_frames} frames")

    metadata = {
        "source": path.name,
        "exists": path.exists(),
        "has_audio": has_audio,
        "sample_rate": int(sample_rate),
        "fps": float(fps) if fps is not None else None,
        "num_samples": int(audio.size),
        "per_frame_dim": int(dim),
        "band_count": int(band_count),
    }
    np.savez_compressed(cache_path, features=features, metadata=json.dumps(metadata, sort_keys=True))
    return features, metadata


TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def hashed_text(text: str, dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for token in TOKEN_RE.findall(text.lower()):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] & 1 else -1.0
        vec[bucket] += sign
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def text_for_frame(info: dict, include_label_text: bool) -> str:
    parts: list[str] = []
    objects = info.get("objects")
    if isinstance(objects, list):
        parts.extend(str(x) for x in objects)
    elif objects:
        parts.append(str(objects))
    if info.get("interaction"):
        parts.append(str(info["interaction"]))
    if include_label_text:
        for key in ("theme", "action_label", "action_desc"):
            if info.get(key):
                parts.append(str(info[key]))
    return " ".join(parts)


def build_text_features(frame_info_map: dict, n_frames: int, dim: int, include_label_text: bool) -> np.ndarray:
    frame_info_map = frame_info_map or {}
    features = np.zeros((n_frames, dim), dtype=np.float32)
    for idx in range(n_frames):
        info = frame_info_map.get(idx, {})
        features[idx] = hashed_text(text_for_frame(info, include_label_text), dim)
    return features


def prepare_modalities(args: argparse.Namespace, ann: dict) -> tuple[dict, list[dict]]:
    data_root = args.annotation.parent
    n_frames = len(ann["img_names"])
    audio_source = getattr(args, "audio_source", PRIMARY_AUDIO_VIDEO)
    if audio_source not in VIDEO_FILES:
        audio_source = PRIMARY_AUDIO_VIDEO
    audio_sample_rate = getattr(args, "audio_sample_rate", 16000)
    audio_band_count = getattr(args, "audio_band_count", 16)
    skip_video_features = bool(getattr(args, "skip_video_features", False))
    extras: dict = {
        "video": OrderedDict(),
        "audio": None,
        "audio_name": audio_source,
        "depth": None,
        "text": None,
        "static": OrderedDict(),
    }
    available = []

    print("Preparing all-modality feature caches")
    print("  depth/confidence")
    depth = read_depth_feature_cache(args.annotation, n_frames, args.cache_dir, args.depth_grid_size, args.force_rebuild_cache)
    extras["depth"] = depth
    available.append({"modality": "depth_confidence", "shape": list(depth.shape)})

    print("  videos")
    if skip_video_features:
        print("    skipped handcrafted video features")
    else:
        for name, filename in VIDEO_FILES.items():
            path = data_root / filename
            feats = read_video_feature_cache(
                path,
                n_frames,
                args.cache_dir,
                args.video_image_size,
                args.video_grid_size,
                args.video_hist_bins,
                args.force_rebuild_cache,
            )
            extras["video"][name] = feats
            available.append({
                "modality": f"video/{name}",
                "path": portable_path(path, args.workspace),
                "shape": list(feats.shape),
                "exists": path.exists(),
            })

    print("  audio")
    audio_path = data_root / VIDEO_FILES[audio_source]
    audio, audio_meta = read_audio_feature_cache(
        audio_path,
        n_frames,
        args.cache_dir,
        audio_sample_rate,
        audio_band_count,
        args.force_rebuild_cache,
    )
    extras["audio"] = audio
    available.append({
        "modality": f"audio/{audio_source}",
        "path": portable_path(audio_path, args.workspace),
        "shape": list(audio.shape),
        **audio_meta,
    })

    print("  caption objects/interaction text")
    text = build_text_features(
        ann["caption_frame_info_map"],
        n_frames,
        args.text_hash_dim,
        args.include_label_text,
    )
    extras["text"] = text
    available.append({
        "modality": "caption_text",
        "shape": list(text.shape),
        "fields": "objects,interaction" + (",theme,action_label,action_desc" if args.include_label_text else ""),
    })

    pc = point_cloud_features(ann.get("slam_point_cloud"))
    if len(pc):
        extras["static"]["slam_point_cloud"] = pc
        available.append({"modality": "slam_point_cloud_static", "shape": [int(len(pc))]})

    calib = calibration_features(ann.get("calib_data"))
    if len(calib):
        extras["static"]["calibration"] = calib
        available.append({"modality": "calibration_static", "shape": [int(len(calib))]})

    return extras, available


def extract_all_window_features(ann: dict, extras: dict, start: int, end: int, return_blocks: bool = False):
    body = safe_window(ann.get("smplh_body_joints"), start, end)
    left = safe_window(ann.get("hand_left_joints"), start, end)
    right = safe_window(ann.get("hand_right_joints"), start, end)
    contacts = safe_window(ann.get("contacts"), start, end)
    cam_t = safe_window(ann.get("t_c2w_all"), start, end)
    cam_R = safe_window(ann.get("R_c2w_all"), start, end)

    blocks: list[tuple[str, np.ndarray]] = []

    def add(name: str, vec: np.ndarray | None) -> None:
        if vec is None:
            return
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        if arr.size:
            blocks.append((name, np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)))

    if left is not None:
        add("hand_left_joints", temporal_stats(center_by_body_root(left, body)))
    if right is not None:
        add("hand_right_joints", temporal_stats(center_by_body_root(right, body)))
    if body is not None:
        root = body[:, :1, :] if body.ndim == 3 else 0.0
        add("body_joints", temporal_stats(body - root))
    if contacts is not None:
        add("body_contacts", temporal_stats(contacts))
    if cam_t is not None:
        add("camera_translation", temporal_stats(cam_t - cam_t[:1]))
    if cam_R is not None:
        add("camera_rotation_matrix", temporal_stats(cam_R))

    imu_accel = ann.get("imu_accel_xyz")
    imu_gyro = ann.get("imu_gyro_xyz")
    imu_keyframes = ann.get("imu_keyframe_indices")
    if imu_accel is not None and imu_gyro is not None and imu_keyframes is not None and len(imu_keyframes) > end - 1:
        imu_start = int(max(0, imu_keyframes[start]))
        imu_end = int(min(len(imu_accel), max(imu_start + 1, imu_keyframes[end - 1] + 1)))
        imu = np.concatenate([imu_accel[imu_start:imu_end], imu_gyro[imu_start:imu_end]], axis=1)
        add("imu_accel_gyro", temporal_stats(imu))

    if extras.get("depth") is not None:
        add("depth_confidence", temporal_stats(extras["depth"][start:end]))
    for name, feats in extras.get("video", {}).items():
        add(f"video_{name}", temporal_stats(feats[start:end]))
    if extras.get("audio") is not None:
        add(f"audio_{extras.get('audio_name', PRIMARY_AUDIO_VIDEO)}_aac", temporal_stats(extras["audio"][start:end]))
    if extras.get("text") is not None:
        add("caption_objects_interaction_text", temporal_stats(extras["text"][start:end]))
    for name, vec in extras.get("static", {}).items():
        add(name, vec)

    if not blocks:
        raise ValueError("No usable modalities found.")
    full = np.concatenate([vec for _, vec in blocks]).astype(np.float32)
    if return_blocks:
        return full, [(name, int(len(vec))) for name, vec in blocks]
    return full


def build_feature_dataset(ann: dict, extras: dict, target: str, window_frames: int, stride_frames: int, min_label_fraction: float):
    frame_info = ann.get("caption_frame_info_map")
    if frame_info is None:
        raise ValueError("No caption_frame_info_map found in annotation.")

    n_frames = len(ann["img_names"])
    X, y_labels, starts, ends, label_fracs = [], [], [], [], []
    feature_manifest = None
    for start in range(0, n_frames - window_frames + 1, stride_frames):
        end = start + window_frames
        labels = [frame_label(frame_info.get(i, {}), target) for i in range(start, end)]
        label, frac = majority_label(labels, min_label_fraction)
        if not label:
            continue
        if feature_manifest is None:
            vec, blocks = extract_all_window_features(ann, extras, start, end, return_blocks=True)
            offset = 0
            feature_manifest = []
            for name, length in blocks:
                feature_manifest.append({"name": name, "start": offset, "end": offset + length, "dim": length})
                offset += length
        else:
            vec = extract_all_window_features(ann, extras, start, end)
        X.append(vec)
        y_labels.append(label)
        starts.append(start)
        ends.append(end - 1)
        label_fracs.append(frac)

    if not X:
        raise ValueError("No labeled windows were created. Try lowering --min-label-fraction.")

    return (
        np.stack(X).astype(np.float32),
        np.asarray(y_labels, dtype=object),
        np.asarray(starts, dtype=np.int64),
        np.asarray(ends, dtype=np.int64),
        np.asarray(label_fracs, dtype=np.float32),
        feature_manifest or [],
    )


def write_extra_reports(output_dir: Path, feature_manifest: list[dict], available_modalities: list[dict], args: argparse.Namespace) -> None:
    (output_dir / "feature_manifest.json").write_text(json.dumps(feature_manifest, indent=2), encoding="utf-8")
    (output_dir / "available_modalities.json").write_text(json.dumps(available_modalities, indent=2), encoding="utf-8")
    with (output_dir / "feature_manifest.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["name", "start", "end", "dim"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(feature_manifest)
    notes = [
        "This is an all-modality lightweight baseline.",
        "RGB/stereo/fisheye/depth/point-cloud/calibration/text are compressed into handcrafted features.",
        "It is not a deep multimodal model.",
        "Do not treat random windows from one episode as a final generalization benchmark.",
    ]
    if args.include_label_text:
        notes.append("WARNING: --include-label-text was used, so language input leaks target semantics.")
    else:
        notes.append("Label text was not included as input; only objects and interaction text were used.")
    (output_dir / "README_model.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    add_toolkit_to_path(args.workspace)
    from data_loader import load_from_annotation_hdf5

    if not args.annotation.exists():
        raise FileNotFoundError(f"annotation.hdf5 not found: {args.annotation}")

    print(f"Loading annotation: {args.annotation}")
    ann = load_from_annotation_hdf5(args.annotation, 0, None, load_slam_point_cloud=True)

    extras, available_modalities = prepare_modalities(args, ann)

    print("Building all-modality windowed feature dataset")
    X, y_labels, starts, ends, label_fracs, feature_manifest = build_feature_dataset(
        ann,
        extras,
        target=args.target,
        window_frames=args.window_frames,
        stride_frames=args.stride_frames,
        min_label_fraction=args.min_label_fraction,
    )
    y, class_names = encode_labels(y_labels)
    train_idx, test_idx = stratified_split(y, args.test_fraction, args.seed)
    if len(test_idx) == 0:
        raise ValueError("No test windows available. Lower --test-fraction or use more data.")

    mean, std = fit_scaler(X[train_idx])
    X_scaled = (X - mean) / std

    print(f"Windows: {len(y)} total, {len(train_idx)} train, {len(test_idx)} test")
    print(f"Features: {X.shape[1]}, classes: {len(class_names)}")
    print("Feature blocks:")
    for block in feature_manifest:
        print(f"  {block['dim']:5d}  {block['name']}")
    for name, count in Counter(y_labels).most_common():
        print(f"  {count:4d} windows  {name}")

    print("Training softmax classifier")
    W, b, history = train_softmax_classifier(
        X_scaled[train_idx],
        y[train_idx],
        n_classes=len(class_names),
        epochs=args.epochs,
        lr=args.learning_rate,
        l2=args.l2,
        use_class_weights=not args.no_class_weights,
        seed=args.seed,
    )

    y_pred, probs = predict(X_scaled[test_idx], W, b)
    metrics, per_class_rows, cm = compute_metrics(y[test_idx], y_pred, class_names)
    majority_class = Counter(y[train_idx]).most_common(1)[0][0]
    metrics["majority_baseline_accuracy"] = float(np.mean(y[test_idx] == majority_class))
    metrics["train_final_accuracy"] = history[-1]["train_accuracy"] if history else float("nan")
    metrics["train_final_loss"] = history[-1]["loss"] if history else float("nan")
    metrics["feature_dim"] = int(X.shape[1])
    metrics["num_windows"] = int(len(y))

    save_artifacts(
        args.output_dir,
        X,
        y,
        y_labels,
        starts,
        ends,
        label_fracs,
        train_idx,
        test_idx,
        class_names,
        mean,
        std,
        W,
        b,
        history,
        metrics,
        per_class_rows,
        cm,
        y_pred,
        probs,
        args,
    )
    write_extra_reports(args.output_dir, feature_manifest, available_modalities, args)

    print("\nEvaluation")
    print(f"  accuracy:          {metrics['accuracy']:.4f}")
    print(f"  balanced_accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  macro_f1:          {metrics['macro_f1']:.4f}")
    print(f"  weighted_f1:       {metrics['weighted_f1']:.4f}")
    print(f"  majority_baseline: {metrics['majority_baseline_accuracy']:.4f}")
    print(f"\nArtifacts written to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

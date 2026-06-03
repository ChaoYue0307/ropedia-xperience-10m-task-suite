#!/usr/bin/env python3
"""Audio ablation and raw-audio feature upgrade for the Xperience-10M task suite.

This script is artifact-driven where possible. It consumes the committed
single-episode task-suite windows and feature manifest, decodes the real AAC
stream from the local public sample MP4, and writes measured task deltas.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import numpy as np

from single_episode_diagnostics import (
    TASKS,
    TASK_DISPLAY,
    block_indices,
    chronological_split,
    classification_metrics,
    encode_labels,
    frame_centers,
    labels_from_windows,
    load_inputs,
    multilabel_metrics,
    onehot,
    read_csv,
    regression_metrics,
    retrieval_metrics,
    ridge_predict,
    standardize,
    transition_labels_from_boundaries,
    write_csv,
    write_json,
)


VARIANTS = [
    "all_handcrafted_audio",
    "all_except_audio",
    "handcrafted_audio_only",
    "raw_logmel_audio_only",
    "replace_handcrafted_with_raw",
    "all_plus_raw_logmel",
]

VARIANT_DISPLAY = {
    "all_handcrafted_audio": "All Current Features",
    "all_except_audio": "All Except Audio",
    "handcrafted_audio_only": "Handcrafted AAC Audio Only",
    "raw_logmel_audio_only": "Raw Log-Mel Audio Only",
    "replace_handcrafted_with_raw": "Replace AAC Block With Raw Log-Mel",
    "all_plus_raw_logmel": "All Current Features + Raw Log-Mel",
}

PRIMARY_METRIC_HIGHER_IS_BETTER = {
    "timeline_action": True,
    "timeline_subtask": True,
    "transition_detection": True,
    "next_action": True,
    "hand_trajectory_forecast": False,
    "contact_prediction": True,
    "object_relevance": True,
    "caption_grounding": True,
    "cross_modal_retrieval": True,
    "modality_reconstruction": False,
    "temporal_order": True,
    "misalignment_detection": True,
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=root)
    parser.add_argument("--suite-dir", type=Path, default=root / "results/episode_task_suite")
    parser.add_argument("--output-dir", type=Path, default=root / "results/audio_ablation")
    parser.add_argument("--raw-sample-dir", type=Path, default=None)
    parser.add_argument("--annotation", type=Path, default=None)
    parser.add_argument("--homie-toolkit", type=Path, default=None)
    parser.add_argument("--audio-source", default="fisheye_cam0.mp4")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--mel-bands", type=int, default=64)
    parser.add_argument("--fft-size", type=int, default=512)
    parser.add_argument("--hop-length", type=int, default=160)
    parser.add_argument("--ridge-l2", type=float, default=10.0)
    parser.add_argument("--test-fraction", type=float, default=0.30)
    parser.add_argument("--future-offset-windows", type=int, default=4)
    parser.add_argument("--forecast-frames", type=int, default=10)
    parser.add_argument("--misalignment-shift-windows", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def infer_raw_sample_dir(workspace: Path, explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit.expanduser().resolve()
    candidates = [
        workspace / "data/sample/xperience-10m-sample",
        workspace.parent / "data/sample/xperience-10m-sample",
        Path.home() / "Library/CloudStorage/Dropbox/Ropedia/data/sample/xperience-10m-sample",
    ]
    for candidate in candidates:
        if (candidate / "fisheye_cam0.mp4").exists():
            return candidate.resolve()
    return None


def infer_homie_toolkit(raw_sample_dir: Path | None, explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit.expanduser().resolve()
    candidates = []
    if raw_sample_dir is not None:
        for parent in raw_sample_dir.parents:
            candidates.append(parent / "HOMIE-toolkit")
    candidates.append(Path.home() / "Library/CloudStorage/Dropbox/Ropedia/HOMIE-toolkit")
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def public_raw_sample_ref(path: Path | None) -> str:
    if path is None:
        return "not_available"
    if path.name == "fisheye_cam0.mp4":
        return "local_public_sample/fisheye_cam0.mp4"
    if path.name == "annotation.hdf5":
        return "local_public_sample/annotation.hdf5"
    return f"local_public_sample/{path.name}"


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
    return np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def video_fps(path: Path) -> float | None:
    if not path.exists() or shutil.which("ffprobe") is None:
        return None
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,r_frame_rate",
        "-of",
        "json",
        str(path),
    ]
    try:
        proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        payload = json.loads(proc.stdout)
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
        return None
    streams = payload.get("streams") or []
    for stream in streams:
        for key in ("avg_frame_rate", "r_frame_rate"):
            value = str(stream.get(key) or "")
            if "/" in value:
                num, den = value.split("/", 1)
                try:
                    fps = float(num) / max(float(den), 1e-12)
                except ValueError:
                    continue
            else:
                try:
                    fps = float(value)
                except ValueError:
                    continue
            if np.isfinite(fps) and fps > 0:
                return fps
    return None


def hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(sample_rate: int, fft_size: int, n_mels: int, f_min: float = 40.0) -> np.ndarray:
    n_freqs = fft_size // 2 + 1
    f_max = sample_rate / 2.0
    mel_points = np.linspace(hz_to_mel(np.asarray([f_min]))[0], hz_to_mel(np.asarray([f_max]))[0], n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((fft_size + 1) * hz_points / sample_rate).astype(int)
    bins = np.clip(bins, 0, n_freqs - 1)
    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = int(bins[i]), int(bins[i + 1]), int(bins[i + 2])
        if center <= left:
            center = min(left + 1, n_freqs - 1)
        if right <= center:
            right = min(center + 1, n_freqs)
        if center > left:
            fb[i, left:center] = (np.arange(left, center) - left) / max(center - left, 1)
        if right > center:
            fb[i, center:right] = (right - np.arange(center, right)) / max(right - center, 1)
    denom = fb.sum(axis=1, keepdims=True)
    denom[denom < 1e-8] = 1.0
    return fb / denom


def stft_power(segment: np.ndarray, fft_size: int, hop_length: int) -> np.ndarray:
    segment = np.asarray(segment, dtype=np.float32).reshape(-1)
    if segment.size == 0:
        return np.zeros((1, fft_size // 2 + 1), dtype=np.float32)
    if segment.size < fft_size:
        segment = np.pad(segment, (0, fft_size - segment.size))
    n_frames = 1 + max(0, (segment.size - fft_size) // hop_length)
    if n_frames <= 0:
        n_frames = 1
    window = np.hanning(fft_size).astype(np.float32)
    frames = np.zeros((n_frames, fft_size), dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_length
        chunk = segment[start : start + fft_size]
        if chunk.size < fft_size:
            chunk = np.pad(chunk, (0, fft_size - chunk.size))
        frames[i] = chunk * window
    spec = np.fft.rfft(frames, n=fft_size, axis=1)
    return (np.abs(spec) ** 2).astype(np.float32)


def raw_audio_segment_embedding(segment: np.ndarray, sample_rate: int, mel_fb: np.ndarray, fft_size: int, hop_length: int) -> np.ndarray:
    segment = np.asarray(segment, dtype=np.float32).reshape(-1)
    if segment.size == 0:
        return np.zeros(mel_fb.shape[0] * 9 + 12, dtype=np.float32)
    segment = np.nan_to_num(segment, nan=0.0, posinf=0.0, neginf=0.0)
    power = stft_power(segment, fft_size, hop_length)
    mel = np.log1p(power @ mel_fb.T)
    delta = np.diff(mel, axis=0) if mel.shape[0] > 1 else np.zeros_like(mel)
    stats = [
        mel.mean(axis=0),
        mel.std(axis=0),
        mel.min(axis=0),
        mel.max(axis=0),
        np.percentile(mel, 10, axis=0),
        np.percentile(mel, 50, axis=0),
        np.percentile(mel, 90, axis=0),
        delta.mean(axis=0),
        delta.std(axis=0),
    ]

    abs_seg = np.abs(segment)
    rms = float(np.sqrt(np.mean(segment * segment)))
    zcr = float(np.mean(segment[1:] * segment[:-1] < 0.0)) if segment.size > 1 else 0.0
    energy = abs_seg.reshape(-1)
    thirds = np.array_split(energy, 3)
    third_means = [float(x.mean()) if len(x) else 0.0 for x in thirds]
    waveform = np.asarray(
        [
            rms,
            float(abs_seg.mean()),
            float(abs_seg.std()),
            float(abs_seg.max(initial=0.0)),
            zcr,
            float(np.log1p(np.mean(segment * segment))),
            *third_means,
            float(third_means[-1] - third_means[0]),
            float(segment.size / max(sample_rate, 1)),
            float(mel.shape[0]),
        ],
        dtype=np.float32,
    )
    return np.concatenate([*stats, waveform]).astype(np.float32)


def extract_raw_audio_window_features(
    audio_path: Path,
    windows: list[dict],
    n_frames: int,
    output_dir: Path,
    sample_rate: int,
    mel_bands: int,
    fft_size: int,
    hop_length: int,
    force: bool,
) -> tuple[np.ndarray, dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / f"raw_logmel_{audio_path.stem}_sr{sample_rate}_mels{mel_bands}_fft{fft_size}_hop{hop_length}.npz"
    if cache_path.exists() and not force:
        data = np.load(cache_path, allow_pickle=True)
        return data["features"].astype(np.float32), json.loads(str(data["metadata"].item()))

    audio = decode_audio_mono(audio_path, sample_rate)
    fps = video_fps(audio_path)
    has_audio = bool(audio.size > 0)
    if has_audio and fps is None:
        fps = n_frames / max(audio.size / float(sample_rate), 1e-6)
    mel_fb = mel_filterbank(sample_rate, fft_size, mel_bands)
    feature_dim = mel_bands * 9 + 12
    features = np.zeros((len(windows), feature_dim), dtype=np.float32)
    if has_audio and fps is not None:
        for i, row in enumerate(windows):
            start_frame = int(row["start_frame"])
            end_frame = int(row["end_frame"]) + 1
            start_sample = int(round((start_frame / fps) * sample_rate))
            end_sample = int(round((end_frame / fps) * sample_rate))
            start_sample = max(0, min(start_sample, audio.size))
            end_sample = max(start_sample + 1, min(end_sample, audio.size))
            features[i] = raw_audio_segment_embedding(audio[start_sample:end_sample], sample_rate, mel_fb, fft_size, hop_length)
            if i and i % 250 == 0:
                print(f"    raw log-mel audio windows: {i}/{len(windows)}")

    metadata = {
        "source": public_raw_sample_ref(audio_path),
        "exists": bool(audio_path.exists()),
        "has_audio": has_audio,
        "sample_rate": int(sample_rate),
        "fps": float(fps) if fps is not None else None,
        "num_samples": int(audio.size),
        "num_windows": int(len(windows)),
        "feature_dim": int(features.shape[1]),
        "mel_bands": int(mel_bands),
        "fft_size": int(fft_size),
        "hop_length": int(hop_length),
        "feature_description": "Per-window raw waveform STFT log-mel statistics plus delta and waveform envelope statistics.",
    }
    np.savez_compressed(cache_path, features=features, metadata=json.dumps(metadata, sort_keys=True))
    return features, metadata


def load_annotation(annotation: Path | None, toolkit: Path | None) -> dict | None:
    if annotation is None or not annotation.exists() or toolkit is None or not toolkit.exists():
        return None
    sys.path.insert(0, str(toolkit))
    from data_loader import load_from_annotation_hdf5

    return load_from_annotation_hdf5(annotation, 0, None, load_slam_point_cloud=False)


def object_targets_from_annotation(ann: dict | None, windows: list[dict]) -> dict | None:
    if ann is None:
        return None
    frame_info = ann.get("caption_frame_info_map")
    if frame_info is None:
        return None
    vocab: OrderedDict[str, int] = OrderedDict()
    labels: list[list[str]] = []
    for row in windows:
        objects: OrderedDict[str, None] = OrderedDict()
        for frame in range(int(row["start_frame"]), int(row["end_frame"]) + 1):
            info = frame_info.get(frame, {})
            raw_objects = info.get("objects")
            if isinstance(raw_objects, list):
                for obj in raw_objects:
                    text = str(obj).strip()
                    if text:
                        objects.setdefault(text, None)
            elif raw_objects:
                text = str(raw_objects).strip()
                if text:
                    objects.setdefault(text, None)
        obj_list = list(objects.keys())
        for obj in obj_list:
            if obj not in vocab:
                vocab[obj] = len(vocab)
        labels.append(obj_list)
    if not vocab:
        return None
    Y = np.zeros((len(windows), len(vocab)), dtype=np.float32)
    for i, obj_list in enumerate(labels):
        for obj in obj_list:
            Y[i, vocab[obj]] = 1.0
    return {"Y": Y, "vocab": list(vocab.keys())}


def exact_hand_targets_from_annotation(ann: dict | None, windows: list[dict], forecast_frames: int) -> tuple[np.ndarray, np.ndarray] | None:
    if ann is None:
        return None
    left = ann.get("hand_left_joints")
    right = ann.get("hand_right_joints")
    body = ann.get("smplh_body_joints")
    if left is None or right is None:
        return None
    valid, targets = [], []
    n_frames = len(left)
    for i, row in enumerate(windows):
        future_start = int(row["end_frame"]) + 1
        future_end = future_start + forecast_frames
        if future_end > n_frames:
            continue
        hand = np.concatenate([left[future_start:future_end], right[future_start:future_end]], axis=1)
        if body is not None and future_end <= len(body):
            root = body[future_start:future_end, :1, :]
            hand = hand - root
        valid.append(i)
        targets.append(hand.reshape(-1))
    if not targets:
        return None
    return np.asarray(valid, dtype=np.int64), np.stack(targets).astype(np.float32)


def exact_contact_labels_from_annotation(ann: dict | None, windows: list[dict]) -> np.ndarray | None:
    if ann is None or ann.get("contacts") is None:
        return None
    contacts = ann["contacts"]
    labels = []
    for row in windows:
        c = contacts[int(row["start_frame"]) : int(row["end_frame"]) + 1]
        labels.append("contact" if np.any(c > 0) else "no_contact")
    return np.asarray(labels, dtype=object)


def exact_next_action_labels_from_annotation(ann: dict | None, windows: list[dict], future_frames: int = 20) -> np.ndarray | None:
    if ann is None or ann.get("caption_frame_info_map") is None:
        return None
    frame_info = ann["caption_frame_info_map"]
    n_frames = len(ann["img_names"])
    labels = []
    for row in windows:
        future_frame = min(n_frames - 1, int(row["end_frame"]) + future_frames)
        info = frame_info.get(future_frame, {})
        label = info.get("action_label") or info.get("action") or ""
        labels.append(str(label))
    return np.asarray(labels, dtype=object)


def setdiff_idx(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.setdiff1d(np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64), assume_unique=False)


def task_base_indices(task: str, manifest: list[dict]) -> np.ndarray:
    audio = block_indices(manifest, ["audio_"])
    caption = block_indices(manifest, ["caption_objects_interaction_text"])
    contact = block_indices(manifest, ["body_contacts"])
    all_idx = block_indices(manifest)
    sensor = setdiff_idx(all_idx, caption)
    if task in {"caption_grounding"}:
        return sensor
    if task in {"cross_modal_retrieval", "modality_reconstruction"}:
        return block_indices(manifest, ["hand_", "body_joints", "body_contacts", "camera_", "imu_", "audio_"])
    if task == "contact_prediction":
        return setdiff_idx(sensor, contact)
    if task == "object_relevance":
        return sensor
    return all_idx


def feature_matrix_for_variant(task: str, variant: str, X: np.ndarray, raw_audio: np.ndarray, manifest: list[dict]) -> tuple[np.ndarray, str]:
    base = task_base_indices(task, manifest)
    audio = block_indices(manifest, ["audio_"])
    base_no_audio = setdiff_idx(base, audio)
    if variant == "all_handcrafted_audio":
        return X[:, base], f"task contract feature blocks with handcrafted AAC audio where applicable ({len(base)} dims)"
    if variant == "all_except_audio":
        return X[:, base_no_audio], f"same task contract with handcrafted AAC audio columns removed ({len(base_no_audio)} dims)"
    if variant == "handcrafted_audio_only":
        return X[:, audio], f"handcrafted AAC audio block only ({len(audio)} dims)"
    if variant == "raw_logmel_audio_only":
        return raw_audio, f"raw waveform log-mel embedding only ({raw_audio.shape[1]} dims)"
    if variant == "replace_handcrafted_with_raw":
        return np.concatenate([X[:, base_no_audio], raw_audio], axis=1), (
            f"task contract with handcrafted AAC removed and raw log-mel added ({len(base_no_audio) + raw_audio.shape[1]} dims)"
        )
    if variant == "all_plus_raw_logmel":
        return np.concatenate([X[:, base], raw_audio], axis=1), (
            f"task contract with existing handcrafted AAC plus raw log-mel ({len(base) + raw_audio.shape[1]} dims)"
        )
    raise KeyError(variant)


def one_dim_target_standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return standardize(train, test)


def fit_classification_matrix(Xv: np.ndarray, labels: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, l2: float) -> dict:
    y, class_names = encode_labels(labels)
    train_classes = set(int(x) for x in y[train_idx])
    test_classes = set(int(x) for x in y[test_idx])
    unseen = [class_names[i] for i in sorted(test_classes - train_classes)]
    X_train, X_test = standardize(Xv[train_idx], Xv[test_idx])
    scores = ridge_predict(X_train, onehot(y[train_idx], len(class_names)), X_test, l2)
    pred = scores.argmax(axis=1)
    metrics = classification_metrics(y[test_idx], pred)
    metrics.update({
        "num_classes": int(len(class_names)),
        "num_train": int(len(train_idx)),
        "num_test": int(len(test_idx)),
        "unseen_test_classes": unseen,
        "unseen_test_class_count": int(len(unseen)),
    })
    return metrics


def fit_multilabel_matrix(Xv: np.ndarray, Y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, l2: float) -> dict:
    X_train, X_test = standardize(Xv[train_idx], Xv[test_idx])
    scores = ridge_predict(X_train, Y[train_idx], X_test, l2)
    pred = (scores >= 0.5).astype(np.float32)
    empty = np.where(pred.sum(axis=1) == 0)[0]
    if len(empty):
        pred[empty, np.argmax(scores[empty], axis=1)] = 1.0
    metrics = multilabel_metrics(Y[test_idx], pred)
    metrics.update({"num_objects": int(Y.shape[1]), "num_train": int(len(train_idx)), "num_test": int(len(test_idx))})
    return metrics


def fit_regression_matrix(Xv: np.ndarray, Y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, l2: float) -> dict:
    X_train, X_test = standardize(Xv[train_idx], Xv[test_idx])
    Y_train, Y_test = standardize(Y[train_idx], Y[test_idx])
    pred = ridge_predict(X_train, Y_train, X_test, l2)
    metrics = regression_metrics(Y_test, pred)
    metrics.update({"num_train": int(len(train_idx)), "num_test": int(len(test_idx)), "target_dim": int(Y.shape[1])})
    return metrics


def fit_retrieval_matrix(Xv: np.ndarray, Y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, l2: float) -> dict:
    X_train, X_test = standardize(Xv[train_idx], Xv[test_idx])
    Y_train, Y_test = standardize(Y[train_idx], Y[test_idx])
    pred = ridge_predict(X_train, Y_train, X_test, l2)
    metrics = retrieval_metrics(pred, Y_test)
    metrics.update({"num_train": int(len(train_idx)), "num_test": int(len(test_idx)), "target_dim": int(Y.shape[1])})
    return metrics


def pair_features_generic(F: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    left = F[pairs[:, 0]]
    right = F[pairs[:, 1]]
    return np.concatenate([left, right, right - left], axis=1).astype(np.float32)


def misalignment_features(
    variant: str,
    X: np.ndarray,
    raw_audio: np.ndarray,
    manifest: list[dict],
    pairs: np.ndarray,
) -> tuple[np.ndarray, str]:
    motion = block_indices(manifest, ["hand_", "body_joints", "body_contacts", "camera_", "imu_"])
    visual_audio = block_indices(manifest, ["depth_confidence", "video_", "audio_"])
    audio = block_indices(manifest, ["audio_"])
    visual_no_audio = setdiff_idx(visual_audio, audio)
    if variant == "all_handcrafted_audio":
        left = X[pairs[:, 0]][:, motion]
        right = X[pairs[:, 1]][:, visual_audio]
        return np.concatenate([left, right], axis=1).astype(np.float32), "motion/current visual+handcrafted audio pair"
    if variant == "all_except_audio":
        left = X[pairs[:, 0]][:, motion]
        right = X[pairs[:, 1]][:, visual_no_audio]
        return np.concatenate([left, right], axis=1).astype(np.float32), "motion/current visual pair with audio removed"
    if variant == "handcrafted_audio_only":
        return pair_features_generic(X[:, audio], pairs), "handcrafted AAC audio self-alignment pair"
    if variant == "raw_logmel_audio_only":
        return pair_features_generic(raw_audio, pairs), "raw log-mel audio self-alignment pair"
    if variant == "replace_handcrafted_with_raw":
        left = X[pairs[:, 0]][:, motion]
        right = np.concatenate([X[pairs[:, 1]][:, visual_no_audio], raw_audio[pairs[:, 1]]], axis=1)
        return np.concatenate([left, right], axis=1).astype(np.float32), "motion/current visual pair with raw log-mel replacing handcrafted audio"
    if variant == "all_plus_raw_logmel":
        left = X[pairs[:, 0]][:, motion]
        right = np.concatenate([X[pairs[:, 1]][:, visual_audio], raw_audio[pairs[:, 1]]], axis=1)
        return np.concatenate([left, right], axis=1).astype(np.float32), "motion/current visual+handcrafted audio pair plus raw log-mel"
    raise KeyError(variant)


def task_target(
    task: str,
    X: np.ndarray,
    windows: list[dict],
    manifest: list[dict],
    suite_dir: Path,
    args: argparse.Namespace,
    raw_targets: dict,
) -> dict:
    n = len(windows)
    all_rows = np.arange(n, dtype=np.int64)
    if task == "timeline_action":
        return {"kind": "classification", "labels": labels_from_windows(windows, "action_label"), "rows": all_rows, "metric": "macro_f1"}
    if task == "timeline_subtask":
        return {"kind": "classification", "labels": labels_from_windows(windows, "subtask_label"), "rows": all_rows, "metric": "macro_f1"}
    if task == "transition_detection":
        labels = transition_labels_from_boundaries(suite_dir, frame_centers(windows))
        return {"kind": "classification", "labels": labels, "rows": all_rows, "metric": "macro_f1"}
    if task == "next_action":
        labels = raw_targets.get("next_action_labels")
        if labels is not None:
            return {"kind": "classification", "labels": labels, "rows": all_rows, "metric": "macro_f1", "target_variant": "future action from annotation frame labels"}
        rows = np.arange(0, n - args.future_offset_windows, dtype=np.int64)
        labels = labels_from_windows(windows, "action_label")[rows + args.future_offset_windows]
        return {"kind": "classification", "labels": labels, "rows": rows, "metric": "macro_f1", "target_variant": "future action from windows.csv"}
    if task == "contact_prediction":
        labels = raw_targets.get("contact_labels")
        if labels is None:
            contacts = block_indices(manifest, ["body_contacts"])
            labels = np.where(np.abs(X[:, contacts]).sum(axis=1) > 1e-8, "contact", "no_contact")
        return {"kind": "classification", "labels": labels, "rows": all_rows, "metric": "macro_f1"}
    if task == "object_relevance":
        obj = raw_targets.get("object_targets")
        if obj is None:
            return {"kind": "not_available", "reason": "object labels require local annotation.hdf5"}
        return {"kind": "multilabel", "target": obj["Y"], "rows": all_rows, "metric": "micro_f1", "num_objects": len(obj["vocab"])}
    if task == "hand_trajectory_forecast":
        exact = raw_targets.get("hand_targets")
        if exact is not None:
            rows, target = exact
            return {"kind": "regression", "target": target, "rows": rows, "metric": "mae", "target_variant": "future hand joints from annotation.hdf5"}
        rows = np.arange(0, n - args.future_offset_windows, dtype=np.int64)
        hand = block_indices(manifest, ["hand_left_joints", "hand_right_joints"])
        return {"kind": "regression", "target": X[rows + args.future_offset_windows][:, hand], "rows": rows, "metric": "mae", "target_variant": "future hand feature block"}
    if task == "caption_grounding":
        text = block_indices(manifest, ["caption_objects_interaction_text"])
        return {"kind": "retrieval", "target": X[:, text], "rows": all_rows, "metric": "mrr"}
    if task in {"cross_modal_retrieval", "modality_reconstruction"}:
        visual = block_indices(manifest, ["depth_confidence", "video_"])
        return {"kind": "retrieval" if task == "cross_modal_retrieval" else "regression", "target": X[:, visual], "rows": all_rows, "metric": "mrr" if task == "cross_modal_retrieval" else "mae"}
    if task == "temporal_order":
        pairs, labels = [], []
        for i in range(n - 1):
            pairs.append((i, i + 1))
            labels.append("forward")
            pairs.append((i + 1, i))
            labels.append("reversed")
        return {"kind": "pair_classification", "pairs": np.asarray(pairs, dtype=np.int64), "labels": np.asarray(labels, dtype=object), "metric": "macro_f1"}
    if task == "misalignment_detection":
        pairs, labels = [], []
        shift = args.misalignment_shift_windows
        for i in range(n - shift):
            pairs.append((i, i))
            labels.append("aligned")
            pairs.append((i, i + shift))
            labels.append("shifted")
        return {"kind": "misalignment", "pairs": np.asarray(pairs, dtype=np.int64), "labels": np.asarray(labels, dtype=object), "metric": "macro_f1"}
    raise KeyError(task)


def evaluate_task_variant(
    task: str,
    variant: str,
    X: np.ndarray,
    raw_audio: np.ndarray,
    windows: list[dict],
    manifest: list[dict],
    suite_dir: Path,
    args: argparse.Namespace,
    raw_targets: dict,
) -> dict:
    info = task_target(task, X, windows, manifest, suite_dir, args, raw_targets)
    row = {
        "task": task,
        "task_display": TASK_DISPLAY.get(task, task),
        "variant": variant,
        "variant_display": VARIANT_DISPLAY[variant],
        "status": "computed",
        "primary_metric": info.get("metric", ""),
        "primary_value": "",
        "higher_is_better": str(PRIMARY_METRIC_HIGHER_IS_BETTER[task]).lower(),
        "feature_dim": "",
        "num_train": "",
        "num_test": "",
        "input_contract": "",
        "target_variant": info.get("target_variant", ""),
        "reason": "",
    }
    if info["kind"] == "not_available":
        row.update({"status": "not_computed", "reason": info["reason"]})
        return row
    try:
        if info["kind"] == "misalignment":
            feats, desc = misalignment_features(variant, X, raw_audio, manifest, np.asarray(info["pairs"], dtype=np.int64))
            labels = np.asarray(info["labels"], dtype=object)
            train_idx, test_idx = chronological_split(len(labels), args.test_fraction)
            metrics = fit_classification_matrix(feats, labels, train_idx, test_idx, args.ridge_l2)
            row["input_contract"] = desc
        elif info["kind"] == "pair_classification":
            F, desc = feature_matrix_for_variant(task, variant, X, raw_audio, manifest)
            feats = pair_features_generic(F, np.asarray(info["pairs"], dtype=np.int64))
            labels = np.asarray(info["labels"], dtype=object)
            train_idx, test_idx = chronological_split(len(labels), args.test_fraction)
            metrics = fit_classification_matrix(feats, labels, train_idx, test_idx, args.ridge_l2)
            row["input_contract"] = desc
        else:
            F, desc = feature_matrix_for_variant(task, variant, X, raw_audio, manifest)
            data_rows = np.asarray(info["rows"], dtype=np.int64)
            train_idx, test_idx = chronological_split(len(data_rows), args.test_fraction)
            if info["kind"] == "classification":
                metrics = fit_classification_matrix(F[data_rows], np.asarray(info["labels"], dtype=object), train_idx, test_idx, args.ridge_l2)
            elif info["kind"] == "multilabel":
                metrics = fit_multilabel_matrix(F[data_rows], np.asarray(info["target"], dtype=np.float32), train_idx, test_idx, args.ridge_l2)
            elif info["kind"] == "regression":
                metrics = fit_regression_matrix(F[data_rows], np.asarray(info["target"], dtype=np.float32), train_idx, test_idx, args.ridge_l2)
            elif info["kind"] == "retrieval":
                metrics = fit_retrieval_matrix(F[data_rows], np.asarray(info["target"], dtype=np.float32), train_idx, test_idx, args.ridge_l2)
            else:
                raise KeyError(info["kind"])
            row["input_contract"] = desc
            row["feature_dim"] = int(F.shape[1])
        row.update(metrics)
        row["primary_value"] = float(metrics[info["metric"]])
        row["num_train"] = int(metrics.get("num_train", row.get("num_train") or 0))
        row["num_test"] = int(metrics.get("num_test", row.get("num_test") or 0))
        if row["feature_dim"] == "":
            row["feature_dim"] = int(feats.shape[1])
    except Exception as exc:
        row.update({"status": "not_computed", "reason": f"{type(exc).__name__}: {exc}"})
    return row


def delta(base: float, compare: float, higher_is_better: bool) -> float:
    return compare - base if higher_is_better else base - compare


def build_summary(rows: list[dict], raw_meta: dict) -> dict:
    by_task: dict[str, dict[str, dict]] = {}
    for row in rows:
        if row.get("status") != "computed":
            continue
        by_task.setdefault(row["task"], {})[row["variant"]] = row
    task_summaries = []
    for task in TASKS:
        variants = by_task.get(task, {})
        base = variants.get("all_handcrafted_audio")
        no_audio = variants.get("all_except_audio")
        raw_only = variants.get("raw_logmel_audio_only")
        replace = variants.get("replace_handcrafted_with_raw")
        plus = variants.get("all_plus_raw_logmel")
        if not base:
            continue
        higher = PRIMARY_METRIC_HIGHER_IS_BETTER[task]
        item = {
            "task": task,
            "task_display": TASK_DISPLAY.get(task, task),
            "primary_metric": base["primary_metric"],
            "higher_is_better": higher,
            "all_handcrafted_audio": float(base["primary_value"]),
        }
        if no_audio:
            item["all_except_audio"] = float(no_audio["primary_value"])
            item["handcrafted_audio_delta"] = delta(float(no_audio["primary_value"]), float(base["primary_value"]), higher)
        if raw_only:
            item["raw_logmel_audio_only"] = float(raw_only["primary_value"])
        if replace and no_audio:
            item["replace_handcrafted_with_raw"] = float(replace["primary_value"])
            item["raw_replacement_delta_vs_no_audio"] = delta(float(no_audio["primary_value"]), float(replace["primary_value"]), higher)
            item["raw_replacement_delta_vs_handcrafted"] = delta(float(base["primary_value"]), float(replace["primary_value"]), higher)
        if plus:
            item["all_plus_raw_logmel"] = float(plus["primary_value"])
            item["all_plus_raw_delta_vs_handcrafted"] = delta(float(base["primary_value"]), float(plus["primary_value"]), higher)
        task_summaries.append(item)

    handcrafted_deltas = [x["handcrafted_audio_delta"] for x in task_summaries if "handcrafted_audio_delta" in x]
    raw_replace_deltas = [x["raw_replacement_delta_vs_handcrafted"] for x in task_summaries if "raw_replacement_delta_vs_handcrafted" in x]
    return {
        "description": "Measured audio ablation and raw log-mel audio upgrade over the single public Xperience-10M sample episode.",
        "scope": "single public sample episode; chronological split; ridge heads over fixed feature contracts",
        "raw_audio_metadata": raw_meta,
        "num_tasks": len(task_summaries),
        "variants": VARIANT_DISPLAY,
        "task_summaries": task_summaries,
        "aggregate": {
            "mean_handcrafted_audio_delta": float(np.mean(handcrafted_deltas)) if handcrafted_deltas else None,
            "tasks_where_handcrafted_audio_improves": int(sum(1 for x in handcrafted_deltas if x > 0)),
            "mean_raw_replacement_delta_vs_handcrafted": float(np.mean(raw_replace_deltas)) if raw_replace_deltas else None,
            "tasks_where_raw_replacement_improves_over_handcrafted": int(sum(1 for x in raw_replace_deltas if x > 0)),
        },
    }


def write_summary_markdown(path: Path, summary: dict) -> None:
    lines = [
        "# Audio Ablation and Raw-Audio Upgrade",
        "",
        "This report is generated from committed task-suite artifacts plus the local public-sample MP4 audio stream.",
        "It measures whether audio changes each single-episode task under the same chronological split.",
        "",
        "## Raw Audio Feature",
        "",
    ]
    meta = summary["raw_audio_metadata"]
    lines.extend([
        f"- Source: `{meta.get('source')}`",
        f"- Has audio: `{meta.get('has_audio')}`",
        f"- Sample rate: `{meta.get('sample_rate')}`",
        f"- Window feature dim: `{meta.get('feature_dim')}`",
        f"- Feature: {meta.get('feature_description')}",
        "",
        "## Task Deltas",
        "",
        "| Task | Metric | Current audio | No audio | Current audio delta | Raw replaces audio | Raw replacement delta |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for item in summary["task_summaries"]:
        lines.append(
            "| {task} | {metric} | {cur:.4f} | {no:.4f} | {d1:.4f} | {raw:.4f} | {d2:.4f} |".format(
                task=item["task_display"],
                metric=item["primary_metric"],
                cur=item.get("all_handcrafted_audio", float("nan")),
                no=item.get("all_except_audio", float("nan")),
                d1=item.get("handcrafted_audio_delta", float("nan")),
                raw=item.get("replace_handcrafted_with_raw", float("nan")),
                d2=item.get("raw_replacement_delta_vs_handcrafted", float("nan")),
            )
        )
    agg = summary["aggregate"]
    lines.extend([
        "",
        "## Aggregate",
        "",
        f"- Mean current-audio delta: `{agg['mean_handcrafted_audio_delta']}`",
        f"- Tasks where current handcrafted audio improves the primary metric: `{agg['tasks_where_handcrafted_audio_improves']}`",
        f"- Mean raw-replacement delta vs current handcrafted audio: `{agg['mean_raw_replacement_delta_vs_handcrafted']}`",
        f"- Tasks where raw log-mel replacement improves over current handcrafted audio: `{agg['tasks_where_raw_replacement_improves_over_handcrafted']}`",
        "",
        "Positive deltas always mean better according to each task's primary metric. For MAE tasks, lower MAE is converted into a positive improvement.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_delta_chart(path: Path, summary: dict) -> None:
    items = summary["task_summaries"]
    width = 1320
    row_h = 42
    height = 120 + row_h * len(items)
    max_abs = max([abs(x.get("handcrafted_audio_delta", 0.0)) for x in items] + [1e-6])
    left = 410
    mid = 680
    scale = 240 / max_abs
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#07110d"/>',
        '<text x="36" y="42" fill="#e6f7ea" font-family="Arial, sans-serif" font-size="28" font-weight="700">Measured Audio Delta Across 12 Xperience-10M Tasks</text>',
        '<text x="36" y="70" fill="#a7b8ab" font-family="Arial, sans-serif" font-size="15">Positive means audio improved the task primary metric on the single public sample split.</text>',
        f'<line x1="{mid}" y1="92" x2="{mid}" y2="{height - 24}" stroke="#5b6f61" stroke-width="1"/>',
    ]
    for i, item in enumerate(items):
        y = 112 + i * row_h
        task = item["task_display"].replace("&", "&amp;")
        value = float(item.get("handcrafted_audio_delta", 0.0))
        bar_w = abs(value) * scale
        x = mid if value >= 0 else mid - bar_w
        color = "#7ae5c3" if value >= 0 else "#ff8a6a"
        lines.extend([
            f'<text x="36" y="{y + 18}" fill="#d8eadc" font-family="Arial, sans-serif" font-size="15">{task}</text>',
            f'<rect x="{x:.2f}" y="{y}" width="{bar_w:.2f}" height="22" rx="3" fill="{color}"/>',
            f'<text x="{mid + 270}" y="{y + 17}" fill="#d8eadc" font-family="Arial, sans-serif" font-size="14">{value:+.4f} {item["primary_metric"]}</text>',
        ])
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_sample_dir = infer_raw_sample_dir(args.workspace, args.raw_sample_dir)
    audio_path = raw_sample_dir / args.audio_source if raw_sample_dir is not None else Path(args.audio_source)
    annotation = args.annotation or (raw_sample_dir / "annotation.hdf5" if raw_sample_dir is not None else None)
    toolkit = infer_homie_toolkit(raw_sample_dir, args.homie_toolkit)

    if raw_sample_dir is None or not audio_path.exists():
        raise FileNotFoundError("Local public sample MP4 is required for raw-audio upgrade. Pass --raw-sample-dir.")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to decode the MP4 audio stream.")

    X, _starts, _ends, windows, manifest, _summary = load_inputs(args.suite_dir)
    raw_audio, raw_meta = extract_raw_audio_window_features(
        audio_path,
        windows,
        X.shape[0],
        args.output_dir,
        args.sample_rate,
        args.mel_bands,
        args.fft_size,
        args.hop_length,
        args.force,
    )
    ann = load_annotation(annotation, toolkit)
    raw_targets = {
        "object_targets": object_targets_from_annotation(ann, windows),
        "hand_targets": exact_hand_targets_from_annotation(ann, windows, args.forecast_frames),
        "contact_labels": exact_contact_labels_from_annotation(ann, windows),
        "next_action_labels": exact_next_action_labels_from_annotation(ann, windows),
    }

    rows: list[dict] = []
    for task in TASKS:
        print(f"Audio ablation task: {task}")
        for variant in VARIANTS:
            rows.append(evaluate_task_variant(task, variant, X, raw_audio, windows, manifest, args.suite_dir, args, raw_targets))

    write_csv(args.output_dir / "audio_ablation_metrics.csv", rows)
    summary = build_summary(rows, raw_meta)
    summary["provenance"] = {
        "suite_dir": "results/episode_task_suite",
        "shared_windows": "results/episode_task_suite/shared_windows.npz",
        "feature_manifest": "results/episode_task_suite/feature_manifest.json",
        "audio_source": public_raw_sample_ref(audio_path),
        "annotation_source": public_raw_sample_ref(annotation) if annotation is not None and annotation.exists() else "not_available",
        "homie_toolkit_available": bool(toolkit is not None and toolkit.exists()),
    }
    write_json(args.output_dir / "audio_ablation_summary.json", summary)
    write_summary_markdown(args.output_dir / "AUDIO_ABLATION_SUMMARY.md", summary)
    write_delta_chart(args.workspace / "docs/assets/charts/audio_ablation_delta.svg", summary)
    write_json(args.workspace / "docs/data/audio_ablation_summary.json", summary)

    compact_rows = []
    for item in summary["task_summaries"]:
        compact_rows.append({
            "task": item["task"],
            "task_display": item["task_display"],
            "metric": item["primary_metric"],
            "current_audio": item.get("all_handcrafted_audio", ""),
            "no_audio": item.get("all_except_audio", ""),
            "current_audio_delta": item.get("handcrafted_audio_delta", ""),
            "raw_audio_only": item.get("raw_logmel_audio_only", ""),
            "replace_with_raw": item.get("replace_handcrafted_with_raw", ""),
            "raw_replacement_delta_vs_current": item.get("raw_replacement_delta_vs_handcrafted", ""),
            "all_plus_raw": item.get("all_plus_raw_logmel", ""),
            "all_plus_raw_delta_vs_current": item.get("all_plus_raw_delta_vs_handcrafted", ""),
        })
    write_csv(args.output_dir / "audio_delta_summary.csv", compact_rows)
    print(f"Wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

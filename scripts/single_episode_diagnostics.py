#!/usr/bin/env python3
"""
Single-episode diagnostics for the Xperience-10M task suite artifacts.

This script is intentionally artifact-driven. It consumes the already exported
one-episode shared feature table and prediction files, validates their shape and
hashes, and writes diagnostics that can be manually traced back to those inputs.

It does not invent labels or claim multi-episode generalization.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import numpy as np


TASKS = [
    "timeline_action",
    "timeline_subtask",
    "transition_detection",
    "next_action",
    "hand_trajectory_forecast",
    "contact_prediction",
    "object_relevance",
    "caption_grounding",
    "cross_modal_retrieval",
    "modality_reconstruction",
    "temporal_order",
    "misalignment_detection",
]


TASK_DISPLAY = {
    "timeline_action": "Current Action Recognition",
    "timeline_subtask": "Current Subtask Recognition",
    "transition_detection": "Action Transition Detection",
    "next_action": "Next-Action Prediction",
    "hand_trajectory_forecast": "Future Hand Motion Forecasting",
    "contact_prediction": "Contact State Prediction",
    "object_relevance": "Relevant Object Prediction",
    "caption_grounding": "Language-to-Time Grounding",
    "cross_modal_retrieval": "Cross-Modal Window Retrieval",
    "modality_reconstruction": "Sensor-to-Visual Reconstruction",
    "temporal_order": "Temporal Order Verification",
    "misalignment_detection": "Cross-Modal Misalignment Detection",
}


GROUP_DISPLAY = {
    "all_features": "All Features",
    "video": "Video",
    "depth": "Depth",
    "pose_slam": "Pose + SLAM",
    "motion_capture": "Motion Capture",
    "inertial": "Inertial",
    "language": "Language",
    "no_language": "All Except Language",
    "motion_pose_inertial": "Motion + Pose + IMU",
}


def parse_args() -> argparse.Namespace:
    workspace_default = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run single-episode diagnostics on real exported artifacts.")
    parser.add_argument("--workspace", type=Path, default=workspace_default)
    parser.add_argument(
        "--suite-dir",
        type=Path,
        default=workspace_default / "results/episode_task_suite",
        help="Existing single-episode task-suite artifact directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=workspace_default / "results/single_episode_diagnostics",
        help="Where to write new diagnostics. Existing task-suite outputs are not overwritten.",
    )
    parser.add_argument("--test-fraction", type=float, default=0.30)
    parser.add_argument("--future-offset-windows", type=int, default=4)
    parser.add_argument("--misalignment-shift-windows", type=int, default=8)
    parser.add_argument("--ridge-l2", type=float, default=10.0)
    parser.add_argument(
        "--annotation",
        type=Path,
        default=None,
        help="Optional raw annotation.hdf5. When provided, object relevance labels are exported from caption_frame_info_map.",
    )
    parser.add_argument(
        "--homie-toolkit",
        type=Path,
        default=None,
        help="Optional HOMIE-toolkit path. If omitted, inferred from --annotation when possible.",
    )
    return parser.parse_args()


def write_json(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: OrderedDict[str, None] = OrderedDict()
        for row in rows:
            for key in row:
                keys.setdefault(key, None)
        fieldnames = list(keys.keys())
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def public_artifact_path(path: Path, repo_root: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        pass
    if path.name == "annotation.hdf5":
        return "external_raw_sample/annotation.hdf5"
    return path.name


def public_source_reference(value: object) -> object:
    if value in (None, ""):
        return value
    text = str(value)
    if text.startswith("/") or "/" + "Users/" in text or "/" + "private/" in text:
        path = Path(text)
        if path.name == "annotation.hdf5":
            return "external_raw_sample/annotation.hdf5"
        return path.name
    return text


def load_inputs(suite_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict], list[dict], dict]:
    npz_path = suite_dir / "shared_windows.npz"
    windows_path = suite_dir / "windows.csv"
    manifest_path = suite_dir / "feature_manifest.json"
    summary_path = suite_dir / "summary_report.json"
    required = [npz_path, windows_path, manifest_path, summary_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required input artifacts: {missing}")

    npz = np.load(npz_path)
    X = np.asarray(npz["X"], dtype=np.float32)
    starts = np.asarray(npz["starts"], dtype=np.int64)
    ends = np.asarray(npz["ends"], dtype=np.int64)
    windows = read_csv(windows_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    if X.ndim != 2:
        raise ValueError(f"Expected X to be 2-D, got shape {X.shape}")
    if len(windows) != X.shape[0]:
        raise ValueError(f"windows.csv rows ({len(windows)}) do not match X rows ({X.shape[0]})")
    if len(starts) != X.shape[0] or len(ends) != X.shape[0]:
        raise ValueError("starts/ends arrays do not match X rows")

    for i, row in enumerate(windows):
        if int(row["start_frame"]) != int(starts[i]) or int(row["end_frame"]) != int(ends[i]):
            raise ValueError(f"Window start/end mismatch at row {i}")

    cursor = 0
    for block in manifest:
        start, end, dim = int(block["start"]), int(block["end"]), int(block["dim"])
        if start != cursor or end <= start or end - start != dim:
            raise ValueError(f"Feature manifest has a gap, overlap, or bad dim at block {block}")
        cursor = end
    if cursor != X.shape[1]:
        raise ValueError(f"Feature manifest ends at {cursor}, but X has {X.shape[1]} columns")

    return X, starts, ends, windows, manifest, summary


def chronological_split(n: int, test_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    if n < 2:
        raise ValueError("Need at least two samples for a chronological split.")
    split = int(round(n * (1.0 - test_fraction)))
    split = max(1, min(split, n - 1))
    return np.arange(split, dtype=np.int64), np.arange(split, n, dtype=np.int64)


def block_indices(manifest: list[dict], include: Iterable[str] | None = None, exclude: Iterable[str] | None = None) -> np.ndarray:
    include = list(include or [])
    exclude = list(exclude or [])
    idxs: list[int] = []
    for block in manifest:
        name = str(block["name"])
        if include and not any(name == p or name.startswith(p) for p in include):
            continue
        if exclude and any(name == p or name.startswith(p) for p in exclude):
            continue
        idxs.extend(range(int(block["start"]), int(block["end"])))
    return np.asarray(idxs, dtype=np.int64)


def modality_groups(manifest: list[dict]) -> dict[str, np.ndarray]:
    all_idx = block_indices(manifest)
    language = block_indices(manifest, ["caption_objects_interaction_text"])
    groups = {
        "all_features": all_idx,
        "video": block_indices(manifest, ["video_"]),
        "depth": block_indices(manifest, ["depth_confidence"]),
        "pose_slam": block_indices(manifest, ["camera_translation", "camera_rotation_matrix", "slam_point_cloud", "calibration"]),
        "motion_capture": block_indices(manifest, ["hand_left_joints", "hand_right_joints", "body_joints", "body_contacts"]),
        "inertial": block_indices(manifest, ["imu_accel_gyro"]),
        "language": language,
        "no_language": np.setdiff1d(all_idx, language),
    }
    return {name: idx for name, idx in groups.items() if len(idx) > 0}


def encode_labels(labels: Iterable[str]) -> tuple[np.ndarray, list[str]]:
    seen: OrderedDict[str, int] = OrderedDict()
    encoded = []
    for label in labels:
        label = str(label)
        if label not in seen:
            seen[label] = len(seen)
        encoded.append(seen[label])
    return np.asarray(encoded, dtype=np.int64), list(seen.keys())


def extract_objects(info: dict) -> list[str]:
    objects = info.get("objects")
    if isinstance(objects, list):
        return [str(x).strip() for x in objects if str(x).strip()]
    if objects:
        return [str(objects).strip()]
    return []


def infer_homie_toolkit(annotation: Path, explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit
    annotation = annotation.resolve()
    candidates = []
    for parent in annotation.parents:
        candidates.append(parent / "HOMIE-toolkit")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_object_targets_from_annotation(annotation: Path, windows: list[dict], toolkit: Path | None) -> dict:
    annotation = annotation.resolve()
    if not annotation.exists():
        raise FileNotFoundError(annotation)
    toolkit = infer_homie_toolkit(annotation, toolkit)
    if toolkit is None or not toolkit.exists():
        raise FileNotFoundError(f"HOMIE-toolkit not found for annotation {annotation}")
    sys.path.insert(0, str(toolkit))
    from data_loader import load_from_annotation_hdf5

    ann = load_from_annotation_hdf5(annotation, 0, None, load_slam_point_cloud=False)
    frame_info = ann["caption_frame_info_map"]
    vocab: OrderedDict[str, int] = OrderedDict()
    labels: list[list[str]] = []
    rows_out: list[dict] = []
    for row in windows:
        counts: OrderedDict[str, int] = OrderedDict()
        for frame in range(int(row["start_frame"]), int(row["end_frame"]) + 1):
            for obj in extract_objects(frame_info.get(frame, {})):
                counts[obj] = counts.get(obj, 0) + 1
        objects = list(counts.keys())
        for obj in objects:
            if obj not in vocab:
                vocab[obj] = len(vocab)
        labels.append(objects)
        rows_out.append({
            "window_index": int(row["window_index"]),
            "start_frame": int(row["start_frame"]),
            "end_frame": int(row["end_frame"]),
            "center_frame": int(row["center_frame"]),
            "objects": "|".join(objects),
            "object_count": int(len(objects)),
        })
    if not vocab:
        raise ValueError("No object labels found in annotation caption_frame_info_map.")
    Y = np.zeros((len(windows), len(vocab)), dtype=np.float32)
    for i, objects in enumerate(labels):
        for obj in objects:
            Y[i, vocab[obj]] = 1.0
    return {
        "Y": Y,
        "labels": labels,
        "vocab": list(vocab.keys()),
        "rows": rows_out,
        "annotation": "external_raw_sample/annotation.hdf5",
        "toolkit": "HOMIE-toolkit",
        "source_note": (
            "Object labels were exported from a raw Xperience-10M sample annotation. "
            "The public artifact stores source type and hash instead of machine-specific file paths."
        ),
    }


def standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (train - mean) / std, (test - mean) / std


def standardize_train_apply(train: np.ndarray, *arrays: np.ndarray) -> list[np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return [(arr - mean) / std for arr in arrays]


def ridge_predict(X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray, l2: float) -> np.ndarray:
    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    Y_train = np.asarray(Y_train, dtype=np.float32)
    Xb = np.concatenate([X_train, np.ones((X_train.shape[0], 1), dtype=np.float32)], axis=1)
    Xtb = np.concatenate([X_test, np.ones((X_test.shape[0], 1), dtype=np.float32)], axis=1)
    if Xb.shape[0] <= Xb.shape[1]:
        K = Xb @ Xb.T
        K.flat[:: K.shape[0] + 1] += l2
        alpha = np.linalg.solve(K, Y_train)
        return Xtb @ Xb.T @ alpha
    A = Xb.T @ Xb
    A.flat[:: A.shape[0] + 1] += l2
    W = np.linalg.solve(A, Xb.T @ Y_train)
    return Xtb @ W


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    classes = np.unique(np.concatenate([y_true, y_pred]))
    f1s = []
    recalls = []
    for cls in classes:
        tp = int(((y_true == cls) & (y_pred == cls)).sum())
        fp = int(((y_true != cls) & (y_pred == cls)).sum())
        fn = int(((y_true == cls) & (y_pred != cls)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        recalls.append(recall)
    return {
        "accuracy": float((y_true == y_pred).mean()) if len(y_true) else 0.0,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "balanced_accuracy": float(np.mean(recalls)) if recalls else 0.0,
    }


def multilabel_metrics(Y_true: np.ndarray, Y_pred: np.ndarray) -> dict:
    Y_true = Y_true.astype(np.int64)
    Y_pred = Y_pred.astype(np.int64)
    tp = int(((Y_true == 1) & (Y_pred == 1)).sum())
    fp = int(((Y_true == 0) & (Y_pred == 1)).sum())
    fn = int(((Y_true == 1) & (Y_pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    micro_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    per_f1 = []
    for j in range(Y_true.shape[1]):
        tpj = int(((Y_true[:, j] == 1) & (Y_pred[:, j] == 1)).sum())
        fpj = int(((Y_true[:, j] == 0) & (Y_pred[:, j] == 1)).sum())
        fnj = int(((Y_true[:, j] == 1) & (Y_pred[:, j] == 0)).sum())
        pj = tpj / (tpj + fpj) if tpj + fpj else 0.0
        rj = tpj / (tpj + fnj) if tpj + fnj else 0.0
        per_f1.append(2 * pj * rj / (pj + rj) if pj + rj else 0.0)
    return {
        "micro_f1": float(micro_f1),
        "macro_f1": float(np.mean(per_f1)) if per_f1 else 0.0,
        "exact_match": float(np.mean(np.all(Y_true == Y_pred, axis=1))) if len(Y_true) else 0.0,
        "precision": float(precision),
        "recall": float(recall),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))
    denom = float(np.sum((y_true - y_true.mean(axis=0, keepdims=True)) ** 2))
    r2 = 1.0 - float(np.sum(err ** 2)) / denom if denom > 1e-12 else 0.0
    return {"mse": mse, "mae": mae, "r2": r2}


def retrieval_metrics(pred_query: np.ndarray, target: np.ndarray) -> dict:
    pred_query = pred_query.astype(np.float32)
    target = target.astype(np.float32)
    q_norm = pred_query / np.maximum(np.linalg.norm(pred_query, axis=1, keepdims=True), 1e-8)
    t_norm = target / np.maximum(np.linalg.norm(target, axis=1, keepdims=True), 1e-8)
    sims = q_norm @ t_norm.T
    ranks = []
    for i in range(sims.shape[0]):
        order = np.argsort(-sims[i])
        rank = int(np.where(order == i)[0][0]) + 1
        ranks.append(rank)
    ranks_arr = np.asarray(ranks, dtype=np.float32)
    return {
        "mrr": float(np.mean(1.0 / ranks_arr)) if len(ranks_arr) else 0.0,
        "top1_accuracy": float(np.mean(ranks_arr <= 1)) if len(ranks_arr) else 0.0,
        "top5_accuracy": float(np.mean(ranks_arr <= 5)) if len(ranks_arr) else 0.0,
        "top10_accuracy": float(np.mean(ranks_arr <= 10)) if len(ranks_arr) else 0.0,
        "median_rank": float(np.median(ranks_arr)) if len(ranks_arr) else 0.0,
        "mean_rank": float(np.mean(ranks_arr)) if len(ranks_arr) else 0.0,
        "num_queries": int(len(ranks_arr)),
    }


def onehot(y: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((len(y), n_classes), dtype=np.float32)
    out[np.arange(len(y)), y] = 1.0
    return out


def fit_classification(
    X: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    l2: float,
) -> tuple[dict, np.ndarray]:
    y, class_names = encode_labels(labels)
    train_classes = set(int(x) for x in y[train_idx])
    test_classes = set(int(x) for x in y[test_idx])
    unseen = [class_names[i] for i in sorted(test_classes - train_classes)]
    X_train, X_test = standardize(X[train_idx], X[test_idx])
    scores = ridge_predict(X_train, onehot(y[train_idx], len(class_names)), X_test, l2)
    pred = scores.argmax(axis=1)
    metrics = classification_metrics(y[test_idx], pred)
    metrics.update({
        "num_classes": len(class_names),
        "num_train": int(len(train_idx)),
        "num_test": int(len(test_idx)),
        "unseen_test_classes": "|".join(unseen),
        "unseen_test_class_count": int(len(unseen)),
    })
    return metrics, pred


def fit_multilabel(
    X: np.ndarray,
    Y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    l2: float,
) -> tuple[dict, np.ndarray]:
    X_train, X_test = standardize(X[train_idx], X[test_idx])
    scores = ridge_predict(X_train, Y[train_idx], X_test, l2)
    pred = (scores >= 0.5).astype(np.float32)
    empty = np.where(pred.sum(axis=1) == 0)[0]
    if len(empty):
        pred[empty, np.argmax(scores[empty], axis=1)] = 1.0
    metrics = multilabel_metrics(Y[test_idx], pred)
    metrics.update({
        "num_objects": int(Y.shape[1]),
        "num_train": int(len(train_idx)),
        "num_test": int(len(test_idx)),
    })
    return metrics, pred


def frame_centers(windows: list[dict]) -> np.ndarray:
    return np.asarray([int(row["center_frame"]) for row in windows], dtype=np.int64)


def labels_from_windows(windows: list[dict], key: str) -> np.ndarray:
    return np.asarray([str(row.get(key, "") or "") for row in windows], dtype=object)


def transition_labels_from_boundaries(suite_dir: Path, centers: np.ndarray, tolerance_frames: int = 10) -> np.ndarray:
    boundaries_path = suite_dir / "transition_detection/true_boundaries.csv"
    if not boundaries_path.exists():
        raise FileNotFoundError(boundaries_path)
    rows = read_csv(boundaries_path)
    boundary_frames = np.asarray([int(row.get("boundary_frame") or row.get("frame")) for row in rows], dtype=np.int64)
    labels = np.zeros(len(centers), dtype=np.int64)
    for i, center in enumerate(centers):
        if len(boundary_frames) and np.min(np.abs(boundary_frames - center)) <= tolerance_frames:
            labels[i] = 1
    return np.asarray(["transition" if x else "steady" for x in labels], dtype=object)


def task_target(
    task: str,
    X: np.ndarray,
    windows: list[dict],
    manifest: list[dict],
    suite_dir: Path,
    future_offset_windows: int,
    object_targets: dict | None = None,
) -> dict:
    centers = frame_centers(windows)
    n = len(windows)
    train_idx, test_idx = chronological_split(n, 0.30)
    all_idx = np.arange(n, dtype=np.int64)
    if task == "timeline_action":
        return {"kind": "classification", "labels": labels_from_windows(windows, "action_label"), "rows": all_idx}
    if task == "timeline_subtask":
        return {"kind": "classification", "labels": labels_from_windows(windows, "subtask_label"), "rows": all_idx}
    if task == "transition_detection":
        return {"kind": "classification", "labels": transition_labels_from_boundaries(suite_dir, centers), "rows": all_idx}
    if task == "next_action":
        rows = np.arange(0, n - future_offset_windows, dtype=np.int64)
        labels = labels_from_windows(windows, "action_label")[rows + future_offset_windows]
        return {"kind": "classification", "labels": labels, "rows": rows, "target_variant": "future action label from windows.csv"}
    if task == "contact_prediction":
        contacts = block_indices(manifest, ["body_contacts"])
        rows = all_idx
        labels = np.where(np.abs(X[:, contacts]).sum(axis=1) > 1e-8, "contact", "no_contact")
        return {
            "kind": "classification",
            "labels": labels,
            "rows": rows,
            "target_source_blocks": "body_contacts",
            "target_variant": "contact proxy derived from body_contacts feature block",
        }
    if task == "hand_trajectory_forecast":
        rows = np.arange(0, n - future_offset_windows, dtype=np.int64)
        hand = block_indices(manifest, ["hand_left_joints", "hand_right_joints"])
        target = X[rows + future_offset_windows][:, hand]
        return {
            "kind": "regression",
            "target": target,
            "rows": rows,
            "target_source_blocks": "future hand_left_joints|hand_right_joints",
            "target_variant": "future hand feature vector from shared_windows.npz",
        }
    if task == "caption_grounding":
        rows = all_idx
        text = block_indices(manifest, ["caption_objects_interaction_text"])
        return {"kind": "retrieval", "target": X[:, text], "rows": rows, "target_source_blocks": "caption_objects_interaction_text"}
    if task in {"cross_modal_retrieval", "modality_reconstruction"}:
        rows = all_idx
        visual = block_indices(manifest, ["depth_confidence", "video_"])
        return {"kind": "retrieval" if task == "cross_modal_retrieval" else "regression", "target": X[:, visual], "rows": rows, "target_source_blocks": "depth_confidence|video_*"}
    if task == "temporal_order":
        pairs = []
        labels = []
        for i in range(n - 1):
            pairs.append((i, i + 1))
            labels.append("forward")
            pairs.append((i + 1, i))
            labels.append("reversed")
        return {"kind": "pair_classification", "pairs": np.asarray(pairs, dtype=np.int64), "labels": np.asarray(labels, dtype=object)}
    if task == "misalignment_detection":
        shift = 8
        pairs = []
        labels = []
        for i in range(n - shift):
            pairs.append((i, i))
            labels.append("aligned")
            pairs.append((i, i + shift))
            labels.append("shifted")
        return {"kind": "pair_classification", "pairs": np.asarray(pairs, dtype=np.int64), "labels": np.asarray(labels, dtype=object)}
    if task == "object_relevance":
        if object_targets is None:
            return {
                "kind": "not_available",
                "reason": "raw annotation.hdf5 was not provided, so full-train object relevance labels could not be exported",
            }
        rows = all_idx
        return {
            "kind": "multilabel",
            "target": object_targets["Y"],
            "rows": rows,
            "target_source_blocks": "caption_objects_interaction_text",
            "target_variant": "object sets exported from annotation.hdf5 caption_frame_info_map",
        }
    raise KeyError(task)


def target_overlap(group_idx: np.ndarray, target_info: dict, manifest: list[dict]) -> bool:
    blocks = str(target_info.get("target_source_blocks", ""))
    if not blocks:
        return False
    target_idx: list[int] = []
    for part in blocks.split("|"):
        part = part.strip()
        if not part:
            continue
        prefix = part[:-1] if part.endswith("*") else part
        target_idx.extend(block_indices(manifest, [prefix]).tolist())
    if not target_idx:
        return False
    return bool(np.intersect1d(group_idx, np.asarray(target_idx, dtype=np.int64)).size)


def pair_features(X: np.ndarray, pairs: np.ndarray, group_idx: np.ndarray, visual_idx: np.ndarray | None = None, task: str = "") -> np.ndarray:
    left = X[pairs[:, 0]][:, group_idx]
    right_source = X[pairs[:, 1]]
    if task == "misalignment_detection" and visual_idx is not None:
        right = right_source[:, visual_idx]
    else:
        right = right_source[:, group_idx]
    diff = right[:, : min(left.shape[1], right.shape[1])] - left[:, : min(left.shape[1], right.shape[1])]
    return np.concatenate([left, right, diff], axis=1).astype(np.float32)


def run_modality_ablation(
    X: np.ndarray,
    windows: list[dict],
    manifest: list[dict],
    suite_dir: Path,
    out_dir: Path,
    args: argparse.Namespace,
    object_targets: dict | None = None,
) -> list[dict]:
    groups = modality_groups(manifest)
    visual_idx = block_indices(manifest, ["depth_confidence", "video_"])
    contact_idx = block_indices(manifest, ["body_contacts"])
    rows: list[dict] = []

    for task in TASKS:
        info = task_target(task, X, windows, manifest, suite_dir, args.future_offset_windows, object_targets)
        for group_name, group_idx_raw in groups.items():
            row = {
                "task": task,
                "task_display": TASK_DISPLAY[task],
                "modality_group": group_name,
                "modality_display": GROUP_DISPLAY[group_name],
                "status": "computed",
                "score": "",
                "primary_metric": "",
                "primary_metric_value": "",
                "target_variant": info.get("target_variant", ""),
                "target_source_overlap": "",
                "reason": "",
            }
            if info["kind"] == "not_available":
                row.update({"status": "not_computed", "reason": info["reason"]})
                rows.append(row)
                continue

            group_idx = group_idx_raw
            if task == "contact_prediction":
                group_idx = np.setdiff1d(group_idx_raw, contact_idx)
                if len(group_idx) == 0:
                    row.update({"status": "not_computed", "reason": "input group would contain only contact target-source features"})
                    rows.append(row)
                    continue

            try:
                if info["kind"] == "classification":
                    data_rows = np.asarray(info["rows"], dtype=np.int64)
                    labels = np.asarray(info["labels"], dtype=object)
                    train_local, test_local = chronological_split(len(data_rows), args.test_fraction)
                    metrics, _ = fit_classification(
                        X[data_rows][:, group_idx],
                        labels,
                        train_local,
                        test_local,
                        args.ridge_l2,
                    )
                    if metrics.get("status") == "not_computed":
                        row.update(metrics)
                    else:
                        row.update(metrics)
                        row["primary_metric"] = "macro_f1"
                        row["primary_metric_value"] = metrics["macro_f1"]
                        row["score"] = metrics["macro_f1"]
                elif info["kind"] == "regression":
                    data_rows = np.asarray(info["rows"], dtype=np.int64)
                    target = np.asarray(info["target"], dtype=np.float32)
                    train_local, test_local = chronological_split(len(data_rows), args.test_fraction)
                    Xin_train, Xin_test = standardize(X[data_rows[train_local]][:, group_idx], X[data_rows[test_local]][:, group_idx])
                    Y_train, Y_test = standardize(info["target"][train_local], info["target"][test_local])
                    pred = ridge_predict(Xin_train, Y_train, Xin_test, args.ridge_l2)
                    metrics = regression_metrics(Y_test, pred)
                    row.update(metrics)
                    row["primary_metric"] = "mae"
                    row["primary_metric_value"] = metrics["mae"]
                    row["score"] = 1.0 / (1.0 + metrics["mae"])
                elif info["kind"] == "multilabel":
                    data_rows = np.asarray(info["rows"], dtype=np.int64)
                    target = np.asarray(info["target"], dtype=np.float32)
                    train_local, test_local = chronological_split(len(data_rows), args.test_fraction)
                    metrics, _ = fit_multilabel(X[data_rows][:, group_idx], target, train_local, test_local, args.ridge_l2)
                    row.update(metrics)
                    row["primary_metric"] = "micro_f1"
                    row["primary_metric_value"] = metrics["micro_f1"]
                    row["score"] = metrics["micro_f1"]
                elif info["kind"] == "retrieval":
                    data_rows = np.asarray(info["rows"], dtype=np.int64)
                    target = np.asarray(info["target"], dtype=np.float32)
                    train_local, test_local = chronological_split(len(data_rows), args.test_fraction)
                    Xin_train, Xin_test = standardize(X[data_rows[train_local]][:, group_idx], X[data_rows[test_local]][:, group_idx])
                    Y_train, Y_test = standardize(target[train_local], target[test_local])
                    pred = ridge_predict(Xin_train, Y_train, Xin_test, args.ridge_l2)
                    metrics = retrieval_metrics(pred, Y_test)
                    row.update(metrics)
                    row["primary_metric"] = "mrr"
                    row["primary_metric_value"] = metrics["mrr"]
                    row["score"] = metrics["mrr"]
                elif info["kind"] == "pair_classification":
                    pairs = np.asarray(info["pairs"], dtype=np.int64)
                    labels = np.asarray(info["labels"], dtype=object)
                    train_local, test_local = chronological_split(len(pairs), args.test_fraction)
                    feats = pair_features(X, pairs, group_idx, visual_idx=visual_idx, task=task)
                    metrics, _ = fit_classification(feats, labels, train_local, test_local, args.ridge_l2)
                    if metrics.get("status") == "not_computed":
                        row.update(metrics)
                    else:
                        row.update(metrics)
                        row["primary_metric"] = "macro_f1"
                        row["primary_metric_value"] = metrics["macro_f1"]
                        row["score"] = metrics["macro_f1"]
                else:
                    raise ValueError(info["kind"])
                row["target_source_overlap"] = str(target_overlap(group_idx, info, manifest)).lower()
            except Exception as exc:  # keep failed pairs visible instead of silently dropping them
                row.update({"status": "not_computed", "reason": f"{type(exc).__name__}: {exc}"})
            rows.append(row)

    write_csv(out_dir / "modality_ablation/ablation_metrics.csv", rows)
    write_json(
        out_dir / "modality_ablation/ablation_summary.json",
        {
            "description": "Compact ridge-head ablation over real shared_windows.npz feature blocks.",
            "num_rows": len(rows),
            "num_computed": sum(1 for r in rows if r.get("status") == "computed"),
            "num_not_computed": sum(1 for r in rows if r.get("status") != "computed"),
            "groups": {name: int(len(idx)) for name, idx in groups.items()},
            "tasks": TASKS,
            "object_relevance_labels": "annotation.hdf5" if object_targets is not None else "not_available",
        },
    )
    render_ablation_svg(rows, out_dir / "modality_ablation/ablation_matrix.svg")
    write_modality_report(rows, out_dir / "modality_ablation/MODALITY_ABLATION_REPORT.md")
    return rows


def color_for_score(score: float | None) -> str:
    if score is None or math.isnan(score):
        return "#20251f"
    score = max(0.0, min(1.0, score))
    r = int(37 + (154 - 37) * (1.0 - score))
    g = int(72 + (224 - 72) * score)
    b = int(54 + (101 - 54) * score)
    return f"#{r:02x}{g:02x}{b:02x}"


def render_ablation_svg(rows: list[dict], path: Path) -> None:
    groups = list(GROUP_DISPLAY.keys())
    tasks = TASKS
    cell_w, cell_h = 132, 34
    left, top = 300, 98
    width = left + cell_w * len(groups) + 44
    height = top + cell_h * len(tasks) + 86
    by_key = {(r["task"], r["modality_group"]): r for r in rows}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#10160f"/>',
        '<text x="28" y="38" fill="#eef5e8" font-family="Inter, Arial" font-size="24" font-weight="700">Single-Episode Modality Ablation Matrix</text>',
        '<text x="28" y="66" fill="#a7b5a3" font-family="Inter, Arial" font-size="13">Scores are recomputed from shared_windows.npz; gray cells are intentionally not computed.</text>',
    ]
    for j, group in enumerate(groups):
        x = left + j * cell_w + 6
        parts.append(
            f'<text x="{x}" y="{top - 18}" fill="#cbd8c8" font-family="Inter, Arial" font-size="12" transform="rotate(-25 {x} {top - 18})">{html.escape(GROUP_DISPLAY[group])}</text>'
        )
    for i, task in enumerate(tasks):
        y = top + i * cell_h
        parts.append(f'<text x="28" y="{y + 22}" fill="#e7efe2" font-family="Inter, Arial" font-size="13">{html.escape(TASK_DISPLAY[task])}</text>')
        for j, group in enumerate(groups):
            x = left + j * cell_w
            row = by_key.get((task, group), {})
            if row.get("status") == "computed" and row.get("score") != "":
                score = float(row["score"])
                fill = color_for_score(score)
                label = f'{score:.2f}'
                label_fill = "#061006" if score > 0.62 else "#edf5e7"
            else:
                fill = "#222720"
                label = "n/a"
                label_fill = "#7b8678"
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w - 8}" height="{cell_h - 7}" rx="5" fill="{fill}" stroke="#34402f" stroke-width="1"/>')
            parts.append(f'<text x="{x + (cell_w - 8) / 2}" y="{y + 19}" text-anchor="middle" fill="{label_fill}" font-family="Inter, Arial" font-size="12" font-weight="700">{label}</text>')
    parts.extend(
        [
            f'<text x="28" y="{height - 34}" fill="#a7b5a3" font-family="Inter, Arial" font-size="12">Metric: macro-F1 / MRR / 1/(1+MAE), depending on task type. See CSV for raw values and overlap flags.</text>',
            "</svg>",
        ]
    )
    write_text(path, "\n".join(parts))


def write_modality_report(rows: list[dict], path: Path) -> None:
    computed = [r for r in rows if r.get("status") == "computed" and r.get("score") != ""]
    by_task: dict[str, list[dict]] = {t: [] for t in TASKS}
    for row in computed:
        by_task[row["task"]].append(row)
    lines = [
        "# Single-Episode Modality Ablation Report",
        "",
        "This diagnostic reruns compact ridge heads on the exported one-episode feature matrix. It is useful for checking which real feature blocks can support each task on this episode, not for estimating dataset-wide generalization.",
        "",
        "No synthetic labels are introduced. Derived proxy targets are marked in `target_variant`, and feature groups that overlap with the target source are marked in `target_source_overlap`.",
        "",
        "## Best Computed Group Per Task",
        "",
    ]
    for task in TASKS:
        task_rows = by_task.get(task, [])
        if not task_rows:
            reasons = sorted({r.get("reason", "") for r in rows if r["task"] == task and r.get("reason")})
            lines.append(f"- {TASK_DISPLAY[task]}: not computed ({'; '.join(reasons)})")
            continue
        best = max(task_rows, key=lambda r: float(r["score"]))
        line = (
            f"- {TASK_DISPLAY[task]}: {best['modality_display']} score={float(best['score']):.4f}, "
            f"{best['primary_metric']}={float(best['primary_metric_value']):.4f}, target overlap={best['target_source_overlap']}"
        )
        if best["target_source_overlap"] == "true":
            no_overlap = [r for r in task_rows if r.get("target_source_overlap") == "false"]
            if no_overlap:
                alt = max(no_overlap, key=lambda r: float(r["score"]))
                line += (
                    f"; best non-overlap: {alt['modality_display']} score={float(alt['score']):.4f}, "
                    f"{alt['primary_metric']}={float(alt['primary_metric_value']):.4f}"
                )
        lines.append(line)
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `ablation_metrics.csv`: every task/modality pair, including not-computed rows and reasons.",
            "- `ablation_matrix.svg`: compact heatmap for manual inspection.",
            "- `ablation_summary.json`: group dimensions and computed/not-computed counts.",
        ]
    )
    write_text(path, "\n".join(lines) + "\n")


def run_timeline_overlay(suite_dir: Path, windows: list[dict], out_dir: Path) -> list[dict]:
    overlay_tasks = [
        "timeline_action",
        "timeline_subtask",
        "transition_detection",
        "next_action",
        "contact_prediction",
        "object_relevance",
    ]
    window_by_index = {int(row["window_index"]): row for row in windows}
    rows: list[dict] = []
    for task in overlay_tasks:
        pred_path = suite_dir / task / "predictions.csv"
        if not pred_path.exists():
            rows.append({"task": task, "status": "not_available", "reason": f"missing {pred_path}"})
            continue
        for pred in read_csv(pred_path):
            try:
                idx = int(pred["window_index"])
            except KeyError:
                continue
            window = window_by_index.get(idx, {})
            true_value = pred.get("true_label") or pred.get("true_objects") or pred.get("true") or ""
            pred_value = pred.get("predicted_label") or pred.get("predicted_objects") or pred.get("predicted") or ""
            if "correct" in pred and pred["correct"] != "":
                correct = int(float(pred["correct"]))
            else:
                correct = int(str(true_value) == str(pred_value))
            rows.append(
                {
                    "task": task,
                    "task_display": TASK_DISPLAY[task],
                    "status": "observed_prediction",
                    "window_index": idx,
                    "start_frame": pred.get("start_frame") or window.get("start_frame", ""),
                    "end_frame": pred.get("end_frame") or window.get("end_frame", ""),
                    "center_frame": pred.get("center_frame") or window.get("center_frame", ""),
                    "true_value": true_value,
                    "predicted_value": pred_value,
                    "confidence": pred.get("confidence", ""),
                    "correct": correct,
                }
            )

    write_csv(out_dir / "timeline_overlay/timeline_overlay.csv", rows)
    render_timeline_svg(rows, windows, suite_dir, out_dir / "timeline_overlay/timeline_overlay.svg")
    write_timeline_report(rows, out_dir / "timeline_overlay/TIMELINE_OVERLAY_REPORT.md")
    return rows


def render_timeline_svg(rows: list[dict], windows: list[dict], suite_dir: Path, path: Path) -> None:
    tasks = ["timeline_action", "timeline_subtask", "transition_detection", "next_action", "contact_prediction", "object_relevance"]
    min_frame = min(int(r["start_frame"]) for r in windows)
    max_frame = max(int(r["end_frame"]) for r in windows)
    left, top = 260, 84
    row_h, plot_w = 48, 1100
    width = left + plot_w + 38
    height = top + row_h * len(tasks) + 92
    by_task: dict[str, list[dict]] = {t: [] for t in tasks}
    for row in rows:
        if row.get("status") == "observed_prediction":
            by_task[row["task"]].append(row)

    def x_for(frame: int) -> float:
        return left + (frame - min_frame) / max(1, max_frame - min_frame) * plot_w

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#10160f"/>',
        '<text x="28" y="38" fill="#eef5e8" font-family="Inter, Arial" font-size="24" font-weight="700">Held-Out Timeline Prediction Overlay</text>',
        '<text x="28" y="64" fill="#a7b5a3" font-family="Inter, Arial" font-size="13">Bars are existing real prediction rows aligned back to the exported episode timeline.</text>',
    ]
    boundaries_path = suite_dir / "transition_detection/true_boundaries.csv"
    boundary_frames = []
    if boundaries_path.exists():
        boundary_frames = [int(r.get("boundary_frame") or r.get("frame")) for r in read_csv(boundaries_path)]
    for i, task in enumerate(tasks):
        y = top + i * row_h
        parts.append(f'<text x="28" y="{y + 25}" fill="#e7efe2" font-family="Inter, Arial" font-size="13">{html.escape(TASK_DISPLAY[task])}</text>')
        parts.append(f'<line x1="{left}" y1="{y + 20}" x2="{left + plot_w}" y2="{y + 20}" stroke="#2a3428" stroke-width="18" stroke-linecap="round"/>')
        for row in by_task[task]:
            if not row.get("start_frame") or not row.get("end_frame"):
                continue
            x1 = x_for(int(float(row["start_frame"])))
            x2 = max(x1 + 2, x_for(int(float(row["end_frame"]))))
            fill = "#8ee06a" if int(row["correct"]) else "#e46b5f"
            parts.append(f'<rect x="{x1:.2f}" y="{y + 11}" width="{x2 - x1:.2f}" height="18" rx="3" fill="{fill}" opacity="0.86"/>')
        for frame in boundary_frames:
            x = x_for(frame)
            parts.append(f'<line x1="{x:.2f}" y1="{y + 4}" x2="{x:.2f}" y2="{y + 36}" stroke="#d8e887" stroke-width="1.2" opacity="0.70"/>')
    parts.extend(
        [
            f'<text x="{left}" y="{height - 42}" fill="#8ee06a" font-family="Inter, Arial" font-size="12">green = exact/correct prediction</text>',
            f'<text x="{left + 220}" y="{height - 42}" fill="#e46b5f" font-family="Inter, Arial" font-size="12">red = mismatch</text>',
            f'<text x="{left + 390}" y="{height - 42}" fill="#d8e887" font-family="Inter, Arial" font-size="12">vertical lines = real transition boundaries</text>',
            "</svg>",
        ]
    )
    write_text(path, "\n".join(parts))


def write_timeline_report(rows: list[dict], path: Path) -> None:
    observed = [r for r in rows if r.get("status") == "observed_prediction"]
    lines = [
        "# Timeline Prediction Overlay Report",
        "",
        "This report aligns existing prediction CSV files to the exported episode timeline. It does not rerun training.",
        "",
        "## Task-Level Correctness",
        "",
    ]
    for task in ["timeline_action", "timeline_subtask", "transition_detection", "next_action", "contact_prediction", "object_relevance"]:
        task_rows = [r for r in observed if r["task"] == task]
        if not task_rows:
            lines.append(f"- {TASK_DISPLAY[task]}: no prediction rows found")
            continue
        correct = sum(int(r["correct"]) for r in task_rows)
        lines.append(f"- {TASK_DISPLAY[task]}: {correct}/{len(task_rows)} correct ({correct / len(task_rows):.4f})")
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `timeline_overlay.csv`: prediction rows with frame positions.",
            "- `timeline_overlay.svg`: visual overlay across the episode.",
        ]
    )
    write_text(path, "\n".join(lines) + "\n")


def run_alignment_stress(
    X: np.ndarray,
    manifest: list[dict],
    windows: list[dict],
    out_dir: Path,
    args: argparse.Namespace,
) -> list[dict]:
    groups = modality_groups(manifest)
    stress_groups = {
        "motion_capture": groups["motion_capture"],
        "pose_slam": groups["pose_slam"],
        "inertial": groups["inertial"],
        "language": groups["language"],
        "motion_pose_inertial": np.unique(np.concatenate([groups["motion_capture"], groups["pose_slam"], groups["inertial"]])),
    }
    target_idx = block_indices(manifest, ["depth_confidence", "video_"])
    n = X.shape[0]
    train_idx, test_idx = chronological_split(n, args.test_fraction)
    shifts = [-40, -20, -10, -5, 0, 5, 10, 20, 40]
    rows: list[dict] = []
    stride = int(windows[1]["start_frame"]) - int(windows[0]["start_frame"]) if len(windows) > 1 else 1
    for group, q_idx in stress_groups.items():
        q_train, q_test_all = standardize(X[train_idx][:, q_idx], X[test_idx][:, q_idx])
        t_train, t_test_all = standardize(X[train_idx][:, target_idx], X[test_idx][:, target_idx])
        projector_pred_all = ridge_predict(q_train, t_train, q_test_all, args.ridge_l2)
        for shift in shifts:
            valid = []
            for local_i in range(len(test_idx)):
                shifted_local = local_i + shift
                if 0 <= shifted_local < len(test_idx):
                    valid.append((local_i, shifted_local))
            if not valid:
                continue
            original = np.asarray([a for a, _ in valid], dtype=np.int64)
            shifted = np.asarray([b for _, b in valid], dtype=np.int64)
            pred = projector_pred_all[shifted]
            target = t_test_all[original]
            metrics = retrieval_metrics(pred, target)
            row = {
                "query_group": group,
                "query_display": GROUP_DISPLAY[group],
                "target_group": "depth_plus_video",
                "shift_windows": shift,
                "shift_frames": int(shift * stride),
                "status": "derived_perturbation",
                **metrics,
            }
            rows.append(row)
    write_csv(out_dir / "alignment_stress/alignment_shift_metrics.csv", rows)
    render_alignment_svg(rows, out_dir / "alignment_stress/alignment_shift_curves.svg")
    write_json(
        out_dir / "alignment_stress/alignment_stress_summary.json",
        {
            "description": "Real feature windows are deliberately time-shifted at evaluation time to test cross-modal alignment sensitivity.",
            "target_group": "depth_confidence + video_*",
            "status_meaning": "derived_perturbation means the features are real but the time shift is an explicit diagnostic perturbation.",
            "num_rows": len(rows),
        },
    )
    write_alignment_report(rows, out_dir / "alignment_stress/ALIGNMENT_STRESS_REPORT.md")
    return rows


def render_alignment_svg(rows: list[dict], path: Path) -> None:
    groups = sorted({r["query_group"] for r in rows})
    width, height = 1020, 520
    left, top, plot_w, plot_h = 94, 80, 810, 330
    shifts = sorted({int(r["shift_windows"]) for r in rows})
    if not shifts:
        write_text(path, "<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>")
        return
    min_shift, max_shift = min(shifts), max(shifts)
    max_mrr = max(float(r["mrr"]) for r in rows) if rows else 1.0
    max_mrr = max(max_mrr, 1e-6)
    palette = ["#8ee06a", "#d8e887", "#7bd3ff", "#f0a45e", "#cba8ff"]

    def x_for(shift: int) -> float:
        return left + (shift - min_shift) / max(1, max_shift - min_shift) * plot_w

    def y_for(mrr: float) -> float:
        return top + plot_h - mrr / max_mrr * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#10160f"/>',
        '<text x="28" y="38" fill="#eef5e8" font-family="Inter, Arial" font-size="24" font-weight="700">Cross-Modal Alignment Stress Test</text>',
        '<text x="28" y="64" fill="#a7b5a3" font-family="Inter, Arial" font-size="13">Query features are shifted in time; the target visual window remains the original held-out window.</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#151d14" stroke="#34402f"/>',
    ]
    for tick in shifts:
        x = x_for(tick)
        parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="#263024" stroke-width="1"/>')
        parts.append(f'<text x="{x:.2f}" y="{top + plot_h + 24}" fill="#a7b5a3" font-family="Inter, Arial" font-size="11" text-anchor="middle">{tick}</text>')
    parts.append(f'<text x="{left + plot_w / 2}" y="{height - 48}" fill="#cbd8c8" font-family="Inter, Arial" font-size="13" text-anchor="middle">shift in windows</text>')
    parts.append(f'<text x="30" y="{top + plot_h / 2}" fill="#cbd8c8" font-family="Inter, Arial" font-size="13" transform="rotate(-90 30 {top + plot_h / 2})">MRR</text>')

    for gi, group in enumerate(groups):
        color = palette[gi % len(palette)]
        group_rows = sorted([r for r in rows if r["query_group"] == group], key=lambda r: int(r["shift_windows"]))
        points = [(x_for(int(r["shift_windows"])), y_for(float(r["mrr"]))) for r in group_rows]
        if points:
            d = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
            parts.append(f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="2.4"/>')
            for x, y in points:
                parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/>')
            parts.append(f'<rect x="{left + plot_w + 28}" y="{top + gi * 25}" width="12" height="12" fill="{color}"/>')
            parts.append(f'<text x="{left + plot_w + 46}" y="{top + 11 + gi * 25}" fill="#e7efe2" font-family="Inter, Arial" font-size="12">{html.escape(GROUP_DISPLAY.get(group, group))}</text>')
    parts.append("</svg>")
    write_text(path, "\n".join(parts))


def write_alignment_report(rows: list[dict], path: Path) -> None:
    lines = [
        "# Cross-Modal Alignment Stress Report",
        "",
        "This diagnostic uses real held-out feature windows, then deliberately shifts the query modality in time at evaluation. The perturbation is derived; it is not treated as observed data.",
        "",
        "## Zero-Shift Versus Worst Shift",
        "",
    ]
    for group in sorted({r["query_group"] for r in rows}):
        group_rows = [r for r in rows if r["query_group"] == group]
        zero = next((r for r in group_rows if int(r["shift_windows"]) == 0), None)
        worst = min(group_rows, key=lambda r: float(r["mrr"])) if group_rows else None
        if zero and worst:
            lines.append(
                f"- {GROUP_DISPLAY.get(group, group)}: zero-shift MRR={float(zero['mrr']):.4f}; "
                f"worst shift={worst['shift_windows']} windows, MRR={float(worst['mrr']):.4f}"
            )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `alignment_shift_metrics.csv`: MRR/rank metrics for each query group and time shift.",
            "- `alignment_shift_curves.svg`: MRR curves across time shifts.",
            "- `alignment_stress_summary.json`: perturbation definition and status.",
        ]
    )
    write_text(path, "\n".join(lines) + "\n")


def build_provenance(
    suite_dir: Path,
    out_dir: Path,
    X: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    manifest: list[dict],
    summary: dict,
    annotation: Path | None = None,
) -> dict:
    repo_root = suite_dir.parent.parent.resolve()
    input_files = [
        suite_dir / "shared_windows.npz",
        suite_dir / "windows.csv",
        suite_dir / "feature_manifest.json",
        suite_dir / "summary_report.json",
        suite_dir / "transition_detection/true_boundaries.csv",
    ]
    for task in ["timeline_action", "timeline_subtask", "transition_detection", "next_action", "contact_prediction", "object_relevance"]:
        pred_path = suite_dir / task / "predictions.csv"
        if pred_path.exists():
            input_files.append(pred_path)
    if annotation is not None and annotation.exists():
        input_files.append(annotation)
    provenance = {
        "artifact_policy": "Only existing local artifacts are consumed. Missing labels/tasks are marked not_computed instead of filled.",
        "source_suite_dir": public_artifact_path(suite_dir, repo_root),
        "output_dir": public_artifact_path(out_dir, repo_root),
        "shared_windows_shape": [int(X.shape[0]), int(X.shape[1])],
        "starts_first_last": [int(starts[0]), int(starts[-1])],
        "ends_first_last": [int(ends[0]), int(ends[-1])],
        "feature_blocks": manifest,
        "summary_report_core": {
            "num_windows": summary.get("num_windows"),
            "feature_dim": summary.get("feature_dim"),
            "window_frames": summary.get("window_frames"),
            "stride_frames": summary.get("stride_frames"),
            "annotation": public_source_reference(summary.get("annotation")),
        },
        "input_file_hashes": {
            public_artifact_path(path, repo_root): sha256(path)
            for path in input_files
            if path.exists()
        },
    }
    write_json(out_dir / "provenance.json", provenance)
    return provenance


def write_index(out_dir: Path, provenance: dict, ablation_rows: list[dict], timeline_rows: list[dict], stress_rows: list[dict]) -> None:
    lines = [
        "# Single-Episode Diagnostics Index",
        "",
        "These outputs are local diagnostics built from the existing one-episode Xperience-10M artifacts. They are designed for manual verification while waiting for full multi-episode data access.",
        "",
        "## Generated Analyses",
        "",
        "- `modality_ablation/`: compact ridge-head ablations across real feature blocks.",
        "- `timeline_overlay/`: existing prediction CSVs aligned to the episode timeline.",
        "- `alignment_stress/`: cross-modal retrieval under explicit time-shift perturbations.",
        "- `provenance.json`: input hashes, feature dimensions, and source artifact identifiers.",
        "",
        "## Validity Boundaries",
        "",
        "- This is a single-episode diagnostic, not a full Xperience-10M benchmark.",
        "- Rows marked `not_computed` are intentionally left blank when train labels or valid splits are unavailable.",
        "- Rows marked `derived_perturbation` use real features with deliberate time shifts for stress testing.",
        "",
        "## Counts",
        "",
        f"- Ablation rows: {len(ablation_rows)}; computed: {sum(1 for r in ablation_rows if r.get('status') == 'computed')}.",
        f"- Timeline overlay rows: {sum(1 for r in timeline_rows if r.get('status') == 'observed_prediction')}.",
        f"- Alignment stress rows: {len(stress_rows)}.",
        f"- Shared feature shape: {provenance['shared_windows_shape'][0]} windows x {provenance['shared_windows_shape'][1]} features.",
    ]
    write_text(out_dir / "README.md", "\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    suite_dir = args.suite_dir.resolve()
    out_dir = args.output_dir.resolve()
    X, starts, ends, windows, manifest, summary = load_inputs(suite_dir)
    object_targets = None
    if args.annotation is not None:
        object_targets = load_object_targets_from_annotation(args.annotation, windows, args.homie_toolkit)
        write_csv(
            out_dir / "object_labels/window_object_labels.csv",
            object_targets["rows"],
            ["window_index", "start_frame", "end_frame", "center_frame", "objects", "object_count"],
        )
        write_json(
            out_dir / "object_labels/object_vocab.json",
            {
                "vocab": object_targets["vocab"],
                "num_objects": len(object_targets["vocab"]),
                "source_annotation": object_targets["annotation"],
                "source_toolkit": object_targets["toolkit"],
                "source_note": object_targets["source_note"],
            },
        )
    provenance = build_provenance(suite_dir, out_dir, X, starts, ends, manifest, summary, args.annotation)
    ablation_rows = run_modality_ablation(X, windows, manifest, suite_dir, out_dir, args, object_targets)
    timeline_rows = run_timeline_overlay(suite_dir, windows, out_dir)
    stress_rows = run_alignment_stress(X, manifest, windows, out_dir, args)
    write_index(out_dir, provenance, ablation_rows, timeline_rows, stress_rows)
    print(f"Wrote diagnostics to {out_dir}")
    print(f"Ablation computed rows: {sum(1 for r in ablation_rows if r.get('status') == 'computed')}/{len(ablation_rows)}")
    print(f"Timeline observed rows: {sum(1 for r in timeline_rows if r.get('status') == 'observed_prediction')}")
    print(f"Alignment stress rows: {len(stress_rows)}")


if __name__ == "__main__":
    main()

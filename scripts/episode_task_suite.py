#!/usr/bin/env python3
"""
End-to-end task suite for one Xperience-10M episode released by Ropedia.

The purpose is not to estimate generalization from one sample episode. It is to
turn the episode into multiple meaningful supervised/self-supervised learning
problems and write reproducible artifacts for each one.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np

from train_all_modalities_model import (
    extract_all_window_features,
    prepare_modalities,
)
from train_min_action_model import (
    add_toolkit_to_path,
    compute_metrics,
    encode_labels,
    fit_scaler,
    frame_label,
    majority_label,
    predict,
    portable_path,
    softmax,
    train_softmax_classifier,
)


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


def parse_args() -> argparse.Namespace:
    workspace_default = Path(__file__).resolve().parents[1]
    annotation_default = workspace_default / "data/sample/xperience-10m-sample/annotation.hdf5"
    parser = argparse.ArgumentParser(description="Run an end-to-end task suite on one Xperience-10M episode.")
    parser.add_argument("--workspace", type=Path, default=workspace_default)
    parser.add_argument("--annotation", type=Path, default=annotation_default)
    parser.add_argument("--output-dir", type=Path, default=workspace_default / "outputs/episode_task_suite")
    parser.add_argument("--cache-dir", type=Path, default=workspace_default / "outputs/feature_cache")
    parser.add_argument("--window-frames", type=int, default=20)
    parser.add_argument("--stride-frames", type=int, default=5)
    parser.add_argument("--min-label-fraction", type=float, default=0.6)
    parser.add_argument("--test-fraction", type=float, default=0.30)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.12)
    parser.add_argument("--l2", type=float, default=2e-3)
    parser.add_argument("--ridge-l2", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--future-frames", type=int, default=20, help="Future offset for next-action prediction.")
    parser.add_argument("--forecast-frames", type=int, default=10, help="Future hand trajectory length.")
    parser.add_argument("--boundary-tolerance-frames", type=int, default=10)
    parser.add_argument("--misalignment-shift-windows", type=int, default=8)
    parser.add_argument("--tasks", default="all", help="Comma-separated task list or 'all'.")

    # Match train_all_modalities_model defaults used by prepare_modalities.
    parser.add_argument("--force-rebuild-cache", action="store_true")
    parser.add_argument("--video-image-size", type=int, default=32)
    parser.add_argument("--video-grid-size", type=int, default=8)
    parser.add_argument("--video-hist-bins", type=int, default=8)
    parser.add_argument("--depth-grid-size", type=int, default=8)
    parser.add_argument("--text-hash-dim", type=int, default=128)
    parser.add_argument("--include-label-text", action="store_true")
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--include-neural", action="store_true", help="Also run lightweight PyTorch MLP baselines for selected tasks.")
    parser.add_argument("--neural-output-name", default="neural_mlp", help="Subdirectory under --output-dir for neural task artifacts.")
    parser.add_argument("--neural-epochs", type=int, default=80)
    parser.add_argument("--neural-learning-rate", type=float, default=1e-3)
    parser.add_argument("--neural-weight-decay", type=float, default=1e-4)
    parser.add_argument("--neural-hidden-dim", type=int, default=128)
    parser.add_argument("--neural-batch-size", type=int, default=128)
    parser.add_argument("--neural-dropout", type=float, default=0.10)
    parser.add_argument("--neural-device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def selected_tasks(spec: str) -> list[str]:
    if spec.strip().lower() == "all":
        return TASKS
    chosen = [x.strip() for x in spec.split(",") if x.strip()]
    unknown = [x for x in chosen if x not in TASKS]
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}. Valid tasks: {TASKS}")
    return chosen


def write_json(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_confusion(path: Path, cm: np.ndarray, class_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["true\\pred"] + class_names)
        for i, name in enumerate(class_names):
            writer.writerow([name] + [int(v) for v in cm[i]])


def chronological_split_indices(n: int, test_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    if n < 2:
        raise ValueError("Need at least two samples for train/test split.")
    split = int(round(n * (1.0 - test_fraction)))
    split = max(1, min(split, n - 1))
    return np.arange(split, dtype=np.int64), np.arange(split, n, dtype=np.int64)


def build_windows(args: argparse.Namespace, ann: dict, extras: dict):
    frame_info = ann["caption_frame_info_map"]
    n_frames = len(ann["img_names"])
    rows = []
    X = []
    feature_manifest = None

    for start in range(0, n_frames - args.window_frames + 1, args.stride_frames):
        end = start + args.window_frames
        action_labels = [frame_label(frame_info.get(i, {}), "action") for i in range(start, end)]
        subtask_labels = [frame_label(frame_info.get(i, {}), "subtask") for i in range(start, end)]
        action, action_frac = majority_label(action_labels, args.min_label_fraction)
        subtask, subtask_frac = majority_label(subtask_labels, args.min_label_fraction)

        if feature_manifest is None:
            vec, blocks = extract_all_window_features(ann, extras, start, end, return_blocks=True)
            offset = 0
            feature_manifest = []
            for name, dim in blocks:
                feature_manifest.append({"name": name, "start": offset, "end": offset + dim, "dim": dim})
                offset += dim
        else:
            vec = extract_all_window_features(ann, extras, start, end)

        X.append(vec)
        rows.append({
            "window_index": len(rows),
            "start_frame": start,
            "end_frame": end - 1,
            "center_frame": (start + end - 1) // 2,
            "action_label": action,
            "action_fraction": action_frac,
            "subtask_label": subtask,
            "subtask_fraction": subtask_frac,
        })

    return np.stack(X).astype(np.float32), rows, feature_manifest or []


def block_indices(feature_manifest: list[dict], include: list[str] | None = None, exclude: list[str] | None = None) -> np.ndarray:
    include = include or []
    exclude = exclude or []
    idxs = []
    for block in feature_manifest:
        name = block["name"]
        if include and not any(name == p or name.startswith(p) for p in include):
            continue
        if exclude and any(name == p or name.startswith(p) for p in exclude):
            continue
        idxs.extend(range(int(block["start"]), int(block["end"])))
    return np.asarray(idxs, dtype=np.int64)


def label_array(rows: list[dict], key: str) -> np.ndarray:
    return np.asarray([str(row.get(key, "") or "") for row in rows], dtype=object)


def classification_task(
    out_dir: Path,
    X: np.ndarray,
    labels: np.ndarray,
    rows: list[dict],
    args: argparse.Namespace,
    task_name: str,
    input_description: str,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    valid = np.asarray([bool(x) for x in labels])
    valid_idx = np.flatnonzero(valid)
    Xv = X[valid_idx]
    labelv = labels[valid_idx]
    rowv = [rows[int(i)] for i in valid_idx]
    y, class_names = encode_labels(labelv)
    train_local, test_local = chronological_split_indices(len(y), args.test_fraction)

    train_classes = set(int(x) for x in y[train_local])
    test_classes = set(int(x) for x in y[test_local])
    unseen_test_classes = sorted(class_names[i] for i in (test_classes - train_classes))

    mean, std = fit_scaler(Xv[train_local])
    Xs = (Xv - mean) / std
    W, b, history = train_softmax_classifier(
        Xs[train_local],
        y[train_local],
        n_classes=len(class_names),
        epochs=args.epochs,
        lr=args.learning_rate,
        l2=args.l2,
        use_class_weights=not args.no_class_weights,
        seed=args.seed,
    )
    pred, probs = predict(Xs[test_local], W, b)
    metrics, per_class, cm = compute_metrics(y[test_local], pred, class_names)
    majority = Counter(y[train_local]).most_common(1)[0][0]
    metrics.update({
        "task": task_name,
        "input": input_description,
        "split": "chronological",
        "num_windows": int(len(y)),
        "num_train_windows": int(len(train_local)),
        "num_test_windows": int(len(test_local)),
        "num_classes": int(len(class_names)),
        "feature_dim": int(X.shape[1]),
        "majority_baseline_accuracy": float(np.mean(y[test_local] == majority)),
        "train_final_accuracy": float(history[-1]["train_accuracy"]),
        "train_final_loss": float(history[-1]["loss"]),
        "unseen_test_classes": unseen_test_classes,
    })

    pred_rows = []
    for local_pos, pred_id in zip(test_local, pred):
        row = rowv[int(local_pos)]
        true_id = int(y[int(local_pos)])
        pred_rows.append({
            "window_index": row["window_index"],
            "start_frame": row["start_frame"],
            "end_frame": row["end_frame"],
            "center_frame": row["center_frame"],
            "true_label": class_names[true_id],
            "predicted_label": class_names[int(pred_id)],
            "confidence": float(probs[list(test_local).index(local_pos), int(pred_id)]),
            "correct": int(true_id == int(pred_id)),
        })

    write_json(out_dir / "metrics.json", metrics)
    write_csv(out_dir / "per_class_metrics.csv", per_class, ["class_id", "class_name", "support", "predicted", "precision", "recall", "f1"])
    write_confusion(out_dir / "confusion_matrix.csv", cm, class_names)
    write_csv(out_dir / "predictions.csv", pred_rows, ["window_index", "start_frame", "end_frame", "center_frame", "true_label", "predicted_label", "confidence", "correct"])
    np.savez_compressed(out_dir / "model.npz", mean=mean, std=std, W=W, b=b, class_names=np.asarray(class_names, dtype=object))
    return metrics


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": float((tp + tn) / max(len(y_true), 1)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "positive_rate_true": float(np.mean(y_true)) if len(y_true) else 0.0,
        "positive_rate_pred": float(np.mean(y_pred)) if len(y_pred) else 0.0,
    }


def boundary_f1(true_frames: list[int], pred_frames: list[int], tolerance: int) -> dict:
    used = set()
    matches = 0
    errors = []
    for pf in pred_frames:
        candidates = [(abs(pf - tf), j, tf) for j, tf in enumerate(true_frames) if j not in used and abs(pf - tf) <= tolerance]
        if not candidates:
            continue
        diff, j, tf = min(candidates)
        used.add(j)
        matches += 1
        errors.append(diff)
    precision = matches / len(pred_frames) if pred_frames else 0.0
    recall = matches / len(true_frames) if true_frames else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": f1,
        "matched_boundaries": matches,
        "true_boundaries": len(true_frames),
        "predicted_boundaries": len(pred_frames),
        "mean_abs_timing_error_frames": float(np.mean(errors)) if errors else None,
    }


def task_transition_detection(out_dir: Path, X: np.ndarray, rows: list[dict], ann: dict, args: argparse.Namespace) -> dict:
    frame_info = ann["caption_frame_info_map"]
    n_frames = len(ann["img_names"])
    per_frame = [frame_label(frame_info.get(i, {}), "action") for i in range(n_frames)]
    true_boundaries = [i for i in range(1, n_frames) if per_frame[i] and per_frame[i - 1] and per_frame[i] != per_frame[i - 1]]

    y = []
    for row in rows:
        c = int(row["center_frame"])
        y.append(int(any(abs(c - b) <= args.boundary_tolerance_frames for b in true_boundaries)))
    labels = np.asarray(["transition" if v else "steady" for v in y], dtype=object)
    metrics = classification_task(out_dir, X, labels, rows, args, "transition_detection", "all modalities -> action boundary/steady")

    pred_path = out_dir / "predictions.csv"
    pred_rows = []
    with pred_path.open("r", encoding="utf-8") as fp:
        for row in csv.DictReader(fp):
            pred_rows.append(row)
    pred_frames = [int(r["center_frame"]) for r in pred_rows if r["predicted_label"] == "transition"]
    test_start = min((int(r["center_frame"]) for r in pred_rows), default=0)
    test_end = max((int(r["center_frame"]) for r in pred_rows), default=0)
    true_test = [b for b in true_boundaries if test_start <= b <= test_end]
    metrics.update(boundary_f1(true_test, pred_frames, args.boundary_tolerance_frames))
    write_json(out_dir / "metrics.json", metrics)
    write_csv(out_dir / "true_boundaries.csv", [{"frame": x} for x in true_boundaries], ["frame"])
    return metrics


def task_next_action(out_dir: Path, X: np.ndarray, rows: list[dict], ann: dict, args: argparse.Namespace) -> dict:
    frame_info = ann["caption_frame_info_map"]
    labels = []
    for row in rows:
        future_frame = min(len(ann["img_names"]) - 1, int(row["end_frame"]) + args.future_frames)
        labels.append(frame_label(frame_info.get(future_frame, {}), "action"))
    return classification_task(out_dir, X, np.asarray(labels, dtype=object), rows, args, "next_action", f"all modalities at t -> action at t+{args.future_frames} frames")


def ridge_fit_predict(X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray, l2: float):
    x_mean, x_std = fit_scaler(X_train)
    y_mean = Y_train.mean(axis=0)
    y_std = Y_train.std(axis=0)
    y_std = np.where(y_std < 1e-6, 1.0, y_std)
    Xtr = (X_train - x_mean) / x_std
    Xte = (X_test - x_mean) / x_std
    Ytr = (Y_train - y_mean) / y_std
    Xtr_aug = np.concatenate([Xtr, np.ones((len(Xtr), 1), dtype=np.float32)], axis=1)
    Xte_aug = np.concatenate([Xte, np.ones((len(Xte), 1), dtype=np.float32)], axis=1)
    K = Xtr_aug @ Xtr_aug.T
    alpha = np.linalg.solve(K + l2 * np.eye(K.shape[0], dtype=np.float32), Ytr)
    W = Xtr_aug.T @ alpha
    pred = (Xte_aug @ W) * y_std + y_mean
    return pred.astype(np.float32), {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean.astype(np.float32), "y_std": y_std.astype(np.float32), "W": W.astype(np.float32)}


def regression_metrics(Y_true: np.ndarray, Y_pred: np.ndarray) -> dict:
    mse = float(np.mean((Y_true - Y_pred) ** 2))
    mae = float(np.mean(np.abs(Y_true - Y_pred)))
    ss_res = float(np.sum((Y_true - Y_pred) ** 2))
    ss_tot = float(np.sum((Y_true - Y_true.mean(axis=0)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"mse": mse, "mae": mae, "r2": r2}


def neural_config(args: argparse.Namespace):
    from neural_task_models import NeuralConfig

    return NeuralConfig(
        epochs=args.neural_epochs,
        learning_rate=args.neural_learning_rate,
        weight_decay=args.neural_weight_decay,
        hidden_dim=args.neural_hidden_dim,
        batch_size=args.neural_batch_size,
        dropout=args.neural_dropout,
        device=args.neural_device,
        seed=args.seed,
    )


def neural_common_metrics(args: argparse.Namespace, result: dict, head: str) -> dict:
    final = result["history"][-1] if result.get("history") else {}
    metrics = {
        "model": "neural_mlp",
        "head": head,
        "neural_epochs": int(args.neural_epochs),
        "neural_hidden_dim": int(args.neural_hidden_dim),
        "neural_batch_size": int(args.neural_batch_size),
        "neural_learning_rate": float(args.neural_learning_rate),
        "neural_weight_decay": float(args.neural_weight_decay),
        "neural_dropout": float(args.neural_dropout),
        "neural_device": result.get("device", args.neural_device),
    }
    if "loss" in final:
        metrics["train_final_loss"] = float(final["loss"])
    if "train_accuracy" in final:
        metrics["train_final_accuracy"] = float(final["train_accuracy"])
    return metrics


def save_neural_model(out_dir: Path, result: dict, model_type: str, extra: dict | None = None) -> None:
    from neural_task_models import save_torch_model

    payload = {
        "model_type": model_type,
        "state_dict": result["state_dict"],
        "scaler": {k: result[k] for k in ("mean", "std", "x_mean", "x_std", "y_mean", "y_std") if k in result},
        "config": extra or {},
    }
    save_torch_model(out_dir / "model.pt", payload)


def neural_classification_task(
    out_dir: Path,
    X: np.ndarray,
    labels: np.ndarray,
    rows: list[dict],
    args: argparse.Namespace,
    task_name: str,
    input_description: str,
) -> dict:
    from neural_task_models import train_classifier

    out_dir.mkdir(parents=True, exist_ok=True)
    valid = np.asarray([bool(x) for x in labels])
    valid_idx = np.flatnonzero(valid)
    Xv = X[valid_idx]
    labelv = labels[valid_idx]
    rowv = [rows[int(i)] for i in valid_idx]
    y, class_names = encode_labels(labelv)
    train_local, test_local = chronological_split_indices(len(y), args.test_fraction)
    train_classes = set(int(x) for x in y[train_local])
    test_classes = set(int(x) for x in y[test_local])
    unseen_test_classes = sorted(class_names[i] for i in (test_classes - train_classes))

    result = train_classifier(
        Xv,
        y,
        train_local,
        test_local,
        n_classes=len(class_names),
        config=neural_config(args),
        use_class_weights=not args.no_class_weights,
    )
    pred = result["pred"]
    probs = result["prob"]
    metrics, per_class, cm = compute_metrics(y[test_local], pred, class_names)
    majority = Counter(y[train_local]).most_common(1)[0][0]
    metrics.update({
        "task": task_name,
        "input": input_description,
        "split": "chronological",
        "num_windows": int(len(y)),
        "num_train_windows": int(len(train_local)),
        "num_test_windows": int(len(test_local)),
        "num_classes": int(len(class_names)),
        "feature_dim": int(X.shape[1]),
        "majority_baseline_accuracy": float(np.mean(y[test_local] == majority)),
        "unseen_test_classes": unseen_test_classes,
    })
    metrics.update(neural_common_metrics(args, result, "z-score -> MLP softmax"))

    pred_rows = []
    for k, (local_pos, pred_id) in enumerate(zip(test_local, pred)):
        row = rowv[int(local_pos)]
        true_id = int(y[int(local_pos)])
        pred_rows.append({
            "window_index": row["window_index"],
            "start_frame": row["start_frame"],
            "end_frame": row["end_frame"],
            "center_frame": row["center_frame"],
            "true_label": class_names[true_id],
            "predicted_label": class_names[int(pred_id)],
            "confidence": float(probs[k, int(pred_id)]),
            "correct": int(true_id == int(pred_id)),
        })

    write_json(out_dir / "metrics.json", metrics)
    write_json(out_dir / "history.json", result["history"])
    write_csv(out_dir / "per_class_metrics.csv", per_class, ["class_id", "class_name", "support", "predicted", "precision", "recall", "f1"])
    write_confusion(out_dir / "confusion_matrix.csv", cm, class_names)
    write_csv(out_dir / "predictions.csv", pred_rows, ["window_index", "start_frame", "end_frame", "center_frame", "true_label", "predicted_label", "confidence", "correct"])
    save_neural_model(out_dir, result, "classifier", {"class_names": class_names})
    return metrics


def neural_transition_detection(out_dir: Path, X: np.ndarray, rows: list[dict], ann: dict, args: argparse.Namespace) -> dict:
    frame_info = ann["caption_frame_info_map"]
    n_frames = len(ann["img_names"])
    per_frame = [frame_label(frame_info.get(i, {}), "action") for i in range(n_frames)]
    true_boundaries = [i for i in range(1, n_frames) if per_frame[i] and per_frame[i - 1] and per_frame[i] != per_frame[i - 1]]
    labels = []
    for row in rows:
        c = int(row["center_frame"])
        labels.append("transition" if any(abs(c - b) <= args.boundary_tolerance_frames for b in true_boundaries) else "steady")
    metrics = neural_classification_task(out_dir, X, np.asarray(labels, dtype=object), rows, args, "transition_detection", "all modalities -> action boundary/steady")
    with (out_dir / "predictions.csv").open("r", encoding="utf-8") as fp:
        pred_rows = list(csv.DictReader(fp))
    pred_frames = [int(r["center_frame"]) for r in pred_rows if r["predicted_label"] == "transition"]
    test_start = min((int(r["center_frame"]) for r in pred_rows), default=0)
    test_end = max((int(r["center_frame"]) for r in pred_rows), default=0)
    true_test = [b for b in true_boundaries if test_start <= b <= test_end]
    metrics.update(boundary_f1(true_test, pred_frames, args.boundary_tolerance_frames))
    write_json(out_dir / "metrics.json", metrics)
    write_csv(out_dir / "true_boundaries.csv", [{"frame": x} for x in true_boundaries], ["frame"])
    return metrics


def neural_next_action(out_dir: Path, X: np.ndarray, rows: list[dict], ann: dict, args: argparse.Namespace) -> dict:
    frame_info = ann["caption_frame_info_map"]
    labels = []
    for row in rows:
        future_frame = min(len(ann["img_names"]) - 1, int(row["end_frame"]) + args.future_frames)
        labels.append(frame_label(frame_info.get(future_frame, {}), "action"))
    return neural_classification_task(out_dir, X, np.asarray(labels, dtype=object), rows, args, "next_action", f"all modalities at t -> action at t+{args.future_frames} frames")


def neural_hand_forecast(out_dir: Path, X: np.ndarray, rows: list[dict], ann: dict, args: argparse.Namespace) -> dict:
    from neural_task_models import train_regressor

    left = ann.get("hand_left_joints")
    right = ann.get("hand_right_joints")
    body = ann.get("smplh_body_joints")
    if left is None or right is None:
        raise ValueError("Hand joints not available.")

    valid_idx, Y = [], []
    n_frames = len(left)
    for i, row in enumerate(rows):
        future_start = int(row["end_frame"]) + 1
        future_end = future_start + args.forecast_frames
        if future_end > n_frames:
            continue
        hand = np.concatenate([left[future_start:future_end], right[future_start:future_end]], axis=1)
        if body is not None and future_end <= len(body):
            root = body[future_start:future_end, :1, :]
            hand = hand - root
        valid_idx.append(i)
        Y.append(hand.reshape(-1))

    valid_idx = np.asarray(valid_idx, dtype=np.int64)
    Y = np.stack(Y).astype(np.float32)
    train, test = chronological_split_indices(len(valid_idx), args.test_fraction)
    result = train_regressor(X[valid_idx], Y, train, test, neural_config(args))
    pred = result["pred"]
    metrics = regression_metrics(Y[test], pred)
    true_hand = Y[test].reshape(len(test), args.forecast_frames, 42, 3)
    pred_hand = pred.reshape(len(test), args.forecast_frames, 42, 3)
    metrics.update({
        "task": "hand_trajectory_forecast",
        "input": "all modalities at t -> future left/right hand 3D joints",
        "split": "chronological",
        "num_windows": int(len(valid_idx)),
        "num_train_windows": int(len(train)),
        "num_test_windows": int(len(test)),
        "forecast_frames": int(args.forecast_frames),
        "mpjpe": float(np.linalg.norm(true_hand - pred_hand, axis=-1).mean()),
        "final_frame_mpjpe": float(np.linalg.norm(true_hand[:, -1] - pred_hand[:, -1], axis=-1).mean()),
        "target_dim": int(Y.shape[1]),
    })
    metrics.update(neural_common_metrics(args, result, "z-score -> MLP regression"))
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "metrics.json", metrics)
    write_json(out_dir / "history.json", result["history"])
    np.savez_compressed(out_dir / "predictions.npz", y_true=Y[test], y_pred=pred, test_window_indices=valid_idx[test])
    save_neural_model(out_dir, result, "regressor", {"target_dim": int(Y.shape[1])})
    return metrics


def neural_object_relevance(out_dir: Path, X: np.ndarray, rows: list[dict], ann: dict, manifest: list[dict], args: argparse.Namespace) -> dict:
    from neural_task_models import train_multilabel

    frame_info = ann["caption_frame_info_map"]
    vocab = OrderedDict()
    labels = []
    for row in rows:
        counts = Counter()
        for frame in range(int(row["start_frame"]), int(row["end_frame"]) + 1):
            counts.update(extract_objects(frame_info.get(frame, {})))
        objects = [obj for obj, count in counts.items() if count > 0]
        for obj in objects:
            if obj not in vocab:
                vocab[obj] = len(vocab)
        labels.append(objects)
    if not vocab:
        raise ValueError("No object labels found.")
    Y = np.zeros((len(rows), len(vocab)), dtype=np.float32)
    for i, objects in enumerate(labels):
        for obj in objects:
            Y[i, vocab[obj]] = 1.0

    keep = block_indices(manifest, exclude=["caption_objects_interaction_text"])
    Xo = X[:, keep]
    train, test = chronological_split_indices(len(rows), args.test_fraction)
    result = train_multilabel(Xo, Y, train, test, neural_config(args))
    prob = result["prob"]
    pred = result["pred"]
    empty = np.where(pred.sum(axis=1) == 0)[0]
    if len(empty):
        pred[empty, np.argmax(prob[empty], axis=1)] = 1
    metrics = multilabel_metrics(Y[test], pred)
    metrics.update({
        "task": "object_relevance",
        "input": "all non-caption modalities -> current relevant object set",
        "split": "chronological",
        "num_windows": int(len(rows)),
        "num_train_windows": int(len(train)),
        "num_test_windows": int(len(test)),
        "num_objects": int(len(vocab)),
        "feature_dim": int(Xo.shape[1]),
    })
    metrics.update(neural_common_metrics(args, result, "z-score -> MLP sigmoid multilabel"))
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "metrics.json", metrics)
    write_json(out_dir / "history.json", result["history"])
    write_json(out_dir / "object_vocab.json", list(vocab.keys()))
    rows_out = []
    names = list(vocab.keys())
    for local_i, global_i in enumerate(test):
        true_objs = [names[j] for j in np.flatnonzero(Y[global_i] > 0)]
        pred_objs = [names[j] for j in np.flatnonzero(pred[local_i] > 0)]
        rows_out.append({
            "window_index": int(global_i),
            "start_frame": rows[int(global_i)]["start_frame"],
            "end_frame": rows[int(global_i)]["end_frame"],
            "true_objects": "|".join(true_objs),
            "predicted_objects": "|".join(pred_objs),
        })
    write_csv(out_dir / "predictions.csv", rows_out, ["window_index", "start_frame", "end_frame", "true_objects", "predicted_objects"])
    save_neural_model(out_dir, result, "multilabel", {"object_vocab": names})
    return metrics


def neural_projection_task(
    out_dir: Path,
    X_in: np.ndarray,
    Y_out: np.ndarray,
    args: argparse.Namespace,
    task_name: str,
    input_desc: str,
    output_desc: str | None = None,
    retrieval_query: np.ndarray | None = None,
    retrieval_candidates: np.ndarray | None = None,
    retrieval_pred_as_query: bool = False,
) -> dict:
    from neural_task_models import train_regressor

    train, test = chronological_split_indices(len(X_in), args.test_fraction)
    result = train_regressor(X_in, Y_out, train, test, neural_config(args))
    pred = result["pred"]
    if retrieval_query is not None and retrieval_candidates is not None:
        if retrieval_pred_as_query:
            metrics = retrieval_metrics(pred, retrieval_candidates[test], np.arange(len(test)))
        else:
            metrics = retrieval_metrics(retrieval_query[test], pred, np.arange(len(test)))
    else:
        metrics = regression_metrics(Y_out[test], pred)
    metrics.update({
        "task": task_name,
        "input": input_desc,
        "split": "chronological",
        "num_train_windows": int(len(train)),
        "num_test_windows": int(len(test)),
        "target_dim": int(Y_out.shape[1]),
    })
    if output_desc is not None:
        metrics["output"] = output_desc
    metrics.update(neural_common_metrics(args, result, "z-score -> MLP projection/regression"))
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "metrics.json", metrics)
    write_json(out_dir / "history.json", result["history"])
    np.savez_compressed(out_dir / "predictions.npz", y_true=Y_out[test], y_pred=pred, test_window_indices=test)
    save_neural_model(out_dir, result, "regressor", {"target_dim": int(Y_out.shape[1])})
    return metrics


def neural_binary_classification_from_arrays(out_dir: Path, X: np.ndarray, y: np.ndarray, args: argparse.Namespace, task: str, input_desc: str) -> dict:
    from neural_task_models import train_classifier

    train, test = chronological_split_indices(len(y), args.test_fraction)
    result = train_classifier(X, y.astype(np.int64), train, test, n_classes=2, config=neural_config(args), use_class_weights=True)
    pred = result["pred"]
    prob = result["prob"]
    metrics = binary_metrics(y[test], pred)
    metrics.update({
        "task": task,
        "input": input_desc,
        "split": "chronological",
        "num_samples": int(len(y)),
        "num_train_samples": int(len(train)),
        "num_test_samples": int(len(test)),
        "feature_dim": int(X.shape[1]),
    })
    metrics.update(neural_common_metrics(args, result, "z-score -> MLP binary softmax"))
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "metrics.json", metrics)
    write_json(out_dir / "history.json", result["history"])
    pred_rows = []
    for k, idx in enumerate(test):
        pred_rows.append({"sample_index": int(idx), "true": int(y[idx]), "predicted": int(pred[k]), "prob_positive": float(prob[k, 1])})
    write_csv(out_dir / "predictions.csv", pred_rows, ["sample_index", "true", "predicted", "prob_positive"])
    save_neural_model(out_dir, result, "classifier", {"class_names": ["negative", "positive"]})
    return metrics


def run_neural_task(task: str, out_dir: Path, X: np.ndarray, rows: list[dict], ann: dict, manifest: list[dict], args: argparse.Namespace) -> dict:
    if task == "timeline_action":
        return neural_classification_task(out_dir, X, label_array(rows, "action_label"), rows, args, task, "all modalities -> current action label")
    if task == "timeline_subtask":
        return neural_classification_task(out_dir, X, label_array(rows, "subtask_label"), rows, args, task, "all modalities -> current subtask label")
    if task == "transition_detection":
        return neural_transition_detection(out_dir, X, rows, ann, args)
    if task == "next_action":
        return neural_next_action(out_dir, X, rows, ann, args)
    if task == "hand_trajectory_forecast":
        return neural_hand_forecast(out_dir, X, rows, ann, args)
    if task == "contact_prediction":
        contacts = ann.get("contacts")
        if contacts is None:
            raise ValueError("Contacts not available.")
        y = []
        for row in rows:
            c = contacts[int(row["start_frame"]):int(row["end_frame"]) + 1]
            y.append("contact" if np.any(c > 0) else "no_contact")
        keep = block_indices(manifest, exclude=["body_contacts", "caption_objects_interaction_text"])
        return neural_classification_task(out_dir, X[:, keep], np.asarray(y, dtype=object), rows, args, task, "all non-contact/non-caption-label modalities -> any body contact")
    if task == "object_relevance":
        return neural_object_relevance(out_dir, X, rows, ann, manifest, args)
    if task == "caption_grounding":
        text_idx = block_indices(manifest, include=["caption_objects_interaction_text"])
        sensor_idx = block_indices(manifest, exclude=["caption_objects_interaction_text"])
        return neural_projection_task(
            out_dir,
            X[:, sensor_idx],
            X[:, text_idx],
            args,
            "caption_grounding",
            "caption objects/interaction text query + candidate sensor windows",
            "matching time window",
            retrieval_query=X[:, text_idx],
            retrieval_candidates=X[:, text_idx],
        )
    if task == "cross_modal_retrieval":
        motion_idx = block_indices(manifest, include=["hand_", "body_joints", "body_contacts", "camera_", "imu_"])
        visual_idx = block_indices(manifest, include=["depth_confidence", "video_"])
        return neural_projection_task(
            out_dir,
            X[:, motion_idx],
            X[:, visual_idx],
            args,
            "cross_modal_retrieval",
            "motion/IMU/camera query",
            "matching depth/video window",
            retrieval_query=X[:, visual_idx],
            retrieval_candidates=X[:, visual_idx],
            retrieval_pred_as_query=True,
        )
    if task == "modality_reconstruction":
        motion_idx = block_indices(manifest, include=["hand_", "body_joints", "body_contacts", "camera_", "imu_"])
        visual_idx = block_indices(manifest, include=["depth_confidence", "video_"])
        return neural_projection_task(out_dir, X[:, motion_idx], X[:, visual_idx], args, "modality_reconstruction", "motion/IMU/camera", "depth/video feature vector")
    if task == "temporal_order":
        pairs, y = [], []
        for i in range(len(X) - 1):
            a, b = X[i], X[i + 1]
            pairs.append(np.concatenate([a, b, b - a]))
            y.append(1)
            pairs.append(np.concatenate([b, a, a - b]))
            y.append(0)
        return neural_binary_classification_from_arrays(out_dir, np.stack(pairs).astype(np.float32), np.asarray(y, dtype=np.int64), args, "temporal_order", "two adjacent windows -> whether order is correct")
    if task == "misalignment_detection":
        motion_idx = block_indices(manifest, include=["hand_", "body_joints", "body_contacts", "camera_", "imu_"])
        visual_idx = block_indices(manifest, include=["depth_confidence", "video_"])
        shift = args.misalignment_shift_windows
        pairs, y = [], []
        limit = len(X) - shift
        for i in range(limit):
            pairs.append(np.concatenate([X[i, motion_idx], X[i, visual_idx]]))
            y.append(1)
            pairs.append(np.concatenate([X[i, motion_idx], X[i + shift, visual_idx]]))
            y.append(0)
        return neural_binary_classification_from_arrays(out_dir, np.stack(pairs).astype(np.float32), np.asarray(y, dtype=np.int64), args, "misalignment_detection", f"motion+visual pair -> aligned vs shifted by {shift} windows")
    raise ValueError(task)


def task_hand_forecast(out_dir: Path, X: np.ndarray, rows: list[dict], ann: dict, args: argparse.Namespace) -> dict:
    left = ann.get("hand_left_joints")
    right = ann.get("hand_right_joints")
    body = ann.get("smplh_body_joints")
    if left is None or right is None:
        raise ValueError("Hand joints not available.")

    valid_idx, Y = [], []
    n_frames = len(left)
    for i, row in enumerate(rows):
        future_start = int(row["end_frame"]) + 1
        future_end = future_start + args.forecast_frames
        if future_end > n_frames:
            continue
        hand = np.concatenate([left[future_start:future_end], right[future_start:future_end]], axis=1)
        if body is not None and future_end <= len(body):
            root = body[future_start:future_end, :1, :]
            hand = hand - root
        valid_idx.append(i)
        Y.append(hand.reshape(-1))

    valid_idx = np.asarray(valid_idx, dtype=np.int64)
    Y = np.stack(Y).astype(np.float32)
    train, test = chronological_split_indices(len(valid_idx), args.test_fraction)
    pred, model = ridge_fit_predict(X[valid_idx[train]], Y[train], X[valid_idx[test]], args.ridge_l2)
    metrics = regression_metrics(Y[test], pred)
    true_hand = Y[test].reshape(len(test), args.forecast_frames, 42, 3)
    pred_hand = pred.reshape(len(test), args.forecast_frames, 42, 3)
    mpjpe = np.linalg.norm(true_hand - pred_hand, axis=-1).mean()
    final_error = np.linalg.norm(true_hand[:, -1] - pred_hand[:, -1], axis=-1).mean()
    metrics.update({
        "task": "hand_trajectory_forecast",
        "input": "all modalities at t -> future left/right hand 3D joints",
        "split": "chronological",
        "num_windows": int(len(valid_idx)),
        "num_train_windows": int(len(train)),
        "num_test_windows": int(len(test)),
        "forecast_frames": int(args.forecast_frames),
        "mpjpe": float(mpjpe),
        "final_frame_mpjpe": float(final_error),
        "target_dim": int(Y.shape[1]),
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "metrics.json", metrics)
    np.savez_compressed(out_dir / "predictions.npz", y_true=Y[test], y_pred=pred, test_window_indices=valid_idx[test], **model)
    return metrics


def task_contact_prediction(out_dir: Path, X: np.ndarray, rows: list[dict], ann: dict, manifest: list[dict], args: argparse.Namespace) -> dict:
    contacts = ann.get("contacts")
    if contacts is None:
        raise ValueError("Contacts not available.")
    y = []
    for row in rows:
        c = contacts[int(row["start_frame"]):int(row["end_frame"]) + 1]
        y.append("contact" if np.any(c > 0) else "no_contact")
    keep = block_indices(manifest, exclude=["body_contacts", "caption_objects_interaction_text"])
    return classification_task(out_dir, X[:, keep], np.asarray(y, dtype=object), rows, args, "contact_prediction", "all non-contact/non-caption-label modalities -> any body contact")


def extract_objects(info: dict) -> list[str]:
    objects = info.get("objects")
    if isinstance(objects, list):
        return [str(x).strip() for x in objects if str(x).strip()]
    if objects:
        return [str(objects).strip()]
    return []


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def train_multilabel_logistic(X: np.ndarray, Y: np.ndarray, epochs: int, lr: float, l2: float, seed: int):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    c = Y.shape[1]
    W = rng.normal(0, 0.01, size=(d, c)).astype(np.float32)
    b = np.zeros(c, dtype=np.float32)
    counts = Y.sum(axis=0)
    pos_weight = (n - counts) / np.maximum(counts, 1.0)
    pos_weight = np.clip(pos_weight, 1.0, 20.0).astype(np.float32)
    history = []
    for epoch in range(1, epochs + 1):
        P = sigmoid(X @ W + b)
        weights = np.where(Y > 0, pos_weight[None, :], 1.0)
        diff = (P - Y) * weights / n
        W -= lr * (X.T @ diff + l2 * W)
        b -= lr * diff.sum(axis=0)
        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 5) == 0:
            pred = (P >= 0.5).astype(np.float32)
            history.append({"epoch": epoch, **multilabel_metrics(Y, pred)})
    return W.astype(np.float32), b.astype(np.float32), history


def multilabel_metrics(Y: np.ndarray, P: np.ndarray) -> dict:
    Y = Y.astype(np.int64)
    P = P.astype(np.int64)
    tp = int(np.sum((Y == 1) & (P == 1)))
    fp = int(np.sum((Y == 0) & (P == 1)))
    fn = int(np.sum((Y == 1) & (P == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    micro_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    per_f1 = []
    for j in range(Y.shape[1]):
        tpj = np.sum((Y[:, j] == 1) & (P[:, j] == 1))
        fpj = np.sum((Y[:, j] == 0) & (P[:, j] == 1))
        fnj = np.sum((Y[:, j] == 1) & (P[:, j] == 0))
        pj = tpj / (tpj + fpj) if tpj + fpj else 0.0
        rj = tpj / (tpj + fnj) if tpj + fnj else 0.0
        per_f1.append(2 * pj * rj / (pj + rj) if pj + rj else 0.0)
    exact = float(np.mean(np.all(Y == P, axis=1)))
    return {"micro_f1": float(micro_f1), "macro_f1": float(np.mean(per_f1)), "exact_match": exact, "precision": precision, "recall": recall}


def task_object_relevance(out_dir: Path, X: np.ndarray, rows: list[dict], ann: dict, manifest: list[dict], args: argparse.Namespace) -> dict:
    frame_info = ann["caption_frame_info_map"]
    vocab = OrderedDict()
    labels = []
    for row in rows:
        counts = Counter()
        for frame in range(int(row["start_frame"]), int(row["end_frame"]) + 1):
            counts.update(extract_objects(frame_info.get(frame, {})))
        objects = [obj for obj, count in counts.items() if count > 0]
        for obj in objects:
            if obj not in vocab:
                vocab[obj] = len(vocab)
        labels.append(objects)
    if not vocab:
        raise ValueError("No object labels found.")
    Y = np.zeros((len(rows), len(vocab)), dtype=np.float32)
    for i, objects in enumerate(labels):
        for obj in objects:
            Y[i, vocab[obj]] = 1.0

    keep = block_indices(manifest, exclude=["caption_objects_interaction_text"])
    Xo = X[:, keep]
    train, test = chronological_split_indices(len(rows), args.test_fraction)
    mean, std = fit_scaler(Xo[train])
    Xs = (Xo - mean) / std
    W, b, history = train_multilabel_logistic(Xs[train], Y[train], args.epochs, 0.05, args.l2, args.seed)
    prob = sigmoid(Xs[test] @ W + b)
    pred = (prob >= 0.5).astype(np.float32)
    # Ensure at least one object is emitted per row.
    empty = np.where(pred.sum(axis=1) == 0)[0]
    if len(empty):
        pred[empty, np.argmax(prob[empty], axis=1)] = 1
    metrics = multilabel_metrics(Y[test], pred)
    metrics.update({
        "task": "object_relevance",
        "input": "all non-caption modalities -> current relevant object set",
        "split": "chronological",
        "num_windows": int(len(rows)),
        "num_train_windows": int(len(train)),
        "num_test_windows": int(len(test)),
        "num_objects": int(len(vocab)),
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "metrics.json", metrics)
    write_json(out_dir / "object_vocab.json", list(vocab.keys()))
    rows_out = []
    names = list(vocab.keys())
    for local_i, global_i in enumerate(test):
        true_objs = [names[j] for j in np.flatnonzero(Y[global_i] > 0)]
        pred_objs = [names[j] for j in np.flatnonzero(pred[local_i] > 0)]
        rows_out.append({
            "window_index": int(global_i),
            "start_frame": rows[int(global_i)]["start_frame"],
            "end_frame": rows[int(global_i)]["end_frame"],
            "true_objects": "|".join(true_objs),
            "predicted_objects": "|".join(pred_objs),
        })
    write_csv(out_dir / "predictions.csv", rows_out, ["window_index", "start_frame", "end_frame", "true_objects", "predicted_objects"])
    np.savez_compressed(out_dir / "model.npz", mean=mean, std=std, W=W, b=b, object_vocab=np.asarray(names, dtype=object), history=np.asarray(history, dtype=object))
    return metrics


def normalize_rows(A: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(A, axis=1, keepdims=True)
    return A / np.maximum(norm, 1e-8)


def retrieval_metrics(query: np.ndarray, candidates: np.ndarray, positive_indices: np.ndarray, topks=(1, 5, 10)) -> dict:
    Q = normalize_rows(query)
    C = normalize_rows(candidates)
    sims = Q @ C.T
    ranks = []
    for i, pos in enumerate(positive_indices):
        order = np.argsort(-sims[i])
        rank = int(np.where(order == pos)[0][0]) + 1
        ranks.append(rank)
    ranks = np.asarray(ranks)
    out = {
        "mrr": float(np.mean(1.0 / ranks)),
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(np.mean(ranks)),
        "num_queries": int(len(ranks)),
    }
    for k in topks:
        out[f"top{k}_accuracy"] = float(np.mean(ranks <= k))
    return out


def task_caption_grounding(out_dir: Path, X: np.ndarray, manifest: list[dict], args: argparse.Namespace) -> dict:
    text_idx = block_indices(manifest, include=["caption_objects_interaction_text"])
    sensor_idx = block_indices(manifest, exclude=["caption_objects_interaction_text"])
    train, test = chronological_split_indices(len(X), args.test_fraction)
    pred_text, model = ridge_fit_predict(X[train][:, sensor_idx], X[train][:, text_idx], X[test][:, sensor_idx], args.ridge_l2)
    # Query is true text; candidates are sensor windows projected into text space.
    metrics = retrieval_metrics(X[test][:, text_idx], pred_text, np.arange(len(test)))
    metrics.update({
        "task": "caption_grounding",
        "input": "caption objects/interaction text query + candidate sensor windows",
        "output": "matching time window",
        "split": "chronological",
        "num_train_windows": int(len(train)),
        "num_test_windows": int(len(test)),
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "metrics.json", metrics)
    np.savez_compressed(out_dir / "model.npz", **model)
    return metrics


def task_cross_modal_retrieval(out_dir: Path, X: np.ndarray, manifest: list[dict], args: argparse.Namespace) -> dict:
    motion_idx = block_indices(manifest, include=["hand_", "body_joints", "body_contacts", "camera_", "imu_"])
    visual_idx = block_indices(manifest, include=["depth_confidence", "video_"])
    train, test = chronological_split_indices(len(X), args.test_fraction)
    pred_visual, model = ridge_fit_predict(X[train][:, motion_idx], X[train][:, visual_idx], X[test][:, motion_idx], args.ridge_l2)
    metrics = retrieval_metrics(pred_visual, X[test][:, visual_idx], np.arange(len(test)))
    metrics.update({
        "task": "cross_modal_retrieval",
        "input": "motion/IMU/camera query",
        "output": "matching depth/video window",
        "split": "chronological",
        "num_train_windows": int(len(train)),
        "num_test_windows": int(len(test)),
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "metrics.json", metrics)
    np.savez_compressed(out_dir / "model.npz", **model)
    return metrics


def task_modality_reconstruction(out_dir: Path, X: np.ndarray, manifest: list[dict], args: argparse.Namespace) -> dict:
    motion_idx = block_indices(manifest, include=["hand_", "body_joints", "body_contacts", "camera_", "imu_"])
    visual_idx = block_indices(manifest, include=["depth_confidence", "video_"])
    train, test = chronological_split_indices(len(X), args.test_fraction)
    pred, model = ridge_fit_predict(X[train][:, motion_idx], X[train][:, visual_idx], X[test][:, motion_idx], args.ridge_l2)
    metrics = regression_metrics(X[test][:, visual_idx], pred)
    metrics.update({
        "task": "modality_reconstruction",
        "input": "motion/IMU/camera",
        "output": "depth/video feature vector",
        "split": "chronological",
        "num_train_windows": int(len(train)),
        "num_test_windows": int(len(test)),
        "target_dim": int(len(visual_idx)),
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "metrics.json", metrics)
    np.savez_compressed(out_dir / "predictions.npz", y_true=X[test][:, visual_idx], y_pred=pred, **model)
    return metrics


def binary_classification_from_arrays(out_dir: Path, X: np.ndarray, y: np.ndarray, args: argparse.Namespace, task: str, input_desc: str) -> dict:
    train, test = chronological_split_indices(len(y), args.test_fraction)
    mean, std = fit_scaler(X[train])
    Xs = (X - mean) / std
    W, b, history = train_softmax_classifier(
        Xs[train],
        y[train].astype(np.int64),
        n_classes=2,
        epochs=args.epochs,
        lr=args.learning_rate,
        l2=args.l2,
        use_class_weights=True,
        seed=args.seed,
    )
    pred, prob = predict(Xs[test], W, b)
    metrics = binary_metrics(y[test], pred)
    metrics.update({
        "task": task,
        "input": input_desc,
        "split": "chronological",
        "num_samples": int(len(y)),
        "num_train_samples": int(len(train)),
        "num_test_samples": int(len(test)),
        "train_final_accuracy": float(history[-1]["train_accuracy"]),
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "metrics.json", metrics)
    pred_rows = []
    for k, idx in enumerate(test):
        pred_rows.append({"sample_index": int(idx), "true": int(y[idx]), "predicted": int(pred[k]), "prob_positive": float(prob[k, 1])})
    write_csv(out_dir / "predictions.csv", pred_rows, ["sample_index", "true", "predicted", "prob_positive"])
    np.savez_compressed(out_dir / "model.npz", mean=mean, std=std, W=W, b=b)
    return metrics


def task_temporal_order(out_dir: Path, X: np.ndarray, args: argparse.Namespace) -> dict:
    pairs, y = [], []
    for i in range(len(X) - 1):
        a, b = X[i], X[i + 1]
        pairs.append(np.concatenate([a, b, b - a]))
        y.append(1)
        pairs.append(np.concatenate([b, a, a - b]))
        y.append(0)
    return binary_classification_from_arrays(out_dir, np.stack(pairs).astype(np.float32), np.asarray(y, dtype=np.int64), args, "temporal_order", "two adjacent windows -> whether order is correct")


def task_misalignment(out_dir: Path, X: np.ndarray, manifest: list[dict], args: argparse.Namespace) -> dict:
    motion_idx = block_indices(manifest, include=["hand_", "body_joints", "body_contacts", "camera_", "imu_"])
    visual_idx = block_indices(manifest, include=["depth_confidence", "video_"])
    shift = args.misalignment_shift_windows
    pairs, y = [], []
    limit = len(X) - shift
    for i in range(limit):
        pairs.append(np.concatenate([X[i, motion_idx], X[i, visual_idx]]))
        y.append(1)
        pairs.append(np.concatenate([X[i, motion_idx], X[i + shift, visual_idx]]))
        y.append(0)
    return binary_classification_from_arrays(out_dir, np.stack(pairs).astype(np.float32), np.asarray(y, dtype=np.int64), args, "misalignment_detection", f"motion+visual pair -> aligned vs shifted by {shift} windows")


def main() -> int:
    args = parse_args()
    add_toolkit_to_path(args.workspace)
    from data_loader import load_from_annotation_hdf5

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks = selected_tasks(args.tasks)

    print(f"Loading annotation: {args.annotation}")
    ann = load_from_annotation_hdf5(args.annotation, 0, None, load_slam_point_cloud=True)
    extras, available_modalities = prepare_modalities(args, ann)
    print("Building shared all-modality windows")
    X, rows, manifest = build_windows(args, ann, extras)

    write_json(args.output_dir / "available_modalities.json", available_modalities)
    write_json(args.output_dir / "feature_manifest.json", manifest)
    write_csv(args.output_dir / "windows.csv", rows, ["window_index", "start_frame", "end_frame", "center_frame", "action_label", "action_fraction", "subtask_label", "subtask_fraction"])
    np.savez_compressed(args.output_dir / "shared_windows.npz", X=X, starts=np.asarray([r["start_frame"] for r in rows]), ends=np.asarray([r["end_frame"] for r in rows]))

    summary = {
        "annotation": portable_path(args.annotation, args.workspace),
        "num_frames": int(len(ann["img_names"])),
        "num_windows": int(len(rows)),
        "feature_dim": int(X.shape[1]),
        "window_frames": int(args.window_frames),
        "stride_frames": int(args.stride_frames),
        "tasks": {},
    }
    if args.include_neural:
        summary["neural_model"] = {
            "name": args.neural_output_name,
            "type": "lightweight PyTorch MLP over shared window features",
            "epochs": int(args.neural_epochs),
            "hidden_dim": int(args.neural_hidden_dim),
            "batch_size": int(args.neural_batch_size),
            "learning_rate": float(args.neural_learning_rate),
            "weight_decay": float(args.neural_weight_decay),
            "dropout": float(args.neural_dropout),
            "device": args.neural_device,
        }
        summary["neural_tasks"] = {}

    print(f"Windows: {len(rows)}, feature_dim: {X.shape[1]}")
    for task in tasks:
        print(f"\nRunning task: {task}")
        out = args.output_dir / task
        try:
            if task == "timeline_action":
                metrics = classification_task(out, X, label_array(rows, "action_label"), rows, args, task, "all modalities -> current action label")
            elif task == "timeline_subtask":
                metrics = classification_task(out, X, label_array(rows, "subtask_label"), rows, args, task, "all modalities -> current subtask label")
            elif task == "transition_detection":
                metrics = task_transition_detection(out, X, rows, ann, args)
            elif task == "next_action":
                metrics = task_next_action(out, X, rows, ann, args)
            elif task == "hand_trajectory_forecast":
                metrics = task_hand_forecast(out, X, rows, ann, args)
            elif task == "contact_prediction":
                metrics = task_contact_prediction(out, X, rows, ann, manifest, args)
            elif task == "object_relevance":
                metrics = task_object_relevance(out, X, rows, ann, manifest, args)
            elif task == "caption_grounding":
                metrics = task_caption_grounding(out, X, manifest, args)
            elif task == "cross_modal_retrieval":
                metrics = task_cross_modal_retrieval(out, X, manifest, args)
            elif task == "modality_reconstruction":
                metrics = task_modality_reconstruction(out, X, manifest, args)
            elif task == "temporal_order":
                metrics = task_temporal_order(out, X, args)
            elif task == "misalignment_detection":
                metrics = task_misalignment(out, X, manifest, args)
            else:
                raise ValueError(task)
            summary["tasks"][task] = metrics
            key_metrics = {k: metrics[k] for k in ("accuracy", "macro_f1", "f1", "mpjpe", "mrr", "r2", "micro_f1") if k in metrics}
            print(f"  done: {key_metrics}")
        except Exception as exc:
            summary["tasks"][task] = {"error": str(exc)}
            write_json(out / "error.json", {"task": task, "error": str(exc)})
            print(f"  error: {exc}")

        if args.include_neural:
            neural_out = args.output_dir / args.neural_output_name / task
            try:
                print(f"  running neural baseline: {args.neural_output_name}")
                neural_metrics = run_neural_task(task, neural_out, X, rows, ann, manifest, args)
                summary["neural_tasks"][task] = neural_metrics
                neural_key_metrics = {k: neural_metrics[k] for k in ("accuracy", "macro_f1", "f1", "mpjpe", "mrr", "r2", "micro_f1") if k in neural_metrics}
                print(f"  neural done: {neural_key_metrics}")
            except Exception as exc:
                summary["neural_tasks"][task] = {"error": str(exc)}
                write_json(neural_out / "error.json", {"task": task, "error": str(exc), "model": args.neural_output_name})
                print(f"  neural error: {exc}")

    write_json(args.output_dir / "summary_report.json", summary)
    print(f"\nSuite artifacts written to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

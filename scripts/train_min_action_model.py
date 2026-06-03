#!/usr/bin/env python3
"""
Minimal end-to-end action-recognition pipeline for an Xperience-10M episode.

Input:
  annotation.hdf5

Features:
  hand joints, body joints, contacts, camera trajectory, IMU summary statistics.

Target:
  caption action_label by default. Use --target subtask for Sub Task labels.

Model:
  Numpy-only multinomial logistic regression.

Outputs:
  metrics.json, per_class_metrics.csv, confusion_matrix.csv, predictions.csv,
  feature_dataset.npz, model.npz.
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


def parse_args() -> argparse.Namespace:
    workspace_default = Path(__file__).resolve().parents[1]
    data_default = workspace_default / "data/sample/xperience-10m-sample/annotation.hdf5"
    out_default = workspace_default / "outputs/min_action_model"

    parser = argparse.ArgumentParser(description="Train a minimal action classifier on Ropedia annotation.hdf5.")
    parser.add_argument("--workspace", type=Path, default=workspace_default, help="Ropedia workspace root.")
    parser.add_argument("--annotation", type=Path, default=data_default, help="Path to annotation.hdf5.")
    parser.add_argument("--output-dir", type=Path, default=out_default, help="Output artifact directory.")
    parser.add_argument("--target", choices=["action", "subtask"], default="action", help="Prediction target.")
    parser.add_argument("--window-frames", type=int, default=20, help="Frames per training window.")
    parser.add_argument("--stride-frames", type=int, default=5, help="Stride between windows.")
    parser.add_argument("--min-label-fraction", type=float, default=0.6, help="Minimum majority-label fraction in a window.")
    parser.add_argument("--test-fraction", type=float, default=0.25, help="Stratified test fraction.")
    parser.add_argument("--epochs", type=int, default=800, help="Training epochs.")
    parser.add_argument("--learning-rate", type=float, default=0.2, help="Softmax learning rate.")
    parser.add_argument("--l2", type=float, default=1e-3, help="L2 weight decay.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--no-class-weights", action="store_true", help="Disable inverse-frequency class weighting.")
    return parser.parse_args()


def add_toolkit_to_path(workspace: Path) -> None:
    toolkit = workspace / "HOMIE-toolkit"
    if not toolkit.exists():
        raise FileNotFoundError(f"HOMIE-toolkit not found: {toolkit}")
    sys.path.insert(0, str(toolkit))


def portable_path(path: Path, workspace: Path | None = None) -> str:
    roots = [workspace, Path.cwd()]
    for root in roots:
        if root is None:
            continue
        try:
            return path.resolve().relative_to(Path(root).resolve()).as_posix()
        except (FileNotFoundError, ValueError):
            continue
    return path.name


def temporal_stats(arr: np.ndarray) -> np.ndarray:
    """Return fixed statistics over time for an array shaped (T, ...)."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr[:, None]
    flat = arr.reshape(arr.shape[0], -1)
    flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
    if flat.shape[0] == 0:
        raise ValueError("temporal_stats received an empty time axis")

    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    amin = flat.min(axis=0)
    amax = flat.max(axis=0)
    delta = flat[-1] - flat[0]
    if flat.shape[0] > 1:
        vel = np.diff(flat, axis=0)
        vel_mean = vel.mean(axis=0)
        vel_std = vel.std(axis=0)
    else:
        vel_mean = np.zeros(flat.shape[1], dtype=np.float32)
        vel_std = np.zeros(flat.shape[1], dtype=np.float32)
    return np.concatenate([mean, std, amin, amax, delta, vel_mean, vel_std]).astype(np.float32)


def safe_window(arr: np.ndarray | None, start: int, end: int) -> np.ndarray | None:
    if arr is None:
        return None
    if start >= len(arr):
        return None
    return np.asarray(arr[start:min(end, len(arr))])


def center_by_body_root(values: np.ndarray, body: np.ndarray | None) -> np.ndarray:
    if body is None or len(body) != len(values) or body.ndim < 3 or body.shape[-1] != 3:
        return values
    root = body[:, :1, :]
    return values - root


def extract_window_features(ann: dict, start: int, end: int) -> np.ndarray:
    body = safe_window(ann.get("smplh_body_joints"), start, end)
    left = safe_window(ann.get("hand_left_joints"), start, end)
    right = safe_window(ann.get("hand_right_joints"), start, end)
    contacts = safe_window(ann.get("contacts"), start, end)
    cam_t = safe_window(ann.get("t_c2w_all"), start, end)

    chunks: list[np.ndarray] = []

    if left is not None:
        chunks.append(temporal_stats(center_by_body_root(left, body)))
    if right is not None:
        chunks.append(temporal_stats(center_by_body_root(right, body)))
    if body is not None:
        root = body[:, :1, :] if body.ndim == 3 else 0.0
        chunks.append(temporal_stats(body - root))
    if contacts is not None:
        chunks.append(temporal_stats(contacts))
    if cam_t is not None:
        cam_t = cam_t - cam_t[:1]
        chunks.append(temporal_stats(cam_t))

    imu_accel = ann.get("imu_accel_xyz")
    imu_gyro = ann.get("imu_gyro_xyz")
    imu_keyframes = ann.get("imu_keyframe_indices")
    if imu_accel is not None and imu_gyro is not None and imu_keyframes is not None and len(imu_keyframes) > end - 1:
        imu_start = int(max(0, imu_keyframes[start]))
        imu_end = int(min(len(imu_accel), max(imu_start + 1, imu_keyframes[end - 1] + 1)))
        imu = np.concatenate([imu_accel[imu_start:imu_end], imu_gyro[imu_start:imu_end]], axis=1)
        chunks.append(temporal_stats(imu))

    if not chunks:
        raise ValueError("No usable numeric modalities found in annotation.")
    return np.concatenate(chunks).astype(np.float32)


def frame_label(info: dict, target: str) -> str:
    if target == "subtask":
        label = info.get("theme", "")
    else:
        label = info.get("action_label", "")
    label = str(label).strip()
    if not label or label.upper() == "N/A":
        return ""
    return label


def majority_label(labels: list[str], min_fraction: float) -> tuple[str, float]:
    labels = [x for x in labels if x]
    if not labels:
        return "", 0.0
    label, count = Counter(labels).most_common(1)[0]
    frac = count / len(labels)
    if frac < min_fraction:
        return "", frac
    return label, frac


def build_feature_dataset(ann: dict, target: str, window_frames: int, stride_frames: int, min_label_fraction: float):
    frame_info = ann.get("caption_frame_info_map")
    if frame_info is None:
        raise ValueError("No caption_frame_info_map found in annotation.")

    n_frames = len(ann["img_names"])
    X, y_labels, starts, ends, label_fracs = [], [], [], [], []
    for start in range(0, n_frames - window_frames + 1, stride_frames):
        end = start + window_frames
        labels = [frame_label(frame_info.get(i, {}), target) for i in range(start, end)]
        label, frac = majority_label(labels, min_label_fraction)
        if not label:
            continue
        X.append(extract_window_features(ann, start, end))
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
    )


def encode_labels(y_labels: np.ndarray) -> tuple[np.ndarray, list[str]]:
    seen = OrderedDict()
    for label in y_labels:
        if label not in seen:
            seen[label] = len(seen)
    class_names = list(seen.keys())
    y = np.asarray([seen[label] for label in y_labels], dtype=np.int64)
    return y, class_names


def stratified_split(y: np.ndarray, test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        if len(idx) < 2:
            train_idx.extend(idx.tolist())
            continue
        n_test = int(round(len(idx) * test_fraction))
        n_test = max(1, min(n_test, len(idx) - 1))
        test_idx.extend(idx[:n_test].tolist())
        train_idx.extend(idx[n_test:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return np.asarray(train_idx, dtype=np.int64), np.asarray(test_idx, dtype=np.int64)


def fit_scaler(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def train_softmax_classifier(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    epochs: int,
    lr: float,
    l2: float,
    use_class_weights: bool,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    rng = np.random.default_rng(seed)
    n, d = X.shape
    W = rng.normal(0.0, 0.01, size=(d, n_classes)).astype(np.float32)
    b = np.zeros(n_classes, dtype=np.float32)
    onehot = np.eye(n_classes, dtype=np.float32)[y]

    if use_class_weights:
        counts = np.bincount(y, minlength=n_classes).astype(np.float32)
        weights_by_class = n / np.maximum(counts, 1.0) / n_classes
        sample_weights = weights_by_class[y]
    else:
        sample_weights = np.ones(n, dtype=np.float32)
    sample_weights = sample_weights / sample_weights.mean()

    history = []
    report_every = max(1, epochs // 10)
    for epoch in range(1, epochs + 1):
        logits = X @ W + b
        probs = softmax(logits)
        weighted_diff = (probs - onehot) * sample_weights[:, None] / n
        grad_W = X.T @ weighted_diff + l2 * W
        grad_b = weighted_diff.sum(axis=0)
        W -= lr * grad_W
        b -= lr * grad_b

        if epoch == 1 or epoch == epochs or epoch % report_every == 0:
            p_true = np.clip(probs[np.arange(n), y], 1e-9, 1.0)
            loss = float(-(sample_weights * np.log(p_true)).mean() + 0.5 * l2 * float(np.sum(W * W)))
            acc = float(np.mean(np.argmax(probs, axis=1) == y))
            history.append({"epoch": epoch, "loss": loss, "train_accuracy": acc})
    return W.astype(np.float32), b.astype(np.float32), history


def predict(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probs = softmax(X @ W + b)
    return np.argmax(probs, axis=1), probs


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> tuple[dict, list[dict], np.ndarray]:
    n_classes = len(class_names)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1

    rows = []
    recalls, f1s, weighted_f1_total = [], [], 0.0
    support_total = int(cm.sum())
    for i, name in enumerate(class_names):
        tp = int(cm[i, i])
        support = int(cm[i, :].sum())
        pred_count = int(cm[:, i].sum())
        precision = tp / pred_count if pred_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if support:
            recalls.append(recall)
            f1s.append(f1)
            weighted_f1_total += f1 * support
        rows.append({
            "class_id": i,
            "class_name": name,
            "support": support,
            "predicted": pred_count,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    accuracy = float(np.mean(y_true == y_pred)) if len(y_true) else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    balanced_accuracy = float(np.mean(recalls)) if recalls else 0.0
    weighted_f1 = float(weighted_f1_total / support_total) if support_total else 0.0
    metrics = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "num_eval_windows": int(len(y_true)),
        "num_classes": n_classes,
    }
    return metrics, rows, cm


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def save_artifacts(
    output_dir: Path,
    X: np.ndarray,
    y: np.ndarray,
    y_labels: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    label_fracs: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    class_names: list[str],
    mean: np.ndarray,
    std: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    history: list[dict],
    metrics: dict,
    per_class_rows: list[dict],
    cm: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_dir / "feature_dataset.npz",
        X=X,
        y=y,
        labels=y_labels.astype(str),
        start_frame=starts,
        end_frame=ends,
        label_fraction=label_fracs,
        train_idx=train_idx,
        test_idx=test_idx,
        class_names=np.asarray(class_names, dtype=object),
    )
    np.savez_compressed(output_dir / "model.npz", mean=mean, std=std, W=W, b=b, class_names=np.asarray(class_names, dtype=object))

    metadata = {
        "annotation": portable_path(args.annotation, args.workspace),
        "target": args.target,
        "window_frames": args.window_frames,
        "stride_frames": args.stride_frames,
        "min_label_fraction": args.min_label_fraction,
        "test_fraction": args.test_fraction,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "l2": args.l2,
        "class_weights": not args.no_class_weights,
        "num_windows": int(len(y)),
        "num_features": int(X.shape[1]),
        "num_train_windows": int(len(train_idx)),
        "num_test_windows": int(len(test_idx)),
        "classes": class_names,
        "history": history,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    write_csv(
        output_dir / "per_class_metrics.csv",
        per_class_rows,
        ["class_id", "class_name", "support", "predicted", "precision", "recall", "f1"],
    )

    with (output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp, lineterminator="\n")
        writer.writerow(["true\\pred"] + class_names)
        for i, name in enumerate(class_names):
            writer.writerow([name] + [int(v) for v in cm[i]])

    pred_rows = []
    pred_lookup = {int(idx): k for k, idx in enumerate(test_idx)}
    for idx in test_idx:
        idx = int(idx)
        k = pred_lookup[idx]
        pred_id = int(y_pred[k])
        true_id = int(y[idx])
        pred_rows.append({
            "window_index": idx,
            "start_frame": int(starts[idx]),
            "end_frame": int(ends[idx]),
            "true_label": class_names[true_id],
            "predicted_label": class_names[pred_id],
            "confidence": float(probs[k, pred_id]),
            "correct": int(pred_id == true_id),
            "label_fraction": float(label_fracs[idx]),
        })
    write_csv(
        output_dir / "predictions.csv",
        pred_rows,
        ["window_index", "start_frame", "end_frame", "true_label", "predicted_label", "confidence", "correct", "label_fraction"],
    )


def main() -> int:
    args = parse_args()
    add_toolkit_to_path(args.workspace)
    from data_loader import load_from_annotation_hdf5

    if not args.annotation.exists():
        raise FileNotFoundError(f"annotation.hdf5 not found: {args.annotation}")

    print(f"Loading annotation: {args.annotation}")
    ann = load_from_annotation_hdf5(args.annotation, 0, None, load_slam_point_cloud=False)

    print("Building windowed feature dataset")
    X, y_labels, starts, ends, label_fracs = build_feature_dataset(
        ann,
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
    metrics["train_final_accuracy"] = history[-1]["train_accuracy"] if history else math.nan
    metrics["train_final_loss"] = history[-1]["loss"] if history else math.nan

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

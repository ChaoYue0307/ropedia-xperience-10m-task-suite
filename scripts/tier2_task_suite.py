#!/usr/bin/env python3
"""Historical provenance rows for the Xperience-10M unified 20-task suite.

The public benchmark is presented as one 20-task suite. The output path keeps
its historical ``tier2_task_suite`` name for backwards-compatible public links,
but the generated rows are read inside the unified suite.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "episode_task_suite"
OUT_DIR = RESULTS / "tier2_task_suite"
DOCS_DATA = ROOT / "docs" / "data"
CHARTS = ROOT / "docs" / "assets" / "charts"

sys.path.insert(0, str(ROOT / "scripts"))

from train_min_action_model import add_toolkit_to_path, compute_metrics, fit_scaler, frame_label, majority_label, predict, train_softmax_classifier
from neural_task_models import NeuralConfig, train_classifier, train_multilabel, train_regressor
from research_direction_extension_tasks import (
    block_indices,
    chronological_split,
    regression_metrics,
    retrieval_metrics,
    ridge_predict,
)


TIER2_TASK_SPECS: OrderedDict[str, dict[str, Any]] = OrderedDict(
    [
        (
            "long_horizon_next_action",
            {
                "name": "Long-Horizon Next-Action Forecasting",
                "family": "classification",
                "input": "Current 20-frame non-caption multimodal window.",
                "target": "Action label five seconds later.",
                "metric_key": "macro_f1",
                "metric_name": "macro-F1",
                "metric_direction": "higher",
                "meaning": "Tests whether the current state carries enough procedure context to forecast beyond the one-second core next-action task.",
            },
        ),
        (
            "next_subtask_forecast",
            {
                "name": "Long-Horizon Next-Subtask Forecasting",
                "family": "classification",
                "input": "Current 20-frame non-caption multimodal window.",
                "target": "Procedure subtask label five seconds later.",
                "metric_key": "macro_f1",
                "metric_name": "macro-F1",
                "metric_direction": "higher",
                "meaning": "Moves from immediate action anticipation to higher-level procedure-state prediction.",
            },
        ),
        (
            "interaction_text_prediction",
            {
                "name": "Interaction Text Prediction",
                "family": "classification",
                "input": "Current 20-frame sensor window with caption-text features removed.",
                "target": "Raw annotation interaction phrase for the same window.",
                "metric_key": "macro_f1",
                "metric_name": "macro-F1",
                "metric_direction": "higher",
                "meaning": "Uses the raw caption JSON interaction field as a language target instead of only the hashed text feature.",
            },
        ),
        (
            "action_object_relation",
            {
                "name": "Action-Object Relation Prediction",
                "family": "classification",
                "input": "Current 20-frame sensor window with caption-text features removed.",
                "target": "Joint action plus active object-set relation.",
                "metric_key": "macro_f1",
                "metric_name": "macro-F1",
                "metric_direction": "higher",
                "meaning": "Evaluates whether a model can bind what action is happening to which objects are involved.",
            },
        ),
        (
            "object_set_forecast",
            {
                "name": "Future Object-Set Forecasting",
                "family": "multi_label",
                "input": "Current 20-frame sensor window with caption-text features removed.",
                "target": "Object set active five seconds later.",
                "metric_key": "micro_f1",
                "metric_name": "micro-F1",
                "metric_direction": "higher",
                "meaning": "Predicts which objects will become relevant soon, not only which objects are relevant now.",
            },
        ),
        (
            "imu_to_hand_pose",
            {
                "name": "IMU-to-Hand Pose Reconstruction",
                "family": "regression",
                "input": "Current IMU acceleration/gyroscope feature block only.",
                "target": "Current left/right hand joint feature blocks.",
                "metric_key": "mae",
                "metric_name": "MAE",
                "metric_direction": "lower",
                "meaning": "A sensor-bridge probe for how much hand configuration can be recovered from inertial motion alone.",
            },
        ),
        (
            "camera_view_sync_retrieval",
            {
                "name": "Camera-View Synchronization Retrieval",
                "family": "retrieval",
                "input": "Fisheye camera-1 feature query projected into fisheye camera-3 feature space.",
                "target": "The synchronized held-out camera-3 window.",
                "metric_key": "mrr",
                "metric_name": "MRR",
                "metric_direction": "higher",
                "meaning": "Stress-tests multi-camera time alignment beyond the core cross-modal retrieval task.",
            },
        ),
        (
            "time_to_transition",
            {
                "name": "Time-to-Next-Transition Regression",
                "family": "regression",
                "input": "Current 20-frame non-caption multimodal window.",
                "target": "Frames until the next action-label boundary, capped at 200 frames.",
                "metric_key": "mae",
                "metric_name": "MAE frames",
                "metric_direction": "lower",
                "meaning": "Turns boundary detection into a continuous timing estimate for procedural control.",
            },
        ),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--annotation", type=Path, default=ROOT / "data/sample/xperience-10m-sample/annotation.hdf5")
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--stride-frames", type=int, default=5)
    parser.add_argument("--future-windows", type=int, default=20, help="Long-horizon offset in 5-frame windows for task 13.")
    parser.add_argument("--transition-cap-frames", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--learning-rate", type=float, default=0.12)
    parser.add_argument("--l2", type=float, default=2e-3)
    parser.add_argument("--ridge-l2", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--skip-neural", action="store_true")
    parser.add_argument("--neural-epochs", type=int, default=25)
    parser.add_argument("--neural-hidden-dim", type=int, default=128)
    parser.add_argument("--neural-batch-size", type=int, default=128)
    parser.add_argument("--neural-learning-rate", type=float, default=1e-3)
    parser.add_argument("--neural-weight-decay", type=float, default=1e-4)
    parser.add_argument("--neural-dropout", type=float, default=0.10)
    parser.add_argument("--neural-device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def load_windows(results_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, str]], list[dict[str, Any]]]:
    z = np.load(results_dir / "shared_windows.npz", allow_pickle=False)
    X = np.nan_to_num(np.asarray(z["X"], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    starts = np.asarray(z["starts"], dtype=np.int64)
    ends = np.asarray(z["ends"], dtype=np.int64)
    with (results_dir / "windows.csv").open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    manifest = json.loads((results_dir / "feature_manifest.json").read_text(encoding="utf-8"))
    if len(rows) != len(X):
        raise ValueError(f"windows.csv has {len(rows)} rows but shared_windows.npz has {len(X)} rows.")
    return X, starts, ends, rows, manifest


def load_annotation(args: argparse.Namespace) -> dict[str, Any]:
    if not args.annotation.exists():
        raise FileNotFoundError(f"Missing annotation file: {args.annotation}")
    try:
        add_toolkit_to_path(args.workspace)
    except FileNotFoundError:
        return load_annotation_direct(args.annotation)

    from data_loader import load_from_annotation_hdf5

    return load_from_annotation_hdf5(args.annotation, 0, None, load_slam_point_cloud=False)


def nearest_frame_index(timestamps: np.ndarray, value: Any) -> int:
    text = str(value or "").strip()
    frame_match = re.fullmatch(r"frame_(\d+)", text)
    if frame_match:
        return max(0, min(len(timestamps) - 1, int(frame_match.group(1))))
    numeric_match = re.search(r"\d+", text)
    if not numeric_match:
        return 0
    target = int(numeric_match.group(0))
    if target < len(timestamps):
        return max(0, min(len(timestamps) - 1, target))
    idx = int(np.searchsorted(timestamps, target))
    if idx <= 0:
        return 0
    if idx >= len(timestamps):
        return len(timestamps) - 1
    before = timestamps[idx - 1]
    after = timestamps[idx]
    return idx - 1 if abs(target - int(before)) <= abs(int(after) - target) else idx


def ensure_frame_info(frame_info: dict[int, dict[str, Any]], frame: int) -> dict[str, Any]:
    return frame_info.setdefault(int(frame), {})


def load_annotation_direct(annotation: Path) -> dict[str, Any]:
    """Load just the caption fields needed for the provenance rows without HOMIE."""
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            "Regenerating the historical provenance rows needs HOMIE-toolkit or h5py. Use the project Ropedia virtualenv if system Python lacks h5py."
        ) from exc

    with h5py.File(annotation, "r") as handle:
        timestamp_raw = handle["video/device_timestamp"][:]
        timestamps = np.asarray(
            [int(item.decode("utf-8") if isinstance(item, bytes) else item) for item in timestamp_raw],
            dtype=np.int64,
        )
        caption_raw = handle["caption"][()]
    caption_text = caption_raw.decode("utf-8") if isinstance(caption_raw, bytes) else str(caption_raw)
    caption = json.loads(caption_text)
    frame_info: dict[int, dict[str, Any]] = {}

    for segment in caption.get("segments", []):
        seg_start = nearest_frame_index(timestamps, segment.get("start_frame"))
        seg_end = nearest_frame_index(timestamps, segment.get("end_frame"))
        if seg_end < seg_start:
            seg_start, seg_end = seg_end, seg_start
        subtask = str(segment.get("Sub Task", "") or "").strip()
        for frame in range(seg_start, seg_end + 1):
            ensure_frame_info(frame_info, frame)["theme"] = subtask

        for action in segment.get("Current Action", []) or []:
            action_start = nearest_frame_index(timestamps, action.get("start_frame"))
            action_end = nearest_frame_index(timestamps, action.get("end_frame"))
            if action_end < action_start:
                action_start, action_end = action_end, action_start
            label = str(action.get("label", "") or "").strip()
            desc = str(action.get("description", "") or "").strip()
            for frame in range(action_start, action_end + 1):
                info = ensure_frame_info(frame_info, frame)
                info["action_label"] = label
                info["action_desc"] = desc
                if subtask:
                    info.setdefault("theme", subtask)

        for timestamp, objects in (segment.get("objects") or {}).items():
            frame = nearest_frame_index(timestamps, timestamp)
            ensure_frame_info(frame_info, frame)["objects"] = objects

        for timestamp, interaction in (segment.get("interaction") or {}).items():
            frame = nearest_frame_index(timestamps, timestamp)
            ensure_frame_info(frame_info, frame)["interaction"] = interaction

    return {
        "img_names": [f"{idx:06d}" for idx in range(len(timestamps))],
        "caption_frame_info_map": frame_info,
    }


def extract_objects(info: dict[str, Any]) -> list[str]:
    objects = info.get("objects")
    if isinstance(objects, list):
        return [str(item).strip() for item in objects if str(item).strip()]
    if objects:
        return [str(objects).strip()]
    return []


def interaction_label(info: dict[str, Any]) -> str:
    value = str(info.get("interaction", "") or "").strip()
    if not value or value.upper() == "N/A":
        return ""
    return value


def majority_window_label(
    frame_info: dict[int, dict[str, Any]],
    start: int,
    end: int,
    getter,
    min_fraction: float = 0.35,
) -> str:
    labels = [getter(frame_info.get(frame, {})) for frame in range(start, end + 1)]
    label, _frac = majority_label(labels, min_fraction)
    return label


def window_objects(frame_info: dict[int, dict[str, Any]], start: int, end: int) -> list[str]:
    counts: Counter[str] = Counter()
    for frame in range(start, end + 1):
        counts.update(extract_objects(frame_info.get(frame, {})))
    return sorted(counts)


def neural_config(args: argparse.Namespace) -> NeuralConfig:
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


def encode_labels_train_first(labels: list[str], train_idx: np.ndarray) -> tuple[np.ndarray, list[str], int]:
    seen: OrderedDict[str, int] = OrderedDict()
    for idx in train_idx:
        label = labels[int(idx)]
        if label not in seen:
            seen[label] = len(seen)
    train_class_count = len(seen)
    for label in labels:
        if label not in seen:
            seen[label] = len(seen)
    y = np.asarray([seen[label] for label in labels], dtype=np.int64)
    return y, list(seen.keys()), train_class_count


def metric_value(task_id: str, metrics: dict[str, Any] | None) -> float | None:
    if not metrics:
        return None
    key = TIER2_TASK_SPECS[task_id]["metric_key"]
    value = metrics.get(key)
    if value is None:
        value = metrics.get("primary_score")
    return float(value) if value is not None else None


def softmax_classification(
    task_id: str,
    X: np.ndarray,
    labels: list[str],
    rows: list[dict[str, str]],
    input_description: str,
    out_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    valid = np.asarray([bool(label) for label in labels])
    keep = np.flatnonzero(valid)
    Xv = X[keep]
    label_values = [labels[int(i)] for i in keep]
    row_values = [rows[int(i)] for i in keep]
    train_idx, test_idx = chronological_split(len(keep), args.train_fraction)
    y, class_names, train_class_count = encode_labels_train_first(label_values, train_idx)

    mean, std = fit_scaler(Xv[train_idx])
    Xs = (Xv - mean) / std
    W, b, history = train_softmax_classifier(
        Xs[train_idx],
        y[train_idx],
        n_classes=train_class_count,
        epochs=args.epochs,
        lr=args.learning_rate,
        l2=args.l2,
        use_class_weights=True,
        seed=args.seed,
    )
    pred, prob = predict(Xs[test_idx], W, b)
    metrics, per_class, cm = compute_metrics(y[test_idx], pred, class_names)
    majority_class = Counter(y[train_idx]).most_common(1)[0][0]
    unseen = sorted(set(int(v) for v in y[test_idx]) - set(int(v) for v in y[train_idx]))
    metrics.update(
        {
            "status": "pass",
            "task": task_id,
            "task_display_name": TIER2_TASK_SPECS[task_id]["name"],
            "suite_position": "unified_20_task_provenance",
            "model_family": "minimal_softmax",
            "input": input_description,
            "split": "single_episode_chronological",
            "num_windows": int(len(keep)),
            "num_train_windows": int(len(train_idx)),
            "num_test_windows": int(len(test_idx)),
            "num_classes": int(len(class_names)),
            "num_train_classes": int(train_class_count),
            "majority_baseline_accuracy": float(np.mean(y[test_idx] == majority_class)),
            "primary_metric": "macro_f1",
            "primary_score": metrics["macro_f1"],
            "unseen_test_class_count": len(unseen),
            "unseen_test_classes": [class_names[i] for i in unseen],
            "history": history,
        }
    )
    task_dir = out_dir / task_id
    pred_rows = []
    for k, local_idx in enumerate(test_idx):
        row = row_values[int(local_idx)]
        true_id = int(y[int(local_idx)])
        pred_id = int(pred[k])
        pred_rows.append(
            {
                "window_index": row["window_index"],
                "start_frame": row["start_frame"],
                "end_frame": row["end_frame"],
                "true_label": class_names[true_id],
                "predicted_label": class_names[pred_id],
                "confidence": float(prob[k, pred_id]),
                "correct": int(true_id == pred_id),
            }
        )
    write_json(task_dir / "metrics.json", metrics)
    write_csv(task_dir / "per_class_metrics.csv", per_class)
    write_csv(task_dir / "predictions.csv", pred_rows)
    np.savez_compressed(task_dir / "model.npz", mean=mean, std=std, W=W, b=b, class_names=np.asarray(class_names, dtype=object))

    neural_metrics = None
    if not args.skip_neural:
        neural_dir = out_dir / "neural_mlp" / task_id
        neural_dir.mkdir(parents=True, exist_ok=True)
        result = train_classifier(Xv, y, train_idx, test_idx, n_classes=len(class_names), config=neural_config(args), use_class_weights=True)
        nn_metrics, nn_per_class, nn_cm = compute_metrics(y[test_idx], result["pred"], class_names)
        nn_metrics.update(
            {
                "status": "pass",
                "task": task_id,
                "task_display_name": TIER2_TASK_SPECS[task_id]["name"],
                "suite_position": "unified_20_task_provenance",
                "model_family": "neural_mlp",
                "input": input_description,
                "split": "single_episode_chronological",
                "num_windows": int(len(keep)),
                "num_train_windows": int(len(train_idx)),
                "num_test_windows": int(len(test_idx)),
                "num_classes": int(len(class_names)),
                "primary_metric": "macro_f1",
                "primary_score": nn_metrics["macro_f1"],
                "history": result["history"],
                "device": result["device"],
            }
        )
        write_json(neural_dir / "metrics.json", nn_metrics)
        write_csv(neural_dir / "per_class_metrics.csv", nn_per_class)
        nn_rows = []
        for k, local_idx in enumerate(test_idx):
            row = row_values[int(local_idx)]
            true_id = int(y[int(local_idx)])
            pred_id = int(result["pred"][k])
            nn_rows.append(
                {
                    "window_index": row["window_index"],
                    "start_frame": row["start_frame"],
                    "end_frame": row["end_frame"],
                    "true_label": class_names[true_id],
                    "predicted_label": class_names[pred_id],
                    "correct": int(true_id == pred_id),
                }
            )
        write_csv(neural_dir / "predictions.csv", nn_rows)
        neural_metrics = nn_metrics

    return {"minimal": metrics, "neural_mlp": neural_metrics}


def multilabel_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tp = float(np.logical_and(y_true == 1, y_pred == 1).sum())
    fp = float(np.logical_and(y_true == 0, y_pred == 1).sum())
    fn = float(np.logical_and(y_true == 1, y_pred == 0).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    micro_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    exact = float(np.all(y_true == y_pred, axis=1).mean()) if len(y_true) else 0.0
    per_label_f1 = []
    for col in range(y_true.shape[1]):
        c_tp = float(np.logical_and(y_true[:, col] == 1, y_pred[:, col] == 1).sum())
        c_fp = float(np.logical_and(y_true[:, col] == 0, y_pred[:, col] == 1).sum())
        c_fn = float(np.logical_and(y_true[:, col] == 1, y_pred[:, col] == 0).sum())
        c_p = c_tp / (c_tp + c_fp) if c_tp + c_fp else 0.0
        c_r = c_tp / (c_tp + c_fn) if c_tp + c_fn else 0.0
        per_label_f1.append(2 * c_p * c_r / (c_p + c_r) if c_p + c_r else 0.0)
    return {
        "precision": precision,
        "recall": recall,
        "micro_f1": micro_f1,
        "macro_f1": float(np.mean(per_label_f1)) if per_label_f1 else 0.0,
        "exact_match": exact,
    }


def object_set_forecast(
    X: np.ndarray,
    rows: list[dict[str, str]],
    manifest: list[dict[str, Any]],
    frame_info: dict[int, dict[str, Any]],
    out_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    task_id = "object_set_forecast"
    horizon = args.future_windows * args.stride_frames
    valid = []
    labels: list[list[str]] = []
    n_frames = max(frame_info.keys()) + 1 if frame_info else 0
    for i, row in enumerate(rows):
        start = int(row["start_frame"]) + horizon
        end = int(row["end_frame"]) + horizon
        if end >= n_frames:
            continue
        objects = window_objects(frame_info, start, end)
        if objects:
            valid.append(i)
            labels.append(objects)
    valid_idx = np.asarray(valid, dtype=np.int64)
    train_idx, test_idx = chronological_split(len(valid_idx), args.train_fraction)
    vocab_counter: Counter[str] = Counter()
    for idx in train_idx:
        vocab_counter.update(labels[int(idx)])
    vocab = [name for name, _count in vocab_counter.most_common()]
    object_to_col = {name: col for col, name in enumerate(vocab)}
    Y = np.zeros((len(valid_idx), len(vocab)), dtype=np.float32)
    unseen_test_objects: Counter[str] = Counter()
    for row_idx, objects in enumerate(labels):
        for obj in objects:
            if obj in object_to_col:
                Y[row_idx, object_to_col[obj]] = 1.0
            elif row_idx in set(int(i) for i in test_idx):
                unseen_test_objects[obj] += 1

    input_idx = block_indices(manifest, exclude=["caption_objects_interaction_text"])
    Xv = X[valid_idx][:, input_idx]
    pred_score = ridge_predict(Xv[train_idx], Y[train_idx], Xv[test_idx], l2=args.ridge_l2, standardize_y=False)
    threshold = np.maximum(0.10, Y[train_idx].mean(axis=0))
    pred = (pred_score >= threshold[None, :]).astype(np.float32)
    empty = np.where(pred.sum(axis=1) == 0)[0]
    if len(empty) and len(vocab):
        pred[empty, np.argmax(pred_score[empty], axis=1)] = 1.0
    metrics = multilabel_metrics(Y[test_idx], pred)
    metrics.update(
        {
            "status": "pass",
            "task": task_id,
            "task_display_name": TIER2_TASK_SPECS[task_id]["name"],
            "suite_position": "unified_20_task_provenance",
            "model_family": "minimal_ridge_multilabel",
            "input": TIER2_TASK_SPECS[task_id]["input"],
            "split": "single_episode_chronological",
            "num_windows": int(len(valid_idx)),
            "num_train_windows": int(len(train_idx)),
            "num_test_windows": int(len(test_idx)),
            "num_objects": int(len(vocab)),
            "future_horizon_frames": int(horizon),
            "primary_metric": "micro_f1",
            "primary_score": metrics["micro_f1"],
            "unseen_test_objects": dict(unseen_test_objects),
        }
    )
    task_dir = out_dir / task_id
    pred_rows = []
    names = vocab
    for local_k, local_idx in enumerate(test_idx):
        global_idx = int(valid_idx[int(local_idx)])
        true_objs = [names[j] for j in np.flatnonzero(Y[int(local_idx)] > 0)]
        pred_objs = [names[j] for j in np.flatnonzero(pred[local_k] > 0)]
        pred_rows.append(
            {
                "window_index": rows[global_idx]["window_index"],
                "future_start_frame": int(rows[global_idx]["start_frame"]) + horizon,
                "future_end_frame": int(rows[global_idx]["end_frame"]) + horizon,
                "true_objects": "|".join(true_objs),
                "predicted_objects": "|".join(pred_objs),
            }
        )
    write_json(task_dir / "metrics.json", metrics)
    write_json(task_dir / "object_vocab.json", names)
    write_csv(task_dir / "predictions.csv", pred_rows)

    neural_metrics = None
    if not args.skip_neural and len(vocab):
        neural_dir = out_dir / "neural_mlp" / task_id
        neural_dir.mkdir(parents=True, exist_ok=True)
        result = train_multilabel(Xv, Y, train_idx, test_idx, neural_config(args))
        nn_pred = result["pred"]
        empty = np.where(nn_pred.sum(axis=1) == 0)[0]
        if len(empty):
            nn_pred[empty, np.argmax(result["prob"][empty], axis=1)] = 1.0
        nn_metrics = multilabel_metrics(Y[test_idx], nn_pred)
        nn_metrics.update(
            {
                "status": "pass",
                "task": task_id,
                "task_display_name": TIER2_TASK_SPECS[task_id]["name"],
                "suite_position": "unified_20_task_provenance",
                "model_family": "neural_mlp_multilabel",
                "input": TIER2_TASK_SPECS[task_id]["input"],
                "split": "single_episode_chronological",
                "num_windows": int(len(valid_idx)),
                "num_train_windows": int(len(train_idx)),
                "num_test_windows": int(len(test_idx)),
                "num_objects": int(len(vocab)),
                "primary_metric": "micro_f1",
                "primary_score": nn_metrics["micro_f1"],
                "history": result["history"],
                "device": result["device"],
            }
        )
        write_json(neural_dir / "metrics.json", nn_metrics)
        neural_metrics = nn_metrics
    return {"minimal": metrics, "neural_mlp": neural_metrics}


def regression_task(
    task_id: str,
    X_in: np.ndarray,
    Y: np.ndarray,
    rows: list[dict[str, str]],
    valid_idx: np.ndarray,
    out_dir: Path,
    args: argparse.Namespace,
    prediction_kind: str,
) -> dict[str, Any]:
    train_idx, test_idx = chronological_split(len(valid_idx), args.train_fraction)
    pred = ridge_predict(X_in[train_idx], Y[train_idx], X_in[test_idx], l2=args.ridge_l2, standardize_y=True)
    metrics = regression_metrics(Y[test_idx], pred)
    if task_id == "time_to_transition":
        metrics["mae_frames"] = metrics["mae"]
    metrics.update(
        {
            "status": "pass",
            "task": task_id,
            "task_display_name": TIER2_TASK_SPECS[task_id]["name"],
            "suite_position": "unified_20_task_provenance",
            "model_family": "minimal_ridge_regression",
            "input": TIER2_TASK_SPECS[task_id]["input"],
            "split": "single_episode_chronological",
            "num_windows": int(len(valid_idx)),
            "num_train_windows": int(len(train_idx)),
            "num_test_windows": int(len(test_idx)),
            "target_dim": int(Y.shape[1]),
            "primary_metric": TIER2_TASK_SPECS[task_id]["metric_key"],
            "primary_score": float(metrics[TIER2_TASK_SPECS[task_id]["metric_key"]]),
        }
    )
    task_dir = out_dir / task_id
    write_json(task_dir / "metrics.json", metrics)
    if prediction_kind == "csv_scalar":
        pred_rows = []
        for local_k, local_idx in enumerate(test_idx):
            global_idx = int(valid_idx[int(local_idx)])
            pred_rows.append(
                {
                    "window_index": rows[global_idx]["window_index"],
                    "start_frame": rows[global_idx]["start_frame"],
                    "end_frame": rows[global_idx]["end_frame"],
                    "true_value": float(Y[int(local_idx), 0]),
                    "pred_value": float(pred[local_k, 0]),
                    "absolute_error": float(abs(Y[int(local_idx), 0] - pred[local_k, 0])),
                }
            )
        write_csv(task_dir / "predictions.csv", pred_rows)
    else:
        np.savez_compressed(task_dir / "predictions.npz", y_true=Y[test_idx], y_pred=pred, test_window_indices=valid_idx[test_idx])

    neural_metrics = None
    if not args.skip_neural:
        neural_dir = out_dir / "neural_mlp" / task_id
        neural_dir.mkdir(parents=True, exist_ok=True)
        result = train_regressor(X_in, Y, train_idx, test_idx, neural_config(args))
        nn_pred = result["pred"]
        nn_metrics = regression_metrics(Y[test_idx], nn_pred)
        if task_id == "time_to_transition":
            nn_metrics["mae_frames"] = nn_metrics["mae"]
        nn_metrics.update(
            {
                "status": "pass",
                "task": task_id,
                "task_display_name": TIER2_TASK_SPECS[task_id]["name"],
                "suite_position": "unified_20_task_provenance",
                "model_family": "neural_mlp_regression",
                "input": TIER2_TASK_SPECS[task_id]["input"],
                "split": "single_episode_chronological",
                "num_windows": int(len(valid_idx)),
                "num_train_windows": int(len(train_idx)),
                "num_test_windows": int(len(test_idx)),
                "target_dim": int(Y.shape[1]),
                "primary_metric": TIER2_TASK_SPECS[task_id]["metric_key"],
                "primary_score": float(nn_metrics[TIER2_TASK_SPECS[task_id]["metric_key"]]),
                "history": result["history"],
                "device": result["device"],
            }
        )
        write_json(neural_dir / "metrics.json", nn_metrics)
        neural_metrics = nn_metrics
    return {"minimal": metrics, "neural_mlp": neural_metrics}


def retrieval_task(
    X: np.ndarray,
    rows: list[dict[str, str]],
    manifest: list[dict[str, Any]],
    out_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    task_id = "camera_view_sync_retrieval"
    query_idx = block_indices(manifest, include=["video_fisheye_cam1"])
    target_idx = block_indices(manifest, include=["video_fisheye_cam3"])
    train_idx, test_idx = chronological_split(len(X), args.train_fraction)
    pred = ridge_predict(X[train_idx][:, query_idx], X[train_idx][:, target_idx], X[test_idx][:, query_idx], l2=args.ridge_l2, standardize_y=True)
    min_metrics, rank_rows = retrieval_metrics(pred, X[test_idx][:, target_idx])
    min_metrics.update(
        {
            "status": "pass",
            "task": task_id,
            "task_display_name": TIER2_TASK_SPECS[task_id]["name"],
            "suite_position": "unified_20_task_provenance",
            "model_family": "minimal_ridge_projection_cosine_retrieval",
            "input": TIER2_TASK_SPECS[task_id]["input"],
            "split": "single_episode_chronological",
            "num_train_windows": int(len(train_idx)),
            "num_test_windows": int(len(test_idx)),
            "query_dim": int(len(query_idx)),
            "target_dim": int(len(target_idx)),
            "primary_metric": "mrr",
            "primary_score": min_metrics["mrr"],
        }
    )
    for row in rank_rows:
        idx = int(test_idx[int(row["test_position"])])
        row["window_index"] = rows[idx]["window_index"]
        row["center_frame"] = rows[idx]["center_frame"]
    task_dir = out_dir / task_id
    write_json(task_dir / "metrics.json", min_metrics)
    write_csv(task_dir / "ranks.csv", rank_rows)

    neural_metrics = None
    if not args.skip_neural:
        neural_dir = out_dir / "neural_mlp" / task_id
        neural_dir.mkdir(parents=True, exist_ok=True)
        result = train_regressor(X[:, query_idx], X[:, target_idx], train_idx, test_idx, neural_config(args))
        nn_metrics, nn_rows = retrieval_metrics(result["pred"], X[test_idx][:, target_idx])
        nn_metrics.update(
            {
                "status": "pass",
                "task": task_id,
                "task_display_name": TIER2_TASK_SPECS[task_id]["name"],
                "suite_position": "unified_20_task_provenance",
                "model_family": "neural_mlp_projection_cosine_retrieval",
                "input": TIER2_TASK_SPECS[task_id]["input"],
                "split": "single_episode_chronological",
                "num_train_windows": int(len(train_idx)),
                "num_test_windows": int(len(test_idx)),
                "query_dim": int(len(query_idx)),
                "target_dim": int(len(target_idx)),
                "primary_metric": "mrr",
                "primary_score": nn_metrics["mrr"],
                "history": result["history"],
                "device": result["device"],
            }
        )
        write_json(neural_dir / "metrics.json", nn_metrics)
        neural_metrics = nn_metrics
    return {"minimal": min_metrics, "neural_mlp": neural_metrics}


def transition_targets(rows: list[dict[str, str]], frame_info: dict[int, dict[str, Any]], n_frames: int, cap_frames: int) -> np.ndarray:
    per_frame = [frame_label(frame_info.get(frame, {}), "action") for frame in range(n_frames)]
    boundaries = [frame for frame in range(1, n_frames) if per_frame[frame] and per_frame[frame - 1] and per_frame[frame] != per_frame[frame - 1]]
    targets = []
    for row in rows:
        center = int(row["center_frame"])
        future = [boundary for boundary in boundaries if boundary >= center]
        delta = float((future[0] - center) if future else (n_frames - 1 - center))
        targets.append(min(delta, float(cap_frames)))
    return np.asarray(targets, dtype=np.float32)[:, None]


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    X, starts, ends, rows, manifest = load_windows(args.results_dir)
    ann = load_annotation(args)
    frame_info = ann["caption_frame_info_map"]
    n_frames = len(ann["img_names"])
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    non_caption_idx = block_indices(manifest, exclude=["caption_objects_interaction_text"])
    horizon = args.future_windows * args.stride_frames

    future_action_labels = []
    future_subtask_labels = []
    interaction_labels = []
    action_object_labels = []
    for row in rows:
        future_start = int(row["start_frame"]) + horizon
        future_end = int(row["end_frame"]) + horizon
        if future_end < n_frames:
            future_action_labels.append(majority_window_label(frame_info, future_start, future_end, lambda info: frame_label(info, "action")))
            future_subtask_labels.append(majority_window_label(frame_info, future_start, future_end, lambda info: frame_label(info, "subtask")))
        else:
            future_action_labels.append("")
            future_subtask_labels.append("")
        start = int(row["start_frame"])
        end = int(row["end_frame"])
        interaction_labels.append(majority_window_label(frame_info, start, end, interaction_label, min_fraction=0.20))
        objects = window_objects(frame_info, start, end)
        action = str(row.get("action_label", "") or "").strip()
        action_object_labels.append(f"{action} :: {' | '.join(objects)}" if action and objects else "")

    tasks: OrderedDict[str, dict[str, Any]] = OrderedDict()
    tasks["long_horizon_next_action"] = softmax_classification(
        "long_horizon_next_action",
        X[:, non_caption_idx],
        future_action_labels,
        rows,
        TIER2_TASK_SPECS["long_horizon_next_action"]["input"],
        out_dir,
        args,
    )
    tasks["next_subtask_forecast"] = softmax_classification(
        "next_subtask_forecast",
        X[:, non_caption_idx],
        future_subtask_labels,
        rows,
        TIER2_TASK_SPECS["next_subtask_forecast"]["input"],
        out_dir,
        args,
    )
    tasks["interaction_text_prediction"] = softmax_classification(
        "interaction_text_prediction",
        X[:, non_caption_idx],
        interaction_labels,
        rows,
        TIER2_TASK_SPECS["interaction_text_prediction"]["input"],
        out_dir,
        args,
    )
    tasks["action_object_relation"] = softmax_classification(
        "action_object_relation",
        X[:, non_caption_idx],
        action_object_labels,
        rows,
        TIER2_TASK_SPECS["action_object_relation"]["input"],
        out_dir,
        args,
    )
    tasks["object_set_forecast"] = object_set_forecast(X, rows, manifest, frame_info, out_dir, args)

    imu_idx = block_indices(manifest, include=["imu_accel_gyro"])
    hand_idx = block_indices(manifest, include=["hand_left_joints", "hand_right_joints"])
    all_valid = np.arange(len(X), dtype=np.int64)
    tasks["imu_to_hand_pose"] = regression_task(
        "imu_to_hand_pose",
        X[:, imu_idx],
        X[:, hand_idx],
        rows,
        all_valid,
        out_dir,
        args,
        "npz",
    )
    tasks["camera_view_sync_retrieval"] = retrieval_task(X, rows, manifest, out_dir, args)
    time_targets = transition_targets(rows, frame_info, n_frames, args.transition_cap_frames)
    tasks["time_to_transition"] = regression_task(
        "time_to_transition",
        X[:, non_caption_idx],
        time_targets,
        rows,
        all_valid,
        out_dir,
        args,
        "csv_scalar",
    )

    payload = {
        "title": "Ropedia Xperience-10M Unified 20-Task Provenance Bundle",
        "status": "pass",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "suite_position": "unified_20_task_provenance",
        "legacy_path_note": "The tier2_task_suite file and directory names are retained for stable public links; these tasks are part of the unified 20-task suite, not a separate public tier.",
        "unified_task_integration": {
            "total_task_count": 20,
            "legacy_provenance_row_count": len(TIER2_TASK_SPECS),
            "shared_metrics": "docs/data/summary_metrics.json",
            "unified_protocol": "docs/data/evaluation_protocol.json",
        },
        "dataset_scope": {
            "sample_episode_count": 1,
            "num_frames": int(n_frames),
            "num_windows": int(len(X)),
            "feature_dim": int(X.shape[1]),
            "window_frames": 20,
            "stride_frames": int(args.stride_frames),
            "future_horizon_windows": int(args.future_windows),
            "future_horizon_frames": int(horizon),
            "future_horizon_seconds_at_20fps": horizon / 20.0,
            "transition_target_cap_frames": int(args.transition_cap_frames),
            "transition_target_cap_seconds_at_20fps": args.transition_cap_frames / 20.0,
            "split_policy": "single_episode_chronological_70_30",
            "raw_hdf5_required_to_regenerate": True,
            "raw_data_redistributed": False,
        },
        "setup_alignment": {
            "same_window_unit_as_unified_suite": True,
            "same_feature_manifest_as_unified_suite": "results/episode_task_suite/feature_manifest.json",
            "same_shared_tensor_as_unified_suite": "results/episode_task_suite/shared_windows.npz",
            "minimal_baselines": "softmax, ridge regression/projection, and ridge multilabel heads",
            "neural_baselines": "compact one-hidden-layer/two-layer PyTorch MLP heads with the same chronological split",
            "leakage_policy": "Caption-derived text features are removed whenever the target is a label, object, relation, interaction phrase, or future semantic state.",
        },
        "source_files": [
            "results/episode_task_suite/shared_windows.npz",
            "results/episode_task_suite/windows.csv",
            "results/episode_task_suite/feature_manifest.json",
            "data/sample/xperience-10m-sample/annotation.hdf5",
        ],
        "task_specs": TIER2_TASK_SPECS,
        "tasks": tasks,
    }
    return payload


def format_metric(value: float | None, metric_key: str) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def write_markdown(payload: dict[str, Any], output_dir: Path) -> None:
    lines = [
        "# Unified 20-Task Provenance Baselines",
        "",
        "This historical result bundle is part of the unified 20-task public-sample suite. The rows here reuse the same 20-frame windows, 5-frame stride, shared feature tensor, chronological split, and minimal/neural baseline discipline as the rest of the suite.",
        "",
        "The file and directory names still contain `tier2_task_suite` for backwards-compatible public links, but this is not a separate benchmark tier.",
        "",
        "## Setup Alignment",
        "",
        f"- Unified task contracts: `{payload['unified_task_integration']['total_task_count']}`",
        f"- Provenance rows in this historical bundle: `{payload['unified_task_integration']['legacy_provenance_row_count']}`",
        f"- Long-horizon offset: `{payload['dataset_scope']['future_horizon_frames']}` frames, about `{payload['dataset_scope']['future_horizon_seconds_at_20fps']:.1f}` seconds at 20 FPS",
        "- Raw public-sample HDF5 is required to regenerate the interaction/object targets; raw media/HDF5 files are not redistributed.",
        "",
        "## Results",
        "",
        "| # | Task | Input | Output | Minimal | Neural MLP | Meaning |",
        "| ---: | --- | --- | --- | ---: | ---: | --- |",
    ]
    for offset, (task_id, spec) in enumerate(payload["task_specs"].items(), start=13):
        result = payload["tasks"][task_id]
        key = spec["metric_key"]
        lines.append(
            f"| {offset} | {spec['name']} | {spec['input']} | {spec['target']} | {format_metric(metric_value(task_id, result.get('minimal')), key)} {spec['metric_name']} | {format_metric(metric_value(task_id, result.get('neural_mlp')), key)} {spec['metric_name']} | {spec['meaning']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "These sample-level baselines are part of the same unified public-sample suite. They prove that the sample can support richer task contracts, but they do not prove cross-episode model quality.",
            "",
        ]
    )
    (output_dir / "TIER2_TASK_BASELINES.md").write_text("\n".join(lines), encoding="utf-8")


def write_svg(payload: dict[str, Any]) -> None:
    CHARTS.mkdir(parents=True, exist_ok=True)
    width = 1440
    row_h = 76
    top = 128
    height = top + row_h * len(TIER2_TASK_SPECS) + 96
    rows = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#020502"/>',
        '<rect x="32" y="32" width="1376" height="{}" rx="12" fill="#071207" stroke="#ccffa0" stroke-opacity="0.22"/>'.format(height - 64),
        '<text x="72" y="82" fill="#f4f8ef" font-size="32" font-weight="760">Xperience-10M unified 20-task provenance</text>',
        '<text x="72" y="112" fill="#a5afa2" font-size="16">Historical bundle rows retained for stable links inside the same 20-task suite, aligned with the same 20-frame window, 5-frame stride, and chronological split.</text>',
    ]
    colors = ["#ccffa0", "#7ae5c3", "#9bdfff", "#d8f4a5"]
    for idx, (task_id, spec) in enumerate(TIER2_TASK_SPECS.items()):
        y = top + idx * row_h
        color = colors[idx % len(colors)]
        result = payload["tasks"][task_id]
        min_value = metric_value(task_id, result.get("minimal"))
        nn_value = metric_value(task_id, result.get("neural_mlp"))
        metric = spec["metric_name"]
        rows.extend(
            [
                f'<rect x="72" y="{y}" width="1296" height="58" rx="8" fill="#020902" stroke="#ccffa0" stroke-opacity="0.14"/>',
                f'<rect x="72" y="{y}" width="8" height="58" rx="4" fill="{color}"/>',
                f'<text x="98" y="{y + 24}" fill="#f4f8ef" font-size="18" font-weight="720">{spec["name"]}</text>',
                f'<text x="98" y="{y + 47}" fill="#a5afa2" font-size="13">{spec["family"]} · {spec["target"]}</text>',
                f'<text x="840" y="{y + 24}" fill="#f4f8ef" font-size="16" font-weight="700">minimal {format_metric(min_value, spec["metric_key"])} {metric}</text>',
                f'<text x="1110" y="{y + 24}" fill="#f4f8ef" font-size="16" font-weight="700">neural {format_metric(nn_value, spec["metric_key"])} {metric}</text>',
            ]
        )
    rows.append("</svg>")
    (CHARTS / "tier2_task_suite.svg").write_text("\n".join(rows), encoding="utf-8")


def main() -> int:
    args = parse_args()
    payload = build_payload(args)
    write_json(args.output_dir / "tier2_task_suite_results.json", payload)
    write_json(DOCS_DATA / "tier2_task_suite.json", payload)
    write_markdown(payload, args.output_dir)
    write_svg(payload)
    print(f"Wrote {args.output_dir / 'tier2_task_suite_results.json'}")
    print(f"Wrote {DOCS_DATA / 'tier2_task_suite.json'}")
    print(f"Wrote {CHARTS / 'tier2_task_suite.svg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

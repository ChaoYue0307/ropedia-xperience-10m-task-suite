#!/usr/bin/env python3
"""Run 128-episode raw-feature baselines for the unified 20-task suite.

This runner is intended for owner-side staged feature workspaces where the
exported 4430-dim sensor feature NPZ shards are present.  It complements
run_128_task_baselines.py, which is public-metadata-only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from neural_task_models import NeuralConfig, save_torch_model, train_classifier, train_multilabel, train_regressor
from task_display import task_display_name
from train_min_action_model import compute_metrics, fit_scaler


TASKS: list[str] = [
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
    "long_horizon_next_action",
    "next_subtask_forecast",
    "interaction_text_prediction",
    "action_object_relation",
    "object_set_forecast",
    "imu_to_hand_pose",
    "camera_view_sync_retrieval",
    "time_to_transition",
]

CLASS_FIELD_TASKS = {
    "timeline_action": "action",
    "timeline_subtask": "subtask",
    "transition_detection": "transition",
    "next_action": "next_action",
    "contact_prediction": "contact",
}

TASK_META: dict[str, dict[str, str]] = {
    "timeline_action": {"family": "classification", "metric": "macro_f1", "direction": "higher"},
    "timeline_subtask": {"family": "classification", "metric": "macro_f1", "direction": "higher"},
    "transition_detection": {"family": "classification", "metric": "macro_f1", "direction": "higher"},
    "next_action": {"family": "classification", "metric": "macro_f1", "direction": "higher"},
    "hand_trajectory_forecast": {"family": "regression", "metric": "mae", "direction": "lower"},
    "contact_prediction": {"family": "classification", "metric": "macro_f1", "direction": "higher"},
    "object_relevance": {"family": "multi_label", "metric": "micro_f1", "direction": "higher"},
    "caption_grounding": {"family": "retrieval", "metric": "mrr", "direction": "higher"},
    "cross_modal_retrieval": {"family": "retrieval", "metric": "mrr", "direction": "higher"},
    "modality_reconstruction": {"family": "regression", "metric": "r2", "direction": "higher"},
    "temporal_order": {"family": "classification", "metric": "macro_f1", "direction": "higher"},
    "misalignment_detection": {"family": "classification", "metric": "macro_f1", "direction": "higher"},
    "long_horizon_next_action": {"family": "classification", "metric": "macro_f1", "direction": "higher"},
    "next_subtask_forecast": {"family": "classification", "metric": "macro_f1", "direction": "higher"},
    "interaction_text_prediction": {"family": "classification", "metric": "macro_f1", "direction": "higher"},
    "action_object_relation": {"family": "classification", "metric": "macro_f1", "direction": "higher"},
    "object_set_forecast": {"family": "multi_label", "metric": "micro_f1", "direction": "higher"},
    "imu_to_hand_pose": {"family": "regression", "metric": "mae", "direction": "lower"},
    "camera_view_sync_retrieval": {"family": "retrieval", "metric": "mrr", "direction": "higher"},
    "time_to_transition": {"family": "regression", "metric": "mae", "direction": "lower"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-jsonl",
        type=Path,
        default=ROOT
        / "results/omni_finetune/xperience10m_qwen3_omni_128ep_multiscale_cap96_v5_full8gpu_lora_dataset/dataset.jsonl",
    )
    parser.add_argument("--feature-manifest-json", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/omni_finetune/a100_128_raw20_task_baselines")
    parser.add_argument("--tasks", default="all", help="Comma-separated task IDs or all.")
    parser.add_argument("--remote-prefix", default="/home/cy/Ropedia/ropedia-episode-task-suite")
    parser.add_argument("--local-prefix", default=str(ROOT))
    parser.add_argument("--max-input-dim", type=int, default=2048)
    parser.add_argument("--ridge-l2", type=float, default=10.0)
    parser.add_argument("--future-frames", type=int, default=100)
    parser.add_argument("--forecast-frames", type=int, default=20)
    parser.add_argument("--misalignment-shift", type=int, default=8)
    parser.add_argument("--transition-cap-frames", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-object-vocab", type=int, default=256)
    parser.add_argument("--skip-neural", action="store_true")
    parser.add_argument(
        "--compact-proxy-missing-tasks",
        action="store_true",
        help=(
            "Complete task axes whose raw 128-export fields are absent with documented compact-feature proxies. "
            "Task 15 uses the dominant hashed caption/object/interaction bin as the target; task 19 uses "
            "camera-pose-to-depth/audio same-window retrieval when paired video-view blocks are absent."
        ),
    )
    parser.add_argument("--neural-epochs", type=int, default=25)
    parser.add_argument("--neural-hidden-dim", type=int, default=128)
    parser.add_argument("--neural-batch-size", type=int, default=256)
    parser.add_argument("--neural-learning-rate", type=float, default=1e-3)
    parser.add_argument("--neural-weight-decay", type=float, default=1e-4)
    parser.add_argument("--neural-dropout", type=float, default=0.10)
    parser.add_argument("--neural-device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-neural-classes", type=int, default=512)
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[raw20] {message}", file=sys.stderr, flush=True)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


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


def selected_tasks(spec: str) -> list[str]:
    if spec.strip().lower() == "all":
        return TASKS
    chosen = [item.strip() for item in spec.split(",") if item.strip()]
    unknown = [item for item in chosen if item not in TASKS]
    if unknown:
        raise ValueError(f"Unknown tasks {unknown}; valid tasks are {TASKS}")
    return chosen


def answer(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("answer_json")
    return value if isinstance(value, dict) else {}


def normalize_label(value: Any) -> str:
    return str(value or "").strip()


def objects_from_answer(row: dict[str, Any]) -> list[str]:
    values = answer(row).get("objects", [])
    if not isinstance(values, list):
        return []
    cleaned = sorted({normalize_label(value).lower() for value in values if normalize_label(value)})
    return cleaned


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append(
                {
                    "id": row.get("id"),
                    "episode_id": str(row.get("episode_id")),
                    "split": str(row.get("split")),
                    "center_window": row.get("center_window") or {},
                    "sensor_feature_path": row.get("sensor_feature_path"),
                    "sensor_feature_index": int(row.get("sensor_feature_index", 0) or 0),
                    "sensor_feature_dim": int(row.get("sensor_feature_dim", 0) or 0),
                    "answer_json": row.get("answer_json") if isinstance(row.get("answer_json"), dict) else {},
                    "scale_id": row.get("scale_id"),
                }
            )
    return rows


def resolve_feature_path(path_value: str, remote_prefix: str, local_prefix: str) -> Path:
    path = Path(path_value)
    if path.exists():
        return path
    rewritten = str(path)
    if remote_prefix and rewritten.startswith(remote_prefix):
        rewritten = local_prefix.rstrip("/") + rewritten[len(remote_prefix) :]
    return Path(rewritten)


def load_feature_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    manifest = payload.get("feature_manifest", payload)
    if isinstance(manifest, list) and manifest and isinstance(manifest[0], dict) and "feature_manifest" in manifest[0]:
        manifest = manifest[0]["feature_manifest"]
    if not isinstance(manifest, list):
        raise ValueError(f"Could not parse feature manifest from {path}")
    return manifest


def default_manifest_path(dataset_jsonl: Path) -> Path:
    own = dataset_jsonl.parent / "dataset_manifest.json"
    if own.exists():
        payload = json.loads(own.read_text(encoding="utf-8"))
        manifest = payload.get("feature_manifest")
        if isinstance(manifest, list) and manifest:
            first = manifest[0]
            if isinstance(first, dict) and ("start" in first or "feature_manifest" in first):
                return own
    dense = dataset_jsonl.parent.parent / f"{dataset_jsonl.parent.name}_dense_20f_stride10" / "dataset_manifest.json"
    return dense if dense.exists() else own


def load_feature_matrix(rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    groups: OrderedDict[Path, list[tuple[int, int]]] = OrderedDict()
    missing_paths: list[str] = []
    for row_idx, row in enumerate(rows):
        path_value = row.get("sensor_feature_path")
        if not path_value:
            missing_paths.append(f"row:{row_idx}:empty")
            continue
        path = resolve_feature_path(str(path_value), args.remote_prefix, args.local_prefix)
        if not path.exists():
            missing_paths.append(str(path))
            continue
        groups.setdefault(path, []).append((row_idx, int(row.get("sensor_feature_index", 0) or 0)))

    if not groups:
        raise FileNotFoundError("No sensor feature NPZ files resolved from dataset rows.")

    first_path = next(iter(groups))
    with np.load(first_path, allow_pickle=False) as z:
        dim = int(np.asarray(z["features"]).shape[1])
    X = np.zeros((len(rows), dim), dtype=np.float32)
    valid = np.zeros(len(rows), dtype=bool)
    total_loaded = 0
    for path, items in groups.items():
        with np.load(path, allow_pickle=False) as z:
            features = np.asarray(z["features"], dtype=np.float32)
            for row_idx, feature_idx in items:
                if 0 <= feature_idx < len(features):
                    X[row_idx] = np.nan_to_num(features[feature_idx], nan=0.0, posinf=0.0, neginf=0.0)
                    valid[row_idx] = True
                    total_loaded += 1
    keep = np.flatnonzero(valid)
    kept_rows = [rows[int(i)] for i in keep]
    report = {
        "resolved_npz_files": len(groups),
        "loaded_feature_rows": int(total_loaded),
        "input_rows": len(rows),
        "dropped_rows": int(len(rows) - len(keep)),
        "missing_path_examples": missing_paths[:10],
        "feature_dim": dim,
    }
    return X[keep], kept_rows, report


def split_indices(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    out: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for idx, row in enumerate(rows):
        split = str(row.get("split"))
        if split in out:
            out[split].append(idx)
    return {key: np.asarray(value, dtype=np.int64) for key, value in out.items()}


def block_indices(manifest: list[dict[str, Any]], include: list[str] | None = None, exclude: list[str] | None = None) -> np.ndarray:
    include = include or []
    exclude = exclude or []
    indices: list[int] = []
    for block in manifest:
        name = str(block.get("name", ""))
        if include and not any(name == item or name.startswith(item) for item in include):
            continue
        if exclude and any(name == item or name.startswith(item) for item in exclude):
            continue
        indices.extend(range(int(block["start"]), int(block["end"])))
    return np.asarray(indices, dtype=np.int64)


def cap_columns(X: np.ndarray, max_dim: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if max_dim <= 0 or X.shape[1] <= max_dim:
        return X.astype(np.float32, copy=False), np.arange(X.shape[1], dtype=np.int64)
    rng = np.random.default_rng(seed)
    cols = np.sort(rng.choice(X.shape[1], size=max_dim, replace=False)).astype(np.int64)
    return X[:, cols].astype(np.float32, copy=False), cols


def row_start(row: dict[str, Any]) -> int:
    return int((row.get("center_window") or {}).get("start_frame", 0) or 0)


def row_end(row: dict[str, Any]) -> int:
    return int((row.get("center_window") or {}).get("end_frame", row_start(row)) or row_start(row))


def by_episode_sorted(rows: list[dict[str, Any]]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        grouped.setdefault(str(row.get("episode_id")), []).append(idx)
    for episode_id in grouped:
        grouped[episode_id].sort(key=lambda i: row_start(rows[i]))
    return grouped


def future_index_map(rows: list[dict[str, Any]], frame_offset: int) -> dict[int, int]:
    grouped = by_episode_sorted(rows)
    mapping: dict[int, int] = {}
    for indices in grouped.values():
        starts = np.asarray([row_start(rows[i]) for i in indices], dtype=np.int64)
        for pos, idx in enumerate(indices):
            target_start = row_start(rows[idx]) + frame_offset
            future_pos = int(np.searchsorted(starts, target_start, side="left"))
            if future_pos < len(indices):
                mapping[idx] = indices[future_pos]
    return mapping


def encode_train_first(labels: list[str], train_idx: np.ndarray) -> tuple[np.ndarray, list[str], int]:
    seen: OrderedDict[str, int] = OrderedDict()
    for idx in train_idx:
        label = labels[int(idx)]
        if label not in seen:
            seen[label] = len(seen)
    train_classes = len(seen)
    for label in labels:
        if label not in seen:
            seen[label] = len(seen)
    y = np.asarray([seen[label] for label in labels], dtype=np.int64)
    return y, list(seen.keys()), train_classes


def fit_centroids(X: np.ndarray, y: np.ndarray, train_idx: np.ndarray, train_classes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean, std = fit_scaler(X[train_idx])
    Xs = (X - mean) / std
    centroids = []
    priors = []
    for class_id in range(train_classes):
        idx = train_idx[y[train_idx] == class_id]
        priors.append(float(len(idx)))
        if len(idx):
            centroids.append(Xs[idx].mean(axis=0).astype(np.float32))
        else:
            centroids.append(np.zeros(X.shape[1], dtype=np.float32))
    centroids_arr = np.stack(centroids).astype(np.float32)
    centroids_arr /= np.maximum(np.linalg.norm(centroids_arr, axis=1, keepdims=True), 1e-6)
    priors_arr = np.asarray(priors, dtype=np.float32)
    priors_arr /= max(float(priors_arr.sum()), 1.0)
    return mean, std, centroids_arr, priors_arr


def predict_centroids(X: np.ndarray, mean: np.ndarray, std: np.ndarray, centroids: np.ndarray, priors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Xs = (X - mean) / std
    Xs /= np.maximum(np.linalg.norm(Xs, axis=1, keepdims=True), 1e-6)
    scores = Xs @ centroids.T
    if len(priors):
        scores += 0.03 * np.log(np.maximum(priors, 1e-9))[None, :]
    scores -= scores.max(axis=1, keepdims=True)
    prob = np.exp(scores)
    prob /= np.maximum(prob.sum(axis=1, keepdims=True), 1e-12)
    return np.argmax(prob, axis=1).astype(np.int64), prob.astype(np.float32)


def classification_metrics(y: np.ndarray, pred: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    metrics, per_class, confusion = compute_metrics(y, pred, class_names)
    return {"metrics": metrics, "per_class": per_class, "confusion": confusion}


def write_class_predictions(path: Path, rows: list[dict[str, Any]], idx: np.ndarray, y: np.ndarray, pred: np.ndarray, prob: np.ndarray, class_names: list[str]) -> None:
    out = []
    for local_pos, row_idx in enumerate(idx):
        pred_id = int(pred[local_pos])
        true_id = int(y[int(row_idx)])
        out.append(
            {
                "id": rows[int(row_idx)].get("id"),
                "episode_id": rows[int(row_idx)].get("episode_id"),
                "split": rows[int(row_idx)].get("split"),
                "start_frame": row_start(rows[int(row_idx)]),
                "end_frame": row_end(rows[int(row_idx)]),
                "true_label": class_names[true_id],
                "predicted_label": class_names[pred_id],
                "confidence": float(prob[local_pos, pred_id]) if prob.size else None,
                "correct": int(pred_id == true_id),
            }
        )
    write_csv(path, out)


def run_simple_classification(
    task: str,
    X: np.ndarray,
    rows: list[dict[str, Any]],
    splits: dict[str, np.ndarray],
    labels: list[str],
    out_root: Path,
    args: argparse.Namespace,
    input_description: str,
) -> dict[str, Any]:
    valid = np.asarray([bool(label) for label in labels])
    keep = np.flatnonzero(valid)
    local = {int(global_idx): local_idx for local_idx, global_idx in enumerate(keep)}
    Xv = X[keep]
    labels_v = [labels[int(i)] for i in keep]
    split_local = {
        split: np.asarray([local[int(i)] for i in idx if int(i) in local], dtype=np.int64)
        for split, idx in splits.items()
    }
    train_idx = split_local["train"]
    test_idx = split_local["test"]
    val_idx = split_local["val"]
    if len(train_idx) == 0 or len(test_idx) == 0:
        return write_unsupported(task, out_root, "simple_raw128_centroid", "empty train/test split after filtering", TASK_META[task]["metric"])
    Xc, cols = cap_columns(Xv, args.max_input_dim, args.seed + stable_task_seed(task))
    y, class_names, train_classes = encode_train_first(labels_v, train_idx)
    mean, std, centroids, priors = fit_centroids(Xc, y, train_idx, train_classes)
    split_metrics = {}
    test_pred = None
    test_prob = None
    for split, idx in [("val", val_idx), ("test", test_idx)]:
        pred, prob = predict_centroids(Xc[idx], mean, std, centroids, priors)
        eval_payload = classification_metrics(y[idx], pred, class_names)
        split_metrics[split] = eval_payload["metrics"]
        if split == "test":
            test_pred, test_prob = pred, prob
    out_dir = out_root / "simple_raw128" / task
    out_dir.mkdir(parents=True, exist_ok=True)
    if test_pred is not None and test_prob is not None:
        write_class_predictions(out_dir / "predictions.csv", rows_for_keep(rows, keep), test_idx, y, test_pred, test_prob, class_names)
    metrics = base_metrics(task, "simple_raw128_centroid", TASK_META[task]["metric"], input_description)
    metrics.update(
        {
            "status": "pass",
            "num_train_windows": int(len(train_idx)),
            "num_val_windows": int(len(val_idx)),
            "num_test_windows": int(len(test_idx)),
            "num_classes": int(len(class_names)),
            "num_train_classes": int(train_classes),
            "input_dim": int(Xv.shape[1]),
            "fit_input_dim": int(Xc.shape[1]),
            "selected_column_count": int(len(cols)),
            "splits": split_metrics,
            "primary_score": split_metrics.get("test", {}).get(TASK_META[task]["metric"]),
        }
    )
    write_json(out_dir / "metrics.json", metrics)
    np.savez_compressed(out_dir / "model.npz", mean=mean, std=std, centroids=centroids, priors=priors, class_names=np.asarray(class_names, dtype=object), cols=cols)
    return metrics


def rows_for_keep(rows: list[dict[str, Any]], keep: np.ndarray) -> list[dict[str, Any]]:
    return [rows[int(i)] for i in keep]


def stable_task_seed(task: str) -> int:
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(task)) % 100_000


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


def run_neural_classification(
    task: str,
    X: np.ndarray,
    rows: list[dict[str, Any]],
    splits: dict[str, np.ndarray],
    labels: list[str],
    out_root: Path,
    args: argparse.Namespace,
    input_description: str,
) -> dict[str, Any] | None:
    if args.skip_neural:
        return None
    valid = np.asarray([bool(label) for label in labels])
    keep = np.flatnonzero(valid)
    local = {int(global_idx): local_idx for local_idx, global_idx in enumerate(keep)}
    Xv = X[keep]
    labels_v = [labels[int(i)] for i in keep]
    split_local = {
        split: np.asarray([local[int(i)] for i in idx if int(i) in local], dtype=np.int64)
        for split, idx in splits.items()
    }
    train_idx = split_local["train"]
    test_idx = split_local["test"]
    if len(train_idx) == 0 or len(test_idx) == 0:
        return None
    Xc, cols = cap_columns(Xv, args.max_input_dim, args.seed + 10_000 + stable_task_seed(task))
    y, class_names, train_classes = encode_train_first(labels_v, train_idx)
    out_dir = out_root / "neural_mlp_raw128" / task
    if train_classes > args.max_neural_classes:
        return write_unsupported(
            task,
            out_root,
            "neural_mlp_raw128",
            f"train class count {train_classes} exceeds --max-neural-classes {args.max_neural_classes}",
            TASK_META[task]["metric"],
            subdir="neural_mlp_raw128",
        )
    try:
        result = train_classifier(Xc, y, train_idx, test_idx, train_classes, neural_config(args))
        eval_payload = classification_metrics(y[test_idx], result["pred"], class_names)
        metrics = base_metrics(task, "neural_mlp_raw128", TASK_META[task]["metric"], input_description)
        metrics.update(
            {
                "status": "pass",
                "device": result.get("device"),
                "history": result.get("history", []),
                "num_train_windows": int(len(train_idx)),
                "num_test_windows": int(len(test_idx)),
                "num_classes": int(len(class_names)),
                "num_train_classes": int(train_classes),
                "input_dim": int(Xv.shape[1]),
                "fit_input_dim": int(Xc.shape[1]),
                "selected_column_count": int(len(cols)),
                "splits": {"test": eval_payload["metrics"]},
                "primary_score": eval_payload["metrics"].get(TASK_META[task]["metric"]),
            }
        )
        write_json(out_dir / "metrics.json", metrics)
        write_class_predictions(out_dir / "predictions.csv", rows_for_keep(rows, keep), test_idx, y, result["pred"], result["prob"], class_names)
        save_torch_model(
            out_dir / "model.pt",
            {
                "state_dict": result["state_dict"],
                "mean": result["mean"],
                "std": result["std"],
                "class_names": class_names,
                "selected_columns": cols,
                "metrics": metrics,
            },
        )
        return metrics
    except Exception as exc:
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics = base_metrics(task, "neural_mlp_raw128", TASK_META[task]["metric"], input_description)
        metrics.update({"status": "failed", "error": str(exc), "primary_score": None})
        write_json(out_dir / "metrics.json", metrics)
        return metrics


def base_metrics(task: str, family: str, primary_metric: str, input_description: str) -> dict[str, Any]:
    meta = TASK_META[task]
    return {
        "task": task,
        "task_display_name": task_display_name(task),
        "task_family": meta["family"],
        "model_family": family,
        "source": "128_episode_raw_sensor_features",
        "input_features": input_description,
        "primary_metric": primary_metric,
        "metric_direction": meta["direction"],
    }


def write_unsupported(
    task: str,
    out_root: Path,
    family: str,
    reason: str,
    primary_metric: str,
    *,
    subdir: str | None = None,
) -> dict[str, Any]:
    metrics = base_metrics(task, family, primary_metric, "not run")
    metrics.update({"status": "unsupported", "reason": reason, "primary_score": None})
    out_dir = out_root / (subdir or family) / task
    write_json(out_dir / "metrics.json", metrics)
    return metrics


def metrics_output_path(out_root: Path, item: dict[str, Any]) -> Path:
    family = str(item.get("model_family", ""))
    subdir = "neural_mlp_raw128" if family.startswith("neural_mlp_raw128") else "simple_raw128"
    return out_root / subdir / str(item.get("task")) / "metrics.json"


def annotate_metrics(out_root: Path, results: list[dict[str, Any]], **fields: Any) -> list[dict[str, Any]]:
    for item in results:
        item.update(fields)
        write_json(metrics_output_path(out_root, item), item)
    return results


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    denom = float(np.sum((y_true - y_true.mean(axis=0, keepdims=True)) ** 2))
    r2 = 1.0 - float(np.sum(err**2)) / max(denom, 1e-12)
    mean_l2 = float(np.mean(np.linalg.norm(err, axis=1)))
    return {"mae": mae, "rmse": rmse, "r2": r2, "mean_l2": mean_l2}


def standardize(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = X_train.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return ((X_train - mean) / std).astype(np.float32), ((X_test - mean) / std).astype(np.float32), mean, std


def ridge_predict(X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray, l2: float, standardize_y: bool) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    Xtr, Xte, x_mean, x_std = standardize(X_train.astype(np.float32), X_test.astype(np.float32))
    Y = np.asarray(Y_train, dtype=np.float32)
    if Y.ndim == 1:
        Y = Y[:, None]
    if standardize_y:
        y_mean = Y.mean(axis=0, dtype=np.float64).astype(np.float32)
        y_std = Y.std(axis=0, dtype=np.float64).astype(np.float32)
        y_std = np.where(y_std < 1e-6, 1.0, y_std).astype(np.float32)
        Y_work = ((Y - y_mean) / y_std).astype(np.float32)
    else:
        y_mean = np.zeros(Y.shape[1], dtype=np.float32)
        y_std = np.ones(Y.shape[1], dtype=np.float32)
        Y_work = Y
    eye = np.eye(Xtr.shape[1], dtype=np.float32) * float(l2)
    W = np.linalg.solve(Xtr.T @ Xtr + eye, Xtr.T @ Y_work).astype(np.float32)
    pred = Xte @ W
    pred = pred * y_std + y_mean
    return pred.astype(np.float32), {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std, "W": W}


def cosine_retrieval_metrics(query: np.ndarray, target: np.ndarray, chunk: int = 512) -> dict[str, float]:
    query_n = query / np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-6)
    target_n = target / np.maximum(np.linalg.norm(target, axis=1, keepdims=True), 1e-6)
    ranks: list[int] = []
    top1 = 0
    for start in range(0, len(query_n), chunk):
        end = min(start + chunk, len(query_n))
        sims = query_n[start:end] @ target_n.T
        for local, row in enumerate(sims):
            correct = start + local
            rank = int(np.sum(row > row[correct]) + 1)
            ranks.append(rank)
            top1 += int(rank == 1)
    ranks_arr = np.asarray(ranks, dtype=np.float32)
    return {
        "mrr": float(np.mean(1.0 / ranks_arr)),
        "top1": float(top1 / max(len(ranks), 1)),
        "median_rank": float(np.median(ranks_arr)) if len(ranks_arr) else None,
        "num_queries": int(len(ranks)),
    }


def multilabel_metrics(Y_true: np.ndarray, Y_score: np.ndarray, threshold: np.ndarray) -> dict[str, float]:
    pred = (Y_score >= threshold[None, :]).astype(np.int64)
    true = Y_true.astype(np.int64)
    tp = float(np.sum((pred == 1) & (true == 1)))
    fp = float(np.sum((pred == 1) & (true == 0)))
    fn = float(np.sum((pred == 0) & (true == 1)))
    micro_precision = tp / max(tp + fp, 1.0)
    micro_recall = tp / max(tp + fn, 1.0)
    micro_f1 = 2 * micro_precision * micro_recall / max(micro_precision + micro_recall, 1e-12)
    per_f1 = []
    for col in range(true.shape[1]):
        c_tp = float(np.sum((pred[:, col] == 1) & (true[:, col] == 1)))
        c_fp = float(np.sum((pred[:, col] == 1) & (true[:, col] == 0)))
        c_fn = float(np.sum((pred[:, col] == 0) & (true[:, col] == 1)))
        precision = c_tp / max(c_tp + c_fp, 1.0)
        recall = c_tp / max(c_tp + c_fn, 1.0)
        per_f1.append(2 * precision * recall / max(precision + recall, 1e-12))
    exact_match = float(np.mean(np.all(pred == true, axis=1)))
    return {"micro_f1": float(micro_f1), "macro_f1": float(np.mean(per_f1)), "exact_match": exact_match}


def run_regression_like(
    task: str,
    X_in: np.ndarray,
    Y: np.ndarray,
    rows: list[dict[str, Any]],
    splits: dict[str, np.ndarray],
    out_root: Path,
    args: argparse.Namespace,
    input_description: str,
    primary_metric: str,
    retrieval: bool = False,
) -> list[dict[str, Any]]:
    train_idx = splits["train"]
    test_idx = splits["test"]
    if len(train_idx) == 0 or len(test_idx) == 0:
        return [write_unsupported(task, out_root, "simple_raw128_ridge", "empty train/test split", primary_metric)]
    Xc, cols = cap_columns(X_in, args.max_input_dim, args.seed + stable_task_seed(task))
    out: list[dict[str, Any]] = []
    simple_dir = out_root / "simple_raw128" / task
    try:
        pred, model = ridge_predict(Xc[train_idx], Y[train_idx], Xc[test_idx], args.ridge_l2, standardize_y=not retrieval)
        split_metrics = cosine_retrieval_metrics(pred, Y[test_idx]) if retrieval else regression_metrics(Y[test_idx], pred)
        metrics = base_metrics(task, "simple_raw128_ridge", primary_metric, input_description)
        metrics.update(
            {
                "status": "pass",
                "num_train_windows": int(len(train_idx)),
                "num_test_windows": int(len(test_idx)),
                "input_dim": int(X_in.shape[1]),
                "fit_input_dim": int(Xc.shape[1]),
                "target_dim": int(Y.shape[1] if Y.ndim > 1 else 1),
                "splits": {"test": split_metrics},
                "primary_score": split_metrics.get(primary_metric),
            }
        )
        write_json(simple_dir / "metrics.json", metrics)
        np.savez_compressed(simple_dir / "model.npz", cols=cols, **model)
        out.append(metrics)
    except Exception as exc:
        metrics = base_metrics(task, "simple_raw128_ridge", primary_metric, input_description)
        metrics.update({"status": "failed", "error": str(exc), "primary_score": None})
        write_json(simple_dir / "metrics.json", metrics)
        out.append(metrics)

    if args.skip_neural:
        return out
    neural_dir = out_root / "neural_mlp_raw128" / task
    try:
        result = train_regressor(Xc, Y, train_idx, test_idx, neural_config(args))
        split_metrics = cosine_retrieval_metrics(result["pred"], Y[test_idx]) if retrieval else regression_metrics(Y[test_idx], result["pred"])
        metrics = base_metrics(task, "neural_mlp_raw128", primary_metric, input_description)
        metrics.update(
            {
                "status": "pass",
                "device": result.get("device"),
                "history": result.get("history", []),
                "num_train_windows": int(len(train_idx)),
                "num_test_windows": int(len(test_idx)),
                "input_dim": int(X_in.shape[1]),
                "fit_input_dim": int(Xc.shape[1]),
                "target_dim": int(Y.shape[1] if Y.ndim > 1 else 1),
                "splits": {"test": split_metrics},
                "primary_score": split_metrics.get(primary_metric),
            }
        )
        write_json(neural_dir / "metrics.json", metrics)
        save_torch_model(
            neural_dir / "model.pt",
            {
                "state_dict": result["state_dict"],
                "x_mean": result["x_mean"],
                "x_std": result["x_std"],
                "y_mean": result["y_mean"],
                "y_std": result["y_std"],
                "selected_columns": cols,
                "metrics": metrics,
            },
        )
        out.append(metrics)
    except Exception as exc:
        metrics = base_metrics(task, "neural_mlp_raw128", primary_metric, input_description)
        metrics.update({"status": "failed", "error": str(exc), "primary_score": None})
        write_json(neural_dir / "metrics.json", metrics)
        out.append(metrics)
    return out


def build_multilabel(objects: list[list[str]], train_idx: np.ndarray, max_vocab: int) -> tuple[np.ndarray, list[str]]:
    counts = Counter(obj for idx in train_idx for obj in objects[int(idx)])
    vocab = [item for item, _count in counts.most_common(max_vocab)]
    pos = {value: idx for idx, value in enumerate(vocab)}
    Y = np.zeros((len(objects), len(vocab)), dtype=np.float32)
    for row_idx, values in enumerate(objects):
        for value in values:
            if value in pos:
                Y[row_idx, pos[value]] = 1.0
    return Y, vocab


def run_multilabel_task(
    task: str,
    X_in: np.ndarray,
    objects: list[list[str]],
    rows: list[dict[str, Any]],
    splits: dict[str, np.ndarray],
    out_root: Path,
    args: argparse.Namespace,
    input_description: str,
) -> list[dict[str, Any]]:
    train_idx = splits["train"]
    test_idx = splits["test"]
    Y, vocab = build_multilabel(objects, train_idx, args.max_object_vocab)
    if Y.shape[1] == 0:
        return [write_unsupported(task, out_root, "simple_raw128_ridge", "no object vocabulary in train split", "micro_f1")]
    Xc, cols = cap_columns(X_in, args.max_input_dim, args.seed + stable_task_seed(task))
    train_rate = Y[train_idx].mean(axis=0)
    threshold = np.clip(train_rate, 0.05, 0.50).astype(np.float32)
    out: list[dict[str, Any]] = []
    simple_dir = out_root / "simple_raw128" / task
    try:
        pred, model = ridge_predict(Xc[train_idx], Y[train_idx], Xc[test_idx], args.ridge_l2, standardize_y=False)
        pred = np.clip(pred, 0.0, 1.0)
        split_metrics = multilabel_metrics(Y[test_idx], pred, threshold)
        metrics = base_metrics(task, "simple_raw128_ridge_multilabel", "micro_f1", input_description)
        metrics.update(
            {
                "status": "pass",
                "num_train_windows": int(len(train_idx)),
                "num_test_windows": int(len(test_idx)),
                "num_labels": int(len(vocab)),
                "input_dim": int(X_in.shape[1]),
                "fit_input_dim": int(Xc.shape[1]),
                "splits": {"test": split_metrics},
                "primary_score": split_metrics["micro_f1"],
            }
        )
        write_json(simple_dir / "metrics.json", metrics)
        np.savez_compressed(simple_dir / "model.npz", cols=cols, vocab=np.asarray(vocab, dtype=object), threshold=threshold, **model)
        out.append(metrics)
    except Exception as exc:
        metrics = base_metrics(task, "simple_raw128_ridge_multilabel", "micro_f1", input_description)
        metrics.update({"status": "failed", "error": str(exc), "primary_score": None})
        write_json(simple_dir / "metrics.json", metrics)
        out.append(metrics)

    if args.skip_neural:
        return out
    neural_dir = out_root / "neural_mlp_raw128" / task
    try:
        result = train_multilabel(Xc, Y, train_idx, test_idx, neural_config(args))
        split_metrics = multilabel_metrics(Y[test_idx], result["prob"], np.full(Y.shape[1], 0.5, dtype=np.float32))
        metrics = base_metrics(task, "neural_mlp_raw128_multilabel", "micro_f1", input_description)
        metrics.update(
            {
                "status": "pass",
                "device": result.get("device"),
                "history": result.get("history", []),
                "num_train_windows": int(len(train_idx)),
                "num_test_windows": int(len(test_idx)),
                "num_labels": int(len(vocab)),
                "input_dim": int(X_in.shape[1]),
                "fit_input_dim": int(Xc.shape[1]),
                "splits": {"test": split_metrics},
                "primary_score": split_metrics["micro_f1"],
            }
        )
        write_json(neural_dir / "metrics.json", metrics)
        save_torch_model(
            neural_dir / "model.pt",
            {
                "state_dict": result["state_dict"],
                "mean": result["mean"],
                "std": result["std"],
                "vocab": vocab,
                "selected_columns": cols,
                "metrics": metrics,
            },
        )
        out.append(metrics)
    except Exception as exc:
        metrics = base_metrics(task, "neural_mlp_raw128_multilabel", "micro_f1", input_description)
        metrics.update({"status": "failed", "error": str(exc), "primary_score": None})
        write_json(neural_dir / "metrics.json", metrics)
        out.append(metrics)
    return out


def subset_splits(splits: dict[str, np.ndarray], keep: np.ndarray) -> dict[str, np.ndarray]:
    local = {int(global_idx): local_idx for local_idx, global_idx in enumerate(keep)}
    return {
        split: np.asarray([local[int(i)] for i in idx if int(i) in local], dtype=np.int64)
        for split, idx in splits.items()
    }


def make_future_subset(rows: list[dict[str, Any]], frame_offset: int) -> tuple[np.ndarray, np.ndarray]:
    mapping = future_index_map(rows, frame_offset)
    current = np.asarray(sorted(mapping.keys()), dtype=np.int64)
    future = np.asarray([mapping[int(i)] for i in current], dtype=np.int64)
    return current, future


def make_temporal_order_pairs(X: np.ndarray, rows: list[dict[str, Any]], source_idx: np.ndarray, splits: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str], dict[str, np.ndarray]]:
    split_by_row = {}
    for split, idx in splits.items():
        for row_idx in idx:
            split_by_row[int(row_idx)] = split
    pair_X = []
    labels = []
    pair_split: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for indices in by_episode_sorted(rows).values():
        for left, right in zip(indices[:-1], indices[1:]):
            split = split_by_row.get(int(left))
            if split not in pair_split:
                continue
            xi = X[left, source_idx]
            xj = X[right, source_idx]
            pair_split[split].append(len(pair_X))
            pair_X.append(np.concatenate([xi, xj]).astype(np.float32))
            labels.append("chronological")
            pair_split[split].append(len(pair_X))
            pair_X.append(np.concatenate([xj, xi]).astype(np.float32))
            labels.append("reversed")
    return np.stack(pair_X).astype(np.float32), labels, {key: np.asarray(value, dtype=np.int64) for key, value in pair_split.items()}


def make_misalignment_pairs(
    X: np.ndarray,
    rows: list[dict[str, Any]],
    source_idx: np.ndarray,
    target_idx: np.ndarray,
    splits: dict[str, np.ndarray],
    shift: int,
) -> tuple[np.ndarray, list[str], dict[str, np.ndarray]]:
    split_by_row = {}
    for split, idx in splits.items():
        for row_idx in idx:
            split_by_row[int(row_idx)] = split
    pair_X = []
    labels = []
    pair_split: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for indices in by_episode_sorted(rows).values():
        if len(indices) <= shift:
            continue
        for pos, idx in enumerate(indices[:-shift]):
            shifted = indices[pos + shift]
            split = split_by_row.get(int(idx))
            if split not in pair_split:
                continue
            src = X[idx, source_idx]
            tgt = X[idx, target_idx]
            bad = X[shifted, target_idx]
            pair_split[split].append(len(pair_X))
            pair_X.append(np.concatenate([src, tgt]).astype(np.float32))
            labels.append("aligned")
            pair_split[split].append(len(pair_X))
            pair_X.append(np.concatenate([src, bad]).astype(np.float32))
            labels.append("shifted")
    return np.stack(pair_X).astype(np.float32), labels, {key: np.asarray(value, dtype=np.int64) for key, value in pair_split.items()}


def time_to_transition_targets(rows: list[dict[str, Any]], cap_frames: int) -> np.ndarray:
    targets = np.full(len(rows), float(cap_frames), dtype=np.float32)
    labels = [normalize_label(answer(row).get("action")) for row in rows]
    for indices in by_episode_sorted(rows).values():
        for pos, idx in enumerate(indices):
            label = labels[idx]
            start = row_start(rows[idx])
            distance = cap_frames
            for next_idx in indices[pos + 1 :]:
                if labels[next_idx] != label:
                    distance = min(row_start(rows[next_idx]) - start, cap_frames)
                    break
            targets[idx] = float(max(distance, 0))
    return targets[:, None]


def caption_hash_bucket_labels(caption_features: np.ndarray) -> list[str]:
    if caption_features.size == 0:
        return []
    magnitudes = np.abs(caption_features)
    top_bins = np.argmax(magnitudes, axis=1)
    top_values = np.max(magnitudes, axis=1)
    labels = []
    for bin_idx, value in zip(top_bins, top_values):
        labels.append(f"caption_hash_bin_{int(bin_idx):03d}" if float(value) > 1e-8 else "")
    return labels


def run_task(
    task: str,
    X: np.ndarray,
    rows: list[dict[str, Any]],
    manifest: list[dict[str, Any]],
    splits: dict[str, np.ndarray],
    out_root: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    non_caption_idx = block_indices(manifest, exclude=["caption_objects_interaction_text"])
    non_hand_idx = block_indices(manifest, exclude=["hand_left_joints", "hand_right_joints", "caption_objects_interaction_text"])
    hand_idx = block_indices(manifest, include=["hand_left_joints", "hand_right_joints"])
    depth_idx = block_indices(manifest, include=["depth_confidence"])
    caption_idx = block_indices(manifest, include=["caption_objects_interaction_text"])
    imu_idx = block_indices(manifest, include=["imu_accel_gyro"])
    motion_camera_imu_idx = block_indices(
        manifest,
        include=["hand_left_joints", "hand_right_joints", "body_joints", "body_contacts", "camera_translation", "camera_rotation_matrix", "imu_accel_gyro"],
    )
    source_no_depth_idx = block_indices(manifest, exclude=["depth_confidence"])

    if task in CLASS_FIELD_TASKS:
        field = CLASS_FIELD_TASKS[task]
        labels = [normalize_label(answer(row).get(field)) for row in rows]
        simple = run_simple_classification(task, X[:, non_caption_idx], rows, splits, labels, out_root, args, "sensor features excluding hashed caption text")
        neural = run_neural_classification(task, X[:, non_caption_idx], rows, splits, labels, out_root, args, "sensor features excluding hashed caption text")
        return [x for x in [simple, neural] if x is not None]

    if task == "object_relevance":
        return run_multilabel_task(task, X[:, non_caption_idx], [objects_from_answer(row) for row in rows], rows, splits, out_root, args, "sensor features excluding hashed caption text")

    if task == "caption_grounding":
        if len(caption_idx) == 0:
            return [write_unsupported(task, out_root, "simple_raw128_ridge", "caption feature block is missing", "mrr")]
        return run_regression_like(task, X[:, non_caption_idx], X[:, caption_idx], rows, splits, out_root, args, "non-caption sensor blocks projected to hashed caption/object/interaction block", "mrr", retrieval=True)

    if task == "cross_modal_retrieval":
        if len(depth_idx) == 0:
            return [write_unsupported(task, out_root, "simple_raw128_ridge", "depth feature block is missing", "mrr")]
        return run_regression_like(task, X[:, source_no_depth_idx], X[:, depth_idx], rows, splits, out_root, args, "all non-depth sensor blocks projected to depth-confidence block", "mrr", retrieval=True)

    if task == "modality_reconstruction":
        if len(depth_idx) == 0:
            return [write_unsupported(task, out_root, "simple_raw128_ridge", "depth feature block is missing", "r2")]
        return run_regression_like(task, X[:, source_no_depth_idx], X[:, depth_idx], rows, splits, out_root, args, "all non-depth sensor blocks reconstruct depth-confidence block", "r2")

    if task == "hand_trajectory_forecast":
        current, future = make_future_subset(rows, args.forecast_frames)
        if len(current) == 0 or len(hand_idx) == 0:
            return [write_unsupported(task, out_root, "simple_raw128_ridge", "no future hand target windows resolved", "mae")]
        return run_regression_like(
            task,
            X[current][:, non_hand_idx],
            X[future][:, hand_idx],
            rows_for_keep(rows, current),
            subset_splits(splits, current),
            out_root,
            args,
            f"current non-hand/non-caption features; target hand joint feature block +{args.forecast_frames} frames",
            "mae",
        )

    if task == "temporal_order":
        pair_X, labels, pair_splits = make_temporal_order_pairs(X, rows, non_caption_idx, splits)
        simple = run_simple_classification(task, pair_X, [{"id": i, "episode_id": "", "split": ""} for i in range(len(pair_X))], pair_splits, labels, out_root, args, "concatenated adjacent sensor-window pairs")
        neural = run_neural_classification(task, pair_X, [{"id": i, "episode_id": "", "split": ""} for i in range(len(pair_X))], pair_splits, labels, out_root, args, "concatenated adjacent sensor-window pairs")
        return [x for x in [simple, neural] if x is not None]

    if task == "misalignment_detection":
        pair_X, labels, pair_splits = make_misalignment_pairs(X, rows, motion_camera_imu_idx, np.concatenate([depth_idx, block_indices(manifest, include=["audio_fisheye_cam0_aac"])]), splits, args.misalignment_shift)
        simple = run_simple_classification(task, pair_X, [{"id": i, "episode_id": "", "split": ""} for i in range(len(pair_X))], pair_splits, labels, out_root, args, "motion/camera/IMU query paired with aligned or shifted depth/audio target")
        neural = run_neural_classification(task, pair_X, [{"id": i, "episode_id": "", "split": ""} for i in range(len(pair_X))], pair_splits, labels, out_root, args, "motion/camera/IMU query paired with aligned or shifted depth/audio target")
        return [x for x in [simple, neural] if x is not None]

    if task in {"long_horizon_next_action", "next_subtask_forecast", "object_set_forecast"}:
        current, future = make_future_subset(rows, args.future_frames)
        if len(current) == 0:
            return [write_unsupported(task, out_root, "simple_raw128", "no long-horizon future windows resolved", TASK_META[task]["metric"])]
        future_rows = [rows[int(i)] for i in future]
        current_rows = rows_for_keep(rows, current)
        current_splits = subset_splits(splits, current)
        if task == "long_horizon_next_action":
            labels = [normalize_label(answer(row).get("action")) for row in future_rows]
            simple = run_simple_classification(task, X[current][:, non_caption_idx], current_rows, current_splits, labels, out_root, args, f"current non-caption features; target action +{args.future_frames} frames")
            neural = run_neural_classification(task, X[current][:, non_caption_idx], current_rows, current_splits, labels, out_root, args, f"current non-caption features; target action +{args.future_frames} frames")
            return [x for x in [simple, neural] if x is not None]
        if task == "next_subtask_forecast":
            labels = [normalize_label(answer(row).get("subtask")) for row in future_rows]
            simple = run_simple_classification(task, X[current][:, non_caption_idx], current_rows, current_splits, labels, out_root, args, f"current non-caption features; target subtask +{args.future_frames} frames")
            neural = run_neural_classification(task, X[current][:, non_caption_idx], current_rows, current_splits, labels, out_root, args, f"current non-caption features; target subtask +{args.future_frames} frames")
            return [x for x in [simple, neural] if x is not None]
        return run_multilabel_task(task, X[current][:, non_caption_idx], [objects_from_answer(row) for row in future_rows], current_rows, current_splits, out_root, args, f"current non-caption features; target object set +{args.future_frames} frames")

    if task == "interaction_text_prediction":
        candidate_fields = ["interaction", "interaction_text", "caption_interaction"]
        labels = []
        for row in rows:
            data = answer(row)
            labels.append(next((normalize_label(data.get(field)) for field in candidate_fields if normalize_label(data.get(field))), ""))
        if not any(labels):
            if args.compact_proxy_missing_tasks and len(caption_idx) > 0:
                labels = caption_hash_bucket_labels(X[:, caption_idx])
                simple = run_simple_classification(
                    task,
                    X[:, non_caption_idx],
                    rows,
                    splits,
                    labels,
                    out_root,
                    args,
                    "compact proxy: non-caption sensor features predict the dominant hashed caption/object/interaction bin",
                )
                neural = run_neural_classification(
                    task,
                    X[:, non_caption_idx],
                    rows,
                    splits,
                    labels,
                    out_root,
                    args,
                    "compact proxy: non-caption sensor features predict the dominant hashed caption/object/interaction bin",
                )
                return annotate_metrics(
                    out_root,
                    [x for x in [simple, neural] if x is not None],
                    proxy_completion=True,
                    proxy_reason=(
                        "raw interaction strings are absent from the 128 JSONL/NPZ export; the published compact "
                        "caption_objects_interaction_text hash block is used only as a documented interaction-text proxy"
                    ),
                    proxy_target="dominant_caption_objects_interaction_text_hash_bin",
                )
            return [
                write_unsupported(
                    task,
                    out_root,
                    "simple_raw128_centroid",
                    "raw 128-episode annotation.hdf5 interaction text is not present in the JSONL export; only hashed caption_objects_interaction_text features are available",
                    "macro_f1",
                ),
                write_unsupported(
                    task,
                    out_root,
                    "neural_mlp_raw128",
                    "raw 128-episode annotation.hdf5 interaction text is not present in the JSONL export; only hashed caption_objects_interaction_text features are available",
                    "macro_f1",
                    subdir="neural_mlp_raw128",
                ),
            ]
        simple = run_simple_classification(task, X[:, non_caption_idx], rows, splits, labels, out_root, args, "sensor features excluding hashed caption text")
        neural = run_neural_classification(task, X[:, non_caption_idx], rows, splits, labels, out_root, args, "sensor features excluding hashed caption text")
        return [x for x in [simple, neural] if x is not None]

    if task == "action_object_relation":
        labels = []
        for row in rows:
            action_label = normalize_label(answer(row).get("action"))
            obj_label = "+".join(objects_from_answer(row))
            labels.append(f"{action_label}|{obj_label}" if action_label and obj_label else "")
        simple = run_simple_classification(task, X[:, non_caption_idx], rows, splits, labels, out_root, args, "sensor features excluding hashed caption text")
        neural = run_neural_classification(task, X[:, non_caption_idx], rows, splits, labels, out_root, args, "sensor features excluding hashed caption text")
        return [x for x in [simple, neural] if x is not None]

    if task == "imu_to_hand_pose":
        if len(imu_idx) == 0 or len(hand_idx) == 0:
            return [write_unsupported(task, out_root, "simple_raw128_ridge", "IMU or hand-joint feature block is missing", "mae")]
        return run_regression_like(task, X[:, imu_idx], X[:, hand_idx], rows, splits, out_root, args, "IMU acceleration/gyroscope block reconstructs hand-joint blocks", "mae")

    if task == "camera_view_sync_retrieval":
        view_blocks = [block for block in manifest if any(token in str(block.get("name", "")) for token in ["fisheye_cam", "stereo_"]) and "audio" not in str(block.get("name", ""))]
        if len(view_blocks) < 2:
            if args.compact_proxy_missing_tasks:
                camera_idx = block_indices(manifest, include=["camera_translation", "camera_rotation_matrix"])
                sync_target_idx = np.concatenate([depth_idx, block_indices(manifest, include=["audio_fisheye_cam0_aac"])])
                if len(camera_idx) > 0 and len(sync_target_idx) > 0:
                    results = run_regression_like(
                        task,
                        X[:, camera_idx],
                        X[:, sync_target_idx],
                        rows,
                        splits,
                        out_root,
                        args,
                        "compact proxy: camera-pose block retrieves synchronized same-window depth/audio stream",
                        "mrr",
                        retrieval=True,
                    )
                    return annotate_metrics(
                        out_root,
                        results,
                        proxy_completion=True,
                        proxy_reason=(
                            "paired video-view embeddings are absent from the 128 NPZ export; camera pose and same-window "
                            "depth/audio are used as a documented compact synchronization proxy"
                        ),
                        proxy_target="same_window_depth_confidence_plus_audio_fisheye_cam0_aac",
                    )
            reason = "128-episode NPZ manifest has camera pose plus audio/depth/caption features, but no two explicit video-view feature blocks for camera-view synchronization"
            return [
                write_unsupported(task, out_root, "simple_raw128_ridge", reason, "mrr"),
                write_unsupported(task, out_root, "neural_mlp_raw128", reason, "mrr", subdir="neural_mlp_raw128"),
            ]
        src = np.asarray(range(int(view_blocks[0]["start"]), int(view_blocks[0]["end"])), dtype=np.int64)
        tgt = np.asarray(range(int(view_blocks[1]["start"]), int(view_blocks[1]["end"])), dtype=np.int64)
        return run_regression_like(task, X[:, src], X[:, tgt], rows, splits, out_root, args, "one camera-view block projected to a synchronized second camera-view block", "mrr", retrieval=True)

    if task == "time_to_transition":
        return run_regression_like(
            task,
            X[:, non_caption_idx],
            time_to_transition_targets(rows, args.transition_cap_frames),
            rows,
            splits,
            out_root,
            args,
            f"non-caption sensor features regress frames to next action boundary capped at {args.transition_cap_frames}",
            "mae",
        )

    raise ValueError(f"Unhandled task {task}")


def write_run_summary(out_root: Path, args: argparse.Namespace, load_report: dict[str, Any], results: list[dict[str, Any]]) -> None:
    rows = []
    for item in results:
        rows.append(
            {
                "task": item.get("task"),
                "task_display_name": item.get("task_display_name"),
                "model_family": item.get("model_family"),
                "status": item.get("status"),
                "primary_metric": item.get("primary_metric"),
                "primary_score": item.get("primary_score"),
                "metric_direction": item.get("metric_direction"),
                "reason": item.get("reason"),
                "error": item.get("error"),
            }
        )
    write_csv(out_root / "metrics_summary.csv", rows)
    write_json(
        out_root / "run_summary.json",
        {
            "dataset_jsonl": str(args.dataset_jsonl),
            "feature_manifest_json": str(args.feature_manifest_json or default_manifest_path(args.dataset_jsonl)),
            "tasks_requested": selected_tasks(args.tasks),
            "load_report": load_report,
            "num_result_records": len(results),
            "status_counts": dict(Counter(str(item.get("status")) for item in results)),
            "results": rows,
        },
    )


def main() -> None:
    args = parse_args()
    args.feature_manifest_json = args.feature_manifest_json or default_manifest_path(args.dataset_jsonl)
    tasks = selected_tasks(args.tasks)
    log(f"loading rows from {args.dataset_jsonl}")
    rows = load_rows(args.dataset_jsonl)
    log(f"loading feature matrix for {len(rows)} rows")
    X, rows, load_report = load_feature_matrix(rows, args)
    log(f"loaded {X.shape[0]} x {X.shape[1]} features from {load_report['resolved_npz_files']} NPZ files")
    manifest = load_feature_manifest(args.feature_manifest_json)
    splits = split_indices(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "input_report.json", {"load_report": load_report, "split_counts": {k: int(len(v)) for k, v in splits.items()}, "feature_manifest": manifest})
    results: list[dict[str, Any]] = []
    for task in tasks:
        log(f"running {task}")
        task_results = run_task(task, X, rows, manifest, splits, args.output_dir, args)
        results.extend(task_results)
        write_run_summary(args.output_dir, args, load_report, results)
    write_run_summary(args.output_dir, args, load_report, results)
    log(f"done; wrote {len(results)} result records to {args.output_dir}")


if __name__ == "__main__":
    main()

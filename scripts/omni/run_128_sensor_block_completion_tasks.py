#!/usr/bin/env python3
"""Complete 128-episode baseline tasks that require staged sensor blocks.

The public JSONL metadata run cannot score tasks whose target is a processed
feature block.  On the staged GPU mirror the JSONL rows still point to the
private 4430-dim sensor NPZ shards, so this runner fills only the task cells
that have a real block-level target available:

* hand_trajectory_forecast
* cross_modal_retrieval
* modality_reconstruction
* misalignment_detection
* imu_to_hand_pose

Raw interaction strings and paired camera-view embeddings are still absent from
the staged 128 export, so those task IDs remain explicit scoreless records.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "omni"))

from train_min_action_model import compute_metrics, fit_scaler, predict, train_softmax_classifier
from run_128_task_baselines import (
    TASKS,
    build_markdown,
    make_future_subset,
    neural_config,
    portable_path,
    regression_metrics,
    retrieval_metrics_from_scores,
    ridge_regression_predict,
    split_indices,
    subset_splits,
    task_display_name,
    write_json,
)


DEFAULT_DATASET = (
    ROOT
    / "results/omni_finetune/xperience10m_qwen3_omni_128ep_multiscale_cap96_v5_full8gpu_lora_dataset"
    / "dataset_a100_eval.jsonl"
)
DEFAULT_OUTPUT = ROOT / "results/omni_finetune/a100_128_metadata_task_baselines_20260616_v2"

FEATURE_BLOCKS = {
    "hand_left_joints": (0, 441),
    "hand_right_joints": (441, 882),
    "body_joints": (882, 1974),
    "body_contacts": (1974, 2121),
    "camera_translation": (2121, 2142),
    "camera_rotation_matrix": (2142, 2205),
    "imu_accel_gyro": (2205, 2247),
    "depth_confidence": (2247, 3227),
    "audio_fisheye_cam0_aac": (3227, 3395),
    "caption_objects_interaction_text": (3395, 4291),
    "slam_point_cloud": (4291, 4313),
    "calibration": (4313, 4430),
}

HAND_BLOCKS = ("hand_left_joints", "hand_right_joints")
MOTION_QUERY_BLOCKS = (
    "hand_left_joints",
    "hand_right_joints",
    "body_joints",
    "body_contacts",
    "camera_translation",
    "camera_rotation_matrix",
    "imu_accel_gyro",
)
VISUAL_TARGET_BLOCKS = ("depth_confidence", "slam_point_cloud", "calibration")
IMU_BLOCKS = ("imu_accel_gyro",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-jsonl", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--future-frames", type=int, default=100)
    parser.add_argument("--l2", type=float, default=2e-3)
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--learning-rate", type=float, default=0.16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--include-neural", action="store_true", default=True)
    parser.add_argument("--neural-epochs", type=int, default=35)
    parser.add_argument("--neural-hidden-dim", type=int, default=128)
    parser.add_argument("--neural-batch-size", type=int, default=256)
    parser.add_argument("--neural-learning-rate", type=float, default=1e-3)
    parser.add_argument("--neural-weight-decay", type=float, default=1e-4)
    parser.add_argument("--neural-dropout", type=float, default=0.10)
    parser.add_argument("--neural-device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[128-sensor-blocks] {message}", file=sys.stderr, flush=True)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def compact_row(row: dict[str, Any]) -> dict[str, Any]:
    window = row.get("center_window") or {}
    return {
        "id": row.get("id"),
        "episode_id": row.get("episode_id"),
        "source_episode_id": row.get("source_episode_id"),
        "split": row.get("split"),
        "center_window": window,
        "sensor_feature_path": row.get("sensor_feature_path"),
        "sensor_feature_index": row.get("sensor_feature_index"),
        "sensor_feature_dim": row.get("sensor_feature_dim"),
    }


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(compact_row(json.loads(line)))
    return rows


def resolve_feature_path(raw_path: str | None, dataset_jsonl: Path) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.exists():
        return path
    if not path.is_absolute():
        candidate = dataset_jsonl.parent / path
        if candidate.exists():
            return candidate
    return None


def load_sensor_matrix(rows: list[dict[str, Any]], dataset_jsonl: Path) -> np.ndarray:
    grouped: dict[Path, list[tuple[int, int]]] = {}
    missing = 0
    for row_idx, row in enumerate(rows):
        path = resolve_feature_path(row.get("sensor_feature_path"), dataset_jsonl)
        feature_idx = row.get("sensor_feature_index")
        if path is None or feature_idx is None:
            missing += 1
            continue
        grouped.setdefault(path, []).append((row_idx, int(feature_idx)))
    if missing:
        raise RuntimeError(f"{missing} rows do not resolve to staged sensor feature blocks")
    if not grouped:
        raise RuntimeError("no staged sensor feature paths were found in the JSONL")

    first_path = next(iter(grouped))
    with np.load(first_path) as data:
        dim = int(data["features"].shape[1])
    if dim < max(end for _start, end in FEATURE_BLOCKS.values()):
        raise RuntimeError(f"sensor feature dim {dim} is smaller than expected 4430-dim manifest")

    matrix = np.empty((len(rows), dim), dtype=np.float32)
    for path, items in grouped.items():
        log(f"loading {len(items)} rows from {path}")
        with np.load(path) as data:
            features = np.asarray(data["features"], dtype=np.float32)
            for row_idx, feature_idx in items:
                matrix[row_idx] = features[feature_idx]
    return matrix


def feature_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    packed = []
    for row in rows:
        window = row.get("center_window") or {}
        packed.append(
            {
                "id": row.get("id"),
                "episode_id": str(row.get("episode_id")),
                "source_episode_id": str(row.get("source_episode_id") or row.get("episode_id")),
                "split": row.get("split"),
                "start_frame": int(window.get("start_frame", 0) or 0),
                "end_frame": int(window.get("end_frame", 0) or 0),
            }
        )
    return packed


def block(matrix: np.ndarray, names: tuple[str, ...]) -> np.ndarray:
    return np.concatenate([matrix[:, FEATURE_BLOCKS[name][0] : FEATURE_BLOCKS[name][1]] for name in names], axis=1)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norm, 1e-6)


def hand_mpjpe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = y_pred - y_true
    if err.shape[1] % 3 != 0:
        return float(np.sqrt(np.mean(err**2)))
    return float(np.mean(np.linalg.norm(err.reshape(err.shape[0], -1, 3), axis=2)))


def regression_summary_metrics(task_id: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    metrics = regression_metrics(y_true, y_pred)
    if task_id == "hand_trajectory_forecast":
        metrics["mpjpe"] = hand_mpjpe(y_true, y_pred)
    return metrics


def regression_prediction_rows(
    rows: list[dict[str, Any]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    test_idx: np.ndarray,
    *,
    task_id: str,
) -> list[dict[str, Any]]:
    records = []
    for local_idx, global_idx in enumerate(test_idx):
        global_idx = int(global_idx)
        err = y_pred[local_idx] - y_true[global_idx]
        record = {
            **rows[global_idx],
            "target_l2": float(np.linalg.norm(y_true[global_idx])),
            "prediction_l2": float(np.linalg.norm(y_pred[local_idx])),
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err**2))),
        }
        if task_id == "hand_trajectory_forecast" and err.shape[0] % 3 == 0:
            record["mpjpe"] = float(np.mean(np.linalg.norm(err.reshape(-1, 3), axis=1)))
        records.append(record)
    return records


def neural_regression(
    task_id: str,
    rows: list[dict[str, Any]],
    X: np.ndarray,
    Y: np.ndarray,
    splits: dict[str, np.ndarray],
    out_dir: Path,
    args: argparse.Namespace,
    *,
    input_features: str,
    target_features: str,
    primary_metric: str,
    metric_direction: str,
) -> dict[str, Any]:
    from neural_task_models import save_torch_model, train_regressor

    out_dir.mkdir(parents=True, exist_ok=True)
    result = train_regressor(X.astype(np.float32), Y.astype(np.float32), splits["train"], splits["test"], neural_config(args))
    test_metrics = regression_summary_metrics(task_id, Y[splits["test"]], result["pred"])
    payload = {
        "status": "pass",
        "task": task_id,
        "task_display_name": task_display_name(task_id),
        "model_family": "neural_mlp_128_sensor_blocks",
        "source": "128_episode_staged_sensor_feature_blocks",
        "scope": "multi_episode_128_aligned_sensor_block_baseline",
        "input_features": input_features,
        "target_features": target_features,
        "split_policy": "train on 128-episode train split, report held-out test split",
        "num_train_windows": int(len(splits["train"])),
        "num_val_windows": int(len(splits["val"])),
        "num_test_windows": int(len(splits["test"])),
        "history": result["history"],
        "device": result["device"],
        "splits": {"test": test_metrics},
        "primary_metric": primary_metric,
        "metric_direction": metric_direction,
        "primary_score": test_metrics[primary_metric],
    }
    write_json(out_dir / "metrics.json", payload)
    write_csv(out_dir / "predictions.csv", regression_prediction_rows(rows, Y, result["pred"], splits["test"], task_id=task_id))
    save_torch_model(
        out_dir / "model.pt",
        {
            "state_dict": result["state_dict"],
            "x_mean": result["x_mean"],
            "x_std": result["x_std"],
            "y_mean": result["y_mean"],
            "y_std": result["y_std"],
            "metrics": payload,
        },
    )
    return payload


def vector_regression_task(
    task_id: str,
    rows: list[dict[str, Any]],
    X: np.ndarray,
    Y: np.ndarray,
    splits: dict[str, np.ndarray],
    out_root: Path,
    args: argparse.Namespace,
    *,
    input_features: str,
    target_features: str,
    primary_metric: str,
    metric_direction: str,
) -> dict[str, Any]:
    out_dir = out_root / task_id
    out_dir.mkdir(parents=True, exist_ok=True)
    pred, model = ridge_regression_predict(X[splits["train"]], Y[splits["train"]], X[splits["test"]], args.l2)
    test_metrics = regression_summary_metrics(task_id, Y[splits["test"]], pred)
    val_metrics = {}
    if len(splits["val"]):
        val_pred, _ = ridge_regression_predict(X[splits["train"]], Y[splits["train"]], X[splits["val"]], args.l2)
        val_metrics = regression_summary_metrics(task_id, Y[splits["val"]], val_pred)
    payload = {
        "status": "pass",
        "task": task_id,
        "task_display_name": task_display_name(task_id),
        "model_family": "simple_ridge_128_sensor_blocks",
        "source": "128_episode_staged_sensor_feature_blocks",
        "scope": "multi_episode_128_aligned_sensor_block_baseline",
        "input_features": input_features,
        "target_features": target_features,
        "split_policy": "train ridge regressor on 128-episode train split, report held-out test split",
        "num_train_windows": int(len(splits["train"])),
        "num_val_windows": int(len(splits["val"])),
        "num_test_windows": int(len(splits["test"])),
        "splits": {"val": val_metrics, "test": test_metrics},
        "primary_metric": primary_metric,
        "metric_direction": metric_direction,
        "primary_score": test_metrics[primary_metric],
    }
    write_json(out_dir / "metrics.json", payload)
    write_csv(out_dir / "predictions.csv", regression_prediction_rows(rows, Y, pred, splits["test"], task_id=task_id))
    np.savez_compressed(out_dir / "model.npz", **model)

    neural_payload = None
    if args.include_neural:
        neural_payload = neural_regression(
            task_id,
            rows,
            X,
            Y,
            splits,
            out_root / "neural_mlp" / task_id,
            args,
            input_features=input_features,
            target_features=target_features,
            primary_metric=primary_metric,
            metric_direction=metric_direction,
        )
    return {"simple": payload, "neural": neural_payload}


def retrieval_scores_from_projection(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    l2: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    pred, model = ridge_regression_predict(X_train, Y_train, X_test, l2)
    scores = normalize_rows(pred) @ normalize_rows(Y_test).T
    return scores.astype(np.float32), pred, model


def neural_retrieval(
    task_id: str,
    rows: list[dict[str, Any]],
    X: np.ndarray,
    Y: np.ndarray,
    splits: dict[str, np.ndarray],
    out_dir: Path,
    args: argparse.Namespace,
    *,
    input_features: str,
    target_features: str,
) -> dict[str, Any]:
    from neural_task_models import save_torch_model, train_regressor

    out_dir.mkdir(parents=True, exist_ok=True)
    result = train_regressor(X.astype(np.float32), Y.astype(np.float32), splits["train"], splits["test"], neural_config(args))
    scores = normalize_rows(result["pred"]) @ normalize_rows(Y[splits["test"]]).T
    retrieval_metrics, rank_rows = retrieval_metrics_from_scores(scores, rows, splits["test"])
    payload = {
        "status": "pass",
        "task": task_id,
        "task_display_name": task_display_name(task_id),
        "model_family": "neural_mlp_128_sensor_block_retrieval",
        "source": "128_episode_staged_sensor_feature_blocks",
        "scope": "multi_episode_128_aligned_sensor_block_baseline",
        "input_features": input_features,
        "target_features": target_features,
        "split_policy": "train neural projection on aligned train pairs, rank held-out target candidates",
        "num_train_windows": int(len(splits["train"])),
        "num_val_windows": int(len(splits["val"])),
        "num_test_windows": int(len(splits["test"])),
        "history": result["history"],
        "device": result["device"],
        **retrieval_metrics,
        "primary_metric": "mrr",
        "metric_direction": "higher",
        "primary_score": retrieval_metrics["mrr"],
    }
    write_json(out_dir / "metrics.json", payload)
    write_csv(out_dir / "ranks.csv", rank_rows)
    save_torch_model(
        out_dir / "model.pt",
        {
            "state_dict": result["state_dict"],
            "x_mean": result["x_mean"],
            "x_std": result["x_std"],
            "y_mean": result["y_mean"],
            "y_std": result["y_std"],
            "metrics": payload,
        },
    )
    return payload


def retrieval_task(
    task_id: str,
    rows: list[dict[str, Any]],
    X: np.ndarray,
    Y: np.ndarray,
    splits: dict[str, np.ndarray],
    out_root: Path,
    args: argparse.Namespace,
    *,
    input_features: str,
    target_features: str,
) -> dict[str, Any]:
    out_dir = out_root / task_id
    out_dir.mkdir(parents=True, exist_ok=True)
    scores, _pred, model = retrieval_scores_from_projection(X[splits["train"]], Y[splits["train"]], X[splits["test"]], Y[splits["test"]], args.l2)
    retrieval_metrics, rank_rows = retrieval_metrics_from_scores(scores, rows, splits["test"])
    payload = {
        "status": "pass",
        "task": task_id,
        "task_display_name": task_display_name(task_id),
        "model_family": "simple_ridge_128_sensor_block_retrieval",
        "source": "128_episode_staged_sensor_feature_blocks",
        "scope": "multi_episode_128_aligned_sensor_block_baseline",
        "input_features": input_features,
        "target_features": target_features,
        "split_policy": "train ridge projection on aligned train pairs, rank held-out target candidates",
        "num_train_windows": int(len(splits["train"])),
        "num_val_windows": int(len(splits["val"])),
        "num_test_windows": int(len(splits["test"])),
        **retrieval_metrics,
        "primary_metric": "mrr",
        "metric_direction": "higher",
        "primary_score": retrieval_metrics["mrr"],
    }
    write_json(out_dir / "metrics.json", payload)
    write_csv(out_dir / "ranks.csv", rank_rows)
    np.savez_compressed(out_dir / "model.npz", **model)

    neural_payload = None
    if args.include_neural:
        neural_payload = neural_retrieval(
            task_id,
            rows,
            X,
            Y,
            splits,
            out_root / "neural_mlp" / task_id,
            args,
            input_features=input_features,
            target_features=target_features,
        )
    return {"simple": payload, "neural": neural_payload}


def pair_rows_for_split(
    feature_rows_: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    query: np.ndarray,
    target: np.ndarray,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    by_episode: dict[str, list[int]] = {}
    for idx in indices:
        by_episode.setdefault(str(rows[int(idx)].get("episode_id")), []).append(int(idx))
    feats = []
    labels = []
    records = []
    for episode_id, items in by_episode.items():
        items = sorted(items, key=lambda idx: int((rows[idx].get("center_window") or {}).get("start_frame", 0) or 0))
        if len(items) < 2:
            continue
        for pos, idx in enumerate(items):
            shifted = items[(pos + 1) % len(items)]
            feats.append(np.concatenate([query[idx], target[idx]]))
            labels.append(1)
            records.append({**feature_rows_[idx], "candidate_id": feature_rows_[idx]["id"], "pair_label": "aligned", "episode_id": episode_id})
            feats.append(np.concatenate([query[idx], target[shifted]]))
            labels.append(0)
            records.append({**feature_rows_[idx], "candidate_id": feature_rows_[shifted]["id"], "pair_label": "shifted", "episode_id": episode_id})
    return np.stack(feats).astype(np.float32), np.asarray(labels, dtype=np.int64), records


def neural_pair_classifier(
    task_id: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_rows: list[dict[str, Any]],
    out_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    from neural_task_models import train_classifier

    out_dir.mkdir(parents=True, exist_ok=True)
    X_all = np.concatenate([X_train, X_test], axis=0).astype(np.float32)
    y_all = np.concatenate([y_train, y_test], axis=0).astype(np.int64)
    train_idx = np.arange(len(X_train), dtype=np.int64)
    test_idx = np.arange(len(X_train), len(X_train) + len(X_test), dtype=np.int64)
    result = train_classifier(X_all, y_all, train_idx, test_idx, n_classes=2, config=neural_config(args), use_class_weights=True)
    pred = result["pred"]
    metrics, per_class, cm = compute_metrics(y_test, pred, ["shifted", "aligned"])
    pred_rows = []
    for k, row in enumerate(test_rows):
        pred_rows.append({**row, "predicted_label": "aligned" if int(pred[k]) else "shifted", "correct": int(pred[k] == y_test[k])})
    payload = {
        "status": "pass",
        "task": task_id,
        "task_display_name": task_display_name(task_id),
        "model_family": "neural_mlp_128_sensor_block_pair",
        "source": "128_episode_staged_sensor_feature_blocks",
        "scope": "multi_episode_128_aligned_sensor_block_baseline",
        "input_features": "motion-side staged feature blocks concatenated with visual/depth-side staged feature blocks",
        "split_policy": "train on aligned/shifted train pairs, report held-out test pairs",
        "num_train_samples": int(len(train_idx)),
        "num_test_samples": int(len(test_idx)),
        "history": result["history"],
        "device": result["device"],
        **metrics,
        "primary_metric": "f1",
        "metric_direction": "higher",
        "primary_score": metrics["macro_f1"],
    }
    write_json(out_dir / "metrics.json", payload)
    write_csv(out_dir / "predictions.csv", pred_rows)
    write_csv(out_dir / "per_class_metrics.csv", per_class)
    write_confusion(out_dir / "confusion_matrix.csv", cm, ["shifted", "aligned"])
    return payload


def write_confusion(path: Path, cm: np.ndarray, class_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["true\\pred"] + class_names)
        for idx, name in enumerate(class_names):
            writer.writerow([name] + [int(v) for v in cm[idx]])


def misalignment_task(
    rows: list[dict[str, Any]],
    feature_rows_: list[dict[str, Any]],
    query: np.ndarray,
    target: np.ndarray,
    splits: dict[str, np.ndarray],
    out_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    task_id = "misalignment_detection"
    X_train, y_train, _train_rows = pair_rows_for_split(feature_rows_, rows, query, target, splits["train"])
    X_test, y_test, test_rows = pair_rows_for_split(feature_rows_, rows, query, target, splits["test"])
    X_all = np.concatenate([X_train, X_test], axis=0)
    mean, std = fit_scaler(X_train)
    Xs = (X_all - mean) / std
    train_idx = np.arange(len(X_train), dtype=np.int64)
    test_idx = np.arange(len(X_train), len(X_train) + len(X_test), dtype=np.int64)
    W, b, history = train_softmax_classifier(Xs[train_idx], y_train, 2, args.epochs, args.learning_rate, args.l2, True, args.seed)
    pred, prob = predict(Xs[test_idx], W, b)
    metrics, per_class, cm = compute_metrics(y_test, pred, ["shifted", "aligned"])
    pred_rows = []
    for k, row in enumerate(test_rows):
        pred_rows.append(
            {
                **row,
                "predicted_label": "aligned" if int(pred[k]) else "shifted",
                "confidence": float(prob[k, int(pred[k])]),
                "correct": int(pred[k] == y_test[k]),
            }
        )
    payload = {
        "status": "pass",
        "task": task_id,
        "task_display_name": task_display_name(task_id),
        "model_family": "simple_softmax_128_sensor_block_pair",
        "source": "128_episode_staged_sensor_feature_blocks",
        "scope": "multi_episode_128_aligned_sensor_block_baseline",
        "input_features": "motion-side staged feature blocks concatenated with visual/depth-side staged feature blocks",
        "split_policy": "train on aligned/shifted train pairs, report held-out test pairs",
        "num_train_samples": int(len(train_idx)),
        "num_test_samples": int(len(test_idx)),
        "history": history,
        **metrics,
        "primary_metric": "f1",
        "metric_direction": "higher",
        "primary_score": metrics["macro_f1"],
    }
    out_dir = out_root / task_id
    write_json(out_dir / "metrics.json", payload)
    write_csv(out_dir / "predictions.csv", pred_rows)
    write_csv(out_dir / "per_class_metrics.csv", per_class)
    write_confusion(out_dir / "confusion_matrix.csv", cm, ["shifted", "aligned"])
    np.savez_compressed(out_dir / "model.npz", mean=mean, std=std, W=W, b=b, class_names=np.asarray(["shifted", "aligned"], dtype=object))

    neural_payload = None
    if args.include_neural:
        neural_payload = neural_pair_classifier(task_id, X_train, y_train, X_test, y_test, test_rows, out_root / "neural_mlp" / task_id, args)
    return {"simple": payload, "neural": neural_payload}


def merge_summary(out_dir: Path, rows: list[dict[str, Any]], task_results: list[dict[str, Any]]) -> None:
    summary_path = out_dir / "summary_report.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        split_counts = Counter(str(row.get("split")) for row in rows)
        summary = {
            "status": "pass",
            "run_id": "xperience10m_128_episode_aligned_task_baselines",
            "num_rows": len(rows),
            "split_counts": {key: int(split_counts.get(key, 0)) for key in ("train", "val", "test")},
            "tasks": [],
        }
    existing = {item["task"]: item for item in summary.get("tasks", [])}
    for result in task_results:
        existing[result["task"]] = result
    summary["tasks"] = [existing[task_id] for task_id in TASKS if task_id in existing]
    summary["source_policy"] = (
        "derived JSONL metadata plus staged 128-episode sensor feature blocks for task cells whose targets are present; "
        "raw annotation interaction text and paired camera-view embeddings remain absent"
    )
    feature_contract = dict(summary.get("feature_contract") or {})
    feature_contract["sensor_block_completion"] = {
        "kind": "private_staged_4430_dim_feature_blocks",
        "blocks": {name: {"start": start, "end": end} for name, (start, end) in FEATURE_BLOCKS.items()},
        "completed_task_ids": [result["task"] for result in task_results],
        "not_completed_task_ids": ["interaction_text_prediction", "camera_view_sync_retrieval"],
    }
    summary["feature_contract"] = feature_contract
    write_json(summary_path, summary)
    write_csv(
        out_dir / "task_metrics.csv",
        [
            {
                "task": row["task"],
                "task_display_name": row.get("task_display_name") or task_display_name(row["task"]),
                "simple_status": (row.get("simple") or {}).get("status"),
                "simple_primary_metric": (row.get("simple") or {}).get("primary_metric"),
                "simple_primary_score": (row.get("simple") or {}).get("primary_score"),
                "neural_status": (row.get("neural") or {}).get("status") if row.get("neural") else "not_run",
                "neural_primary_metric": (row.get("neural") or {}).get("primary_metric") if row.get("neural") else "",
                "neural_primary_score": (row.get("neural") or {}).get("primary_score") if row.get("neural") else "",
            }
            for row in summary["tasks"]
        ],
    )
    (out_dir / "BASELINE_ALIGNMENT_REPORT.md").write_text(build_markdown(summary), encoding="utf-8")


def write_sensor_report(out_dir: Path, task_results: list[dict[str, Any]]) -> None:
    lines = [
        "# 128-Episode Sensor-Block Completion Tasks",
        "",
        "This supplement fills task cells that cannot be produced by JSONL metadata alone but can be produced from the staged 4430-dim processed feature blocks on the staged GPU mirror.",
        "",
        "| task | simple status | simple primary | neural status | neural primary |",
        "| --- | --- | ---: | --- | ---: |",
    ]
    for item in task_results:
        simple = item.get("simple") or {}
        neural = item.get("neural") or {}
        simple_score = simple.get("primary_score")
        neural_score = neural.get("primary_score") if neural else None
        lines.append(
            "| {task} | {simple_status} | {simple_score} | {neural_status} | {neural_score} |".format(
                task=item.get("task_display_name") or task_display_name(item["task"]),
                simple_status=simple.get("status", ""),
                simple_score="" if simple_score is None else f"{float(simple_score):.4f}",
                neural_status=neural.get("status", "not_run") if neural else "not_run",
                neural_score="" if neural_score is None else f"{float(neural_score):.4f}",
            )
        )
    lines.extend(
        [
            "",
            "Still scoreless for this layer: `interaction_text_prediction` needs raw annotation interaction text, and `camera_view_sync_retrieval` needs paired per-camera feature embeddings.",
            "",
        ]
    )
    (out_dir / "SENSOR_BLOCK_COMPLETION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not args.dataset_jsonl.exists():
        raise FileNotFoundError(f"missing dataset JSONL: {args.dataset_jsonl}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log(f"loading rows from {portable_path(args.dataset_jsonl)}")
    rows = load_rows(args.dataset_jsonl)
    splits = split_indices(rows)
    feature_rows_ = feature_rows(rows)
    log(f"loading staged sensor matrix for {len(rows)} rows")
    S = load_sensor_matrix(rows, args.dataset_jsonl)
    log(f"sensor matrix shape: {S.shape[0]}x{S.shape[1]}")

    motion = block(S, MOTION_QUERY_BLOCKS)
    visual = block(S, VISUAL_TARGET_BLOCKS)
    hand = block(S, HAND_BLOCKS)
    imu = block(S, IMU_BLOCKS)

    task_results: list[dict[str, Any]] = []

    log("hand_trajectory_forecast: resolving future hand targets")
    current_idx, future_idx = make_future_subset(rows, args.future_frames)
    current_rows = [rows[int(idx)] for idx in current_idx]
    current_feature_rows = [feature_rows_[int(idx)] for idx in current_idx]
    current_splits = subset_splits(splits, current_idx)
    hand_result = vector_regression_task(
        "hand_trajectory_forecast",
        current_feature_rows,
        S[current_idx],
        hand[future_idx],
        current_splits,
        args.output_dir,
        args,
        input_features="current staged 4430-dim sensor feature vector",
        target_features=f"future left/right hand joint feature blocks at +{args.future_frames} frames",
        primary_metric="mpjpe",
        metric_direction="lower",
    )
    task_results.append({"task": "hand_trajectory_forecast", "task_display_name": task_display_name("hand_trajectory_forecast"), **hand_result})

    log("cross_modal_retrieval: start")
    retrieval_result = retrieval_task(
        "cross_modal_retrieval",
        feature_rows_,
        motion,
        visual,
        splits,
        args.output_dir,
        args,
        input_features="motion, body, contact, camera-pose, and IMU staged feature blocks",
        target_features="depth-confidence, SLAM point-cloud, and calibration staged feature blocks",
    )
    task_results.append({"task": "cross_modal_retrieval", "task_display_name": task_display_name("cross_modal_retrieval"), **retrieval_result})

    log("modality_reconstruction: start")
    reconstruction_result = vector_regression_task(
        "modality_reconstruction",
        feature_rows_,
        motion,
        visual,
        splits,
        args.output_dir,
        args,
        input_features="motion, body, contact, camera-pose, and IMU staged feature blocks",
        target_features="depth-confidence, SLAM point-cloud, and calibration staged feature blocks",
        primary_metric="r2",
        metric_direction="higher",
    )
    task_results.append({"task": "modality_reconstruction", "task_display_name": task_display_name("modality_reconstruction"), **reconstruction_result})

    log("misalignment_detection: start")
    misalignment_result = misalignment_task(rows, feature_rows_, motion, visual, splits, args.output_dir, args)
    task_results.append({"task": "misalignment_detection", "task_display_name": task_display_name("misalignment_detection"), **misalignment_result})

    log("imu_to_hand_pose: start")
    imu_result = vector_regression_task(
        "imu_to_hand_pose",
        feature_rows_,
        imu,
        hand,
        splits,
        args.output_dir,
        args,
        input_features="current IMU acceleration/gyroscope staged feature block only",
        target_features="current left/right hand joint staged feature blocks",
        primary_metric="mae",
        metric_direction="lower",
    )
    task_results.append({"task": "imu_to_hand_pose", "task_display_name": task_display_name("imu_to_hand_pose"), **imu_result})

    merge_summary(args.output_dir, rows, task_results)
    write_sensor_report(args.output_dir, task_results)
    print(json.dumps({"status": "pass", "output_dir": str(args.output_dir), "completed_tasks": [item["task"] for item in task_results]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

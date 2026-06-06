#!/usr/bin/env python3
"""Run 128-episode aligned task baselines from the Qwen JSONL export.

This is the multi-episode companion to scripts/episode_task_suite.py.  The
single-episode suite uses a full numeric feature tensor from one public sample.
The 128-episode public package intentionally does not redistribute raw sensor
feature NPZ files, so this runner uses only public-safe JSONL metadata and
strictly avoids answer fields as input features.

The output keeps the same twelve task IDs.  Tasks with enough JSONL signal get
simple and, where appropriate, neural baselines over the same train/val/test
episode split used by the Qwen3-Omni pilot.  Tasks whose original target
requires missing raw motion/depth/audio feature blocks are emitted as explicit
unsupported records instead of fabricated scores.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from train_min_action_model import compute_metrics, fit_scaler, predict, train_softmax_classifier
from task_display import task_display_name


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

CLASSIFICATION_TASKS = {
    "timeline_action": ("action", "macro_f1"),
    "timeline_subtask": ("subtask", "macro_f1"),
    "transition_detection": ("transition", "macro_f1"),
    "next_action": ("next_action", "macro_f1"),
    "contact_prediction": ("contact", "macro_f1"),
}

UNSUPPORTED_TASKS = {
    "hand_trajectory_forecast": {
        "primary_metric": "mpjpe",
        "reason": "requires future hand-joint trajectories from raw sensor feature NPZ blocks, which are not in the public 128 package",
    },
    "cross_modal_retrieval": {
        "primary_metric": "mrr",
        "reason": "requires paired motion/IMU/camera/audio/depth feature blocks, which are not in the public 128 package",
    },
    "modality_reconstruction": {
        "primary_metric": "r2",
        "reason": "requires source and target modality feature blocks such as depth/video vectors, which are not in the public 128 package",
    },
    "misalignment_detection": {
        "primary_metric": "f1",
        "reason": "requires deliberately shifted cross-modal feature pairs, which cannot be reconstructed from the public JSONL labels alone",
    },
}

DEFAULT_DATASET = ROOT / "tmp/omni_128_dataset_fetch/dataset.jsonl"
DEFAULT_PACKAGE = (
    ROOT
    / "results/omni_finetune/verified_public"
    / "xperience10m_qwen3_omni_128ep_96train_16val_16test_valmon_20260605_eval"
)
TOKEN_RE = re.compile(r"[a-z0-9]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-jsonl", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--verified-package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--output-dir", type=Path, default=Path("results/omni_finetune/multi_episode_128_task_baselines"))
    parser.add_argument("--hash-dim", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--learning-rate", type=float, default=0.16)
    parser.add_argument("--l2", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--softmax-max-train-classes",
        type=int,
        default=256,
        help="Use centroid classification instead of dense softmax when the train label space is larger than this.",
    )
    parser.add_argument("--include-neural", action="store_true", default=True)
    parser.add_argument("--neural-epochs", type=int, default=35)
    parser.add_argument("--neural-hidden-dim", type=int, default=128)
    parser.add_argument("--neural-batch-size", type=int, default=256)
    parser.add_argument("--neural-learning-rate", type=float, default=1e-3)
    parser.add_argument("--neural-weight-decay", type=float, default=1e-4)
    parser.add_argument("--neural-dropout", type=float, default=0.10)
    parser.add_argument("--neural-device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-object-vocab", type=int, default=256)
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(data), indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def log(message: str) -> None:
    print(f"[128-baselines] {message}", file=sys.stderr, flush=True)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compact_options(values: Any, max_items: int = 24, max_item_chars: int = 80) -> tuple[list[str], int]:
    if not isinstance(values, list):
        return [], 0
    compacted = []
    for value in values[:max_items]:
        text = str(value or "").strip()
        compacted.append(text[:max_item_chars])
    return compacted, len(values)


def compact_dataset_row(row: dict[str, Any]) -> dict[str, Any]:
    media = row.get("media") or {}
    action_options, action_option_count = compact_options(row.get("action_options", []))
    subtask_options, subtask_option_count = compact_options(row.get("subtask_options", []))
    video_paths = []
    for item in media.get("video_paths", []) or []:
        if isinstance(item, dict):
            video_paths.append({"name": item.get("name")})
    return {
        "id": row.get("id"),
        "episode_id": row.get("episode_id"),
        "split": row.get("split"),
        "question": row.get("question"),
        "prompt_type": row.get("prompt_type"),
        "action_options": action_options,
        "action_option_count": action_option_count,
        "subtask_options": subtask_options,
        "subtask_option_count": subtask_option_count,
        "center_window": row.get("center_window") or {},
        "media": {
            "context_start_frame": media.get("context_start_frame"),
            "context_end_frame": media.get("context_end_frame"),
            "video_paths": video_paths,
        },
        "sensor_feature_index": row.get("sensor_feature_index"),
        "sensor_feature_dim": row.get("sensor_feature_dim"),
        "answer_json": row.get("answer_json"),
        "true_json": row.get("true_json"),
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(compact_dataset_row(json.loads(line)))
    return rows


def norm(value: Any) -> str:
    return str(value or "").strip()


def portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def stable_hash(text: str, modulo: int) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % modulo


def hash_tokens(text: str, dim: int, namespace: str, scale: float = 1.0) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for token in TOKEN_RE.findall(text.lower()):
        idx = stable_hash(f"{namespace}:{token}", dim)
        sign = 1.0 if stable_hash(f"sign:{namespace}:{token}", 2) == 0 else -1.0
        vec[idx] += sign * scale
    norm_value = float(np.linalg.norm(vec))
    if norm_value > 0:
        vec /= norm_value
    return vec


def answer(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("answer_json")
    if isinstance(value, dict):
        return value
    value = row.get("true_json")
    return value if isinstance(value, dict) else {}


def bounded_text(value: Any, max_chars: int = 2048) -> str:
    text = norm(value)
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def option_summary(values: list[Any], total_count: int | None = None, max_items: int = 24, max_item_chars: int = 80) -> str:
    values = values or []
    total = len(values) if total_count is None else total_count
    head = " ".join(bounded_text(value, max_item_chars) for value in values[:max_items])
    return f"count_{total} {head}".strip()


def load_episode_context(package_dir: Path) -> dict[str, dict[str, Any]]:
    manifest_path = package_dir / "dataset/episode_manifest.json"
    if not manifest_path.exists():
        return {}
    manifest = load_json(manifest_path)
    return {str(ep.get("episode_id")): ep for ep in manifest.get("episodes", [])}


def row_text_features(row: dict[str, Any], episode: dict[str, Any] | None) -> str:
    parts = [
        "question:",
        bounded_text(row.get("question"), 2048),
        "prompt_type:",
        norm(row.get("prompt_type")),
        "action_options:",
        option_summary(row.get("action_options", []), row.get("action_option_count")),
        "subtask_options:",
        option_summary(row.get("subtask_options", []), row.get("subtask_option_count")),
    ]
    if episode:
        parts.extend([
            "main_task:",
            norm(episode.get("main_task")),
            "episode_split:",
            norm(episode.get("split")),
        ])
    media = row.get("media") or {}
    parts.extend([
        "media_names:",
        " ".join(norm(item.get("name")) for item in media.get("video_paths", []) if isinstance(item, dict)),
    ])
    return " ".join(parts)


def build_feature_matrix(rows: list[dict[str, Any]], episodes: dict[str, dict[str, Any]], hash_dim: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
    by_episode: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_episode.setdefault(str(row.get("episode_id")), []).append(row)
    episode_max_end = {}
    for episode_id, items in by_episode.items():
        episode_max_end[episode_id] = max(int((item.get("center_window") or {}).get("end_frame", 0) or 0) for item in items)

    features = []
    feature_rows = []
    for row in rows:
        episode_id = str(row.get("episode_id"))
        episode = episodes.get(episode_id)
        window = row.get("center_window") or {}
        media = row.get("media") or {}
        start = float(window.get("start_frame", 0) or 0)
        end = float(window.get("end_frame", start) or start)
        center = (start + end) / 2.0
        max_end = max(float(episode_max_end.get(episode_id, end)), 1.0)
        context_start = float(media.get("context_start_frame", start) or 0)
        context_end = float(media.get("context_end_frame", end) or end)
        numeric = np.asarray(
            [
                start / max_end,
                end / max_end,
                center / max_end,
                (end - start + 1.0) / max(max_end, 1.0),
                context_start / max_end,
                context_end / max_end,
                float(row.get("sensor_feature_index", 0) or 0) / max(len(by_episode.get(episode_id, [])), 1),
                float(row.get("sensor_feature_dim", 0) or 0) / 10000.0,
                float(row.get("action_option_count", len(row.get("action_options", []))) or 0) / 2000.0,
                float(row.get("subtask_option_count", len(row.get("subtask_options", []))) or 0) / 2000.0,
            ],
            dtype=np.float32,
        )
        text_vec = hash_tokens(row_text_features(row, episode), hash_dim, "row_text")
        features.append(np.concatenate([numeric, text_vec]).astype(np.float32))
        feature_rows.append(
            {
                "id": row.get("id"),
                "episode_id": episode_id,
                "split": row.get("split"),
                "start_frame": int(start),
                "end_frame": int(end),
                "main_task": norm((episode or {}).get("main_task")),
            }
        )
    return np.stack(features).astype(np.float32), feature_rows


def split_indices(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    indices: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for idx, row in enumerate(rows):
        split = str(row.get("split"))
        if split in indices:
            indices[split].append(idx)
    return {key: np.asarray(value, dtype=np.int64) for key, value in indices.items()}


def encode_labels(values: list[str]) -> tuple[np.ndarray, list[str]]:
    seen: OrderedDict[str, int] = OrderedDict()
    for value in values:
        if value not in seen:
            seen[value] = len(seen)
    return np.asarray([seen[value] for value in values], dtype=np.int64), list(seen.keys())


def encode_labels_train_first(values: list[str], train_idx: np.ndarray) -> tuple[np.ndarray, list[str], int]:
    seen: OrderedDict[str, int] = OrderedDict()
    for idx in train_idx:
        value = values[int(idx)]
        if value not in seen:
            seen[value] = len(seen)
    train_class_count = len(seen)
    for value in values:
        if value not in seen:
            seen[value] = len(seen)
    return np.asarray([seen[value] for value in values], dtype=np.int64), list(seen.keys()), train_class_count


def evaluate_classifier(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    metrics, per_class, cm = compute_metrics(y_true, y_pred, class_names)
    return {"metrics": metrics, "per_class": per_class, "confusion": cm}


def write_confusion(path: Path, cm: np.ndarray, class_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["true\\pred"] + class_names)
        for idx, name in enumerate(class_names):
            writer.writerow([name] + [int(v) for v in cm[idx]])


def probabilities_from_scores(scores: np.ndarray) -> np.ndarray:
    scores = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(scores)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


def fit_centroid_classifier(X: np.ndarray, y: np.ndarray, train_idx: np.ndarray, train_class_count: int) -> tuple[np.ndarray, np.ndarray]:
    centroids = []
    priors = []
    for class_id in range(train_class_count):
        class_rows = train_idx[y[train_idx] == class_id]
        if len(class_rows) == 0:
            centroids.append(np.zeros(X.shape[1], dtype=np.float32))
            priors.append(0.0)
            continue
        centroids.append(X[class_rows].mean(axis=0).astype(np.float32))
        priors.append(float(len(class_rows)))
    centroids_arr = np.stack(centroids).astype(np.float32)
    centroid_norm = np.linalg.norm(centroids_arr, axis=1, keepdims=True)
    centroids_arr = centroids_arr / np.maximum(centroid_norm, 1e-6)
    priors_arr = np.asarray(priors, dtype=np.float32)
    priors_arr = priors_arr / max(float(priors_arr.sum()), 1.0)
    return centroids_arr, priors_arr


def predict_centroid(X: np.ndarray, centroids: np.ndarray, priors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-6)
    scores = Xn @ centroids.T
    if len(priors):
        scores = scores + 0.03 * np.log(np.maximum(priors, 1e-9))[None, :]
    prob = probabilities_from_scores(scores)
    return np.argmax(prob, axis=1).astype(np.int64), prob.astype(np.float32)


def classification_baseline(
    task_id: str,
    rows: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    X: np.ndarray,
    splits: dict[str, np.ndarray],
    label_getter: Callable[[dict[str, Any]], str],
    out_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    labels = [label_getter(row) for row in rows]
    valid = np.asarray([bool(label) for label in labels])
    if not all(valid):
        keep = np.flatnonzero(valid)
    else:
        keep = np.arange(len(rows), dtype=np.int64)
    local = {int(global_idx): local_idx for local_idx, global_idx in enumerate(keep)}
    Xv = X[keep]
    label_values = [labels[int(i)] for i in keep]
    split_local = {
        name: np.asarray([local[int(idx)] for idx in values if int(idx) in local], dtype=np.int64)
        for name, values in splits.items()
    }
    train_idx = split_local["train"]
    val_idx = split_local["val"]
    test_idx = split_local["test"]
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError(f"{task_id}: train/test split is empty")
    y, class_names, train_class_count = encode_labels_train_first(label_values, train_idx)
    log(f"{task_id}: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test, {train_class_count} train classes, {len(class_names)} total classes")

    mean, std = fit_scaler(Xv[train_idx])
    Xs = (Xv - mean) / std
    if train_class_count <= args.softmax_max_train_classes:
        classifier_kind = "softmax"
        log(f"{task_id}: fitting softmax")
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
        centroid_model = None
    else:
        classifier_kind = "centroid"
        log(f"{task_id}: fitting centroid classifier")
        W, b = None, None
        centroid_model = fit_centroid_classifier(Xs, y, train_idx, train_class_count)
        history = [
            {
                "method": "train_class_centroid_cosine",
                "reason": f"train_class_count={train_class_count} exceeds softmax_max_train_classes={args.softmax_max_train_classes}",
            }
        ]

    simple_dir = out_dir / task_id
    simple_dir.mkdir(parents=True, exist_ok=True)
    split_metrics = {}
    test_pred_rows = []
    split_order = [("val", val_idx), ("test", test_idx)]
    if train_class_count <= args.softmax_max_train_classes:
        split_order.append(("train", train_idx))
    for split_name, idx in split_order:
        log(f"{task_id}: scoring {split_name}")
        if classifier_kind == "softmax":
            pred, prob = predict(Xs[idx], W, b)
        else:
            assert centroid_model is not None
            pred, prob = predict_centroid(Xs[idx], centroid_model[0], centroid_model[1])
        eval_payload = evaluate_classifier(y[idx], pred, class_names)
        split_metrics[split_name] = eval_payload["metrics"]
        if split_name == "test":
            write_csv(simple_dir / "per_class_metrics.csv", eval_payload["per_class"])
            write_confusion(simple_dir / "confusion_matrix.csv", eval_payload["confusion"], class_names)
            for k, local_idx in enumerate(idx):
                global_idx = int(keep[int(local_idx)])
                pred_id = int(pred[k])
                true_id = int(y[int(local_idx)])
                test_pred_rows.append(
                    {
                        **feature_rows[global_idx],
                        "true_label": class_names[true_id],
                        "predicted_label": class_names[pred_id],
                        "confidence": float(prob[k, pred_id]),
                        "correct": int(pred_id == true_id),
                    }
                )
    majority_class = Counter(y[train_idx]).most_common(1)[0][0]
    majority_accuracy = float(np.mean(y[test_idx] == majority_class))
    unseen_ids = sorted(set(int(v) for v in y[test_idx]) - set(int(v) for v in y[train_idx]))
    metrics = {
        "status": "pass",
        "task": task_id,
        "task_display_name": task_display_name(task_id),
        "model_family": f"simple_{classifier_kind}_metadata",
        "source": "128_episode_qwen_jsonl_metadata",
        "input_features": "frame/context metadata plus hashed prompt/options/main_task text; answer_json fields are excluded from inputs",
        "split_policy": "train on train split, report val and held-out test split",
        "num_train_windows": int(len(train_idx)),
        "num_val_windows": int(len(val_idx)),
        "num_test_windows": int(len(test_idx)),
        "num_classes": int(len(class_names)),
        "num_train_classes": int(train_class_count),
        "majority_baseline_accuracy": majority_accuracy,
        "history": history,
        "splits": split_metrics,
        "primary_metric": "macro_f1",
        "primary_score": split_metrics["test"].get("macro_f1"),
        "unseen_test_class_count": len(unseen_ids),
        "unseen_test_classes": [class_names[class_id] for class_id in unseen_ids],
    }
    log(f"{task_id}: writing simple artifacts")
    write_json(simple_dir / "metrics.json", metrics)
    write_csv(simple_dir / "predictions.csv", test_pred_rows)
    if classifier_kind == "softmax":
        np.savez_compressed(
            simple_dir / "model.npz",
            mean=mean,
            std=std,
            W=W,
            b=b,
            class_names=np.asarray(class_names, dtype=object),
            train_class_count=np.asarray(train_class_count, dtype=np.int64),
        )
    else:
        assert centroid_model is not None
        np.savez_compressed(
            simple_dir / "model.npz",
            mean=mean,
            std=std,
            centroids=centroid_model[0],
            priors=centroid_model[1],
            class_names=np.asarray(class_names, dtype=object),
            train_class_count=np.asarray(train_class_count, dtype=np.int64),
        )

    neural_result = None
    if args.include_neural:
        neural_dir = simple_dir.parent / "neural_mlp" / task_id
        try:
            log(f"{task_id}: fitting neural MLP")
            neural_result = neural_classification(
                task_id,
                Xv,
                y,
                class_names,
                train_class_count,
                train_idx,
                val_idx,
                test_idx,
                keep,
                feature_rows,
                neural_dir,
                args,
            )
        except Exception as exc:  # pragma: no cover - protects long batch runs from optional NN environment failures.
            neural_result = {
                "status": "failed",
                "task": task_id,
                "task_display_name": task_display_name(task_id),
                "model_family": "neural_mlp_metadata",
                "source": "128_episode_qwen_jsonl_metadata",
                "primary_metric": "macro_f1",
                "primary_score": None,
                "error": str(exc),
            }
            write_json(neural_dir / "metrics.json", neural_result)
        log(f"{task_id}: done")
    return {"simple": metrics, "neural": neural_result}


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


def neural_classification(
    task_id: str,
    X: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    train_class_count: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    keep: np.ndarray,
    feature_rows: list[dict[str, Any]],
    out_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    from neural_task_models import train_classifier

    out_dir.mkdir(parents=True, exist_ok=True)
    result = train_classifier(
        X.astype(np.float32),
        y,
        train_idx,
        test_idx,
        n_classes=train_class_count,
        config=neural_config(args),
        use_class_weights=True,
    )
    split_metrics = {}
    # The helper returns test predictions only; run a small second pass is avoided
    # here to keep artifacts public-safe and runtime bounded.  Train/val counts are
    # still recorded for split alignment.
    eval_payload = evaluate_classifier(y[test_idx], result["pred"], class_names)
    split_metrics["test"] = eval_payload["metrics"]
    pred_rows = []
    for k, local_idx in enumerate(test_idx):
        global_idx = int(keep[int(local_idx)])
        pred_id = int(result["pred"][k])
        true_id = int(y[int(local_idx)])
        pred_rows.append(
            {
                **feature_rows[global_idx],
                "true_label": class_names[true_id],
                "predicted_label": class_names[pred_id],
                "correct": int(pred_id == true_id),
            }
        )
    metrics = {
        "status": "pass",
        "task": task_id,
        "task_display_name": task_display_name(task_id),
        "model_family": "neural_mlp_metadata",
        "source": "128_episode_qwen_jsonl_metadata",
        "input_features": "frame/context metadata plus hashed prompt/options/main_task text; answer_json fields are excluded from inputs",
        "split_policy": "train on train split, report held-out test split",
        "num_train_windows": int(len(train_idx)),
        "num_val_windows": int(len(val_idx)),
        "num_test_windows": int(len(test_idx)),
        "num_classes": int(len(class_names)),
        "num_train_classes": int(train_class_count),
        "history": result["history"],
        "device": result["device"],
        "splits": split_metrics,
        "primary_metric": "macro_f1",
        "primary_score": split_metrics["test"].get("macro_f1"),
    }
    write_json(out_dir / "metrics.json", metrics)
    write_csv(out_dir / "predictions.csv", pred_rows)
    write_csv(out_dir / "per_class_metrics.csv", eval_payload["per_class"])
    write_confusion(out_dir / "confusion_matrix.csv", eval_payload["confusion"], class_names)
    return metrics


def object_matrix(rows: list[dict[str, Any]], vocab: list[str]) -> np.ndarray:
    index = {name: i for i, name in enumerate(vocab)}
    Y = np.zeros((len(rows), len(vocab)), dtype=np.float32)
    for row_idx, row in enumerate(rows):
        for obj in answer(row).get("objects", []) or []:
            key = norm(obj).lower()
            if key in index:
                Y[row_idx, index[key]] = 1.0
    return Y


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


def simple_multilabel(
    rows: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    X: np.ndarray,
    splits: dict[str, np.ndarray],
    out_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    train_objects = Counter()
    for idx in splits["train"]:
        for obj in answer(rows[int(idx)]).get("objects", []) or []:
            key = norm(obj).lower()
            if key:
                train_objects[key] += 1
    vocab = [name for name, _count in train_objects.most_common(args.max_object_vocab)]
    Y = object_matrix(rows, vocab)
    train_idx, val_idx, test_idx = splits["train"], splits["val"], splits["test"]
    freq = Y[train_idx].mean(axis=0)
    # Public-safe simple baseline: predict object labels seen in at least 10% of
    # training windows.  This mirrors a frequency prior instead of using target
    # labels from the held-out split.
    pred_all = np.tile((freq >= 0.10).astype(np.float32), (len(rows), 1))
    test_metrics = multilabel_metrics(Y[test_idx], pred_all[test_idx])
    val_metrics = multilabel_metrics(Y[val_idx], pred_all[val_idx]) if len(val_idx) else {}
    out_dir = out_root / "object_relevance"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_rows = []
    for idx in test_idx:
        idx = int(idx)
        true_objs = [vocab[i] for i in np.flatnonzero(Y[idx] > 0)]
        pred_objs = [vocab[i] for i in np.flatnonzero(pred_all[idx] > 0)]
        pred_rows.append({**feature_rows[idx], "true_objects": ";".join(true_objs), "predicted_objects": ";".join(pred_objs)})
    metrics = {
        "status": "pass",
        "task": "object_relevance",
        "task_display_name": task_display_name("object_relevance"),
        "model_family": "simple_train_object_frequency",
        "source": "128_episode_qwen_jsonl_metadata",
        "split_policy": "object vocabulary and frequencies are learned from train split only",
        "num_train_windows": int(len(train_idx)),
        "num_val_windows": int(len(val_idx)),
        "num_test_windows": int(len(test_idx)),
        "num_objects": int(len(vocab)),
        "splits": {"val": val_metrics, "test": test_metrics},
        "primary_metric": "micro_f1",
        "primary_score": test_metrics["micro_f1"],
    }
    write_json(out_dir / "metrics.json", metrics)
    write_json(out_dir / "object_vocab.json", vocab)
    write_csv(out_dir / "predictions.csv", pred_rows)

    neural_result = None
    if args.include_neural:
        neural_dir = out_root / "neural_mlp" / "object_relevance"
        try:
            neural_result = neural_multilabel(rows, feature_rows, X, Y, splits, neural_dir, args, vocab)
        except Exception as exc:  # pragma: no cover - protects long batch runs from optional NN environment failures.
            neural_result = {
                "status": "failed",
                "task": "object_relevance",
                "task_display_name": task_display_name("object_relevance"),
                "model_family": "neural_mlp_metadata_multilabel",
                "source": "128_episode_qwen_jsonl_metadata",
                "primary_metric": "micro_f1",
                "primary_score": None,
                "error": str(exc),
            }
            write_json(neural_dir / "metrics.json", neural_result)
    return {"simple": metrics, "neural": neural_result}


def neural_multilabel(
    rows: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    X: np.ndarray,
    Y: np.ndarray,
    splits: dict[str, np.ndarray],
    out_dir: Path,
    args: argparse.Namespace,
    vocab: list[str],
) -> dict[str, Any]:
    from neural_task_models import train_multilabel

    out_dir.mkdir(parents=True, exist_ok=True)
    result = train_multilabel(X.astype(np.float32), Y, splits["train"], splits["test"], neural_config(args))
    test_metrics = multilabel_metrics(Y[splits["test"]], result["pred"])
    pred_rows = []
    for local_k, idx in enumerate(splits["test"]):
        idx = int(idx)
        true_objs = [vocab[i] for i in np.flatnonzero(Y[idx] > 0)]
        pred_objs = [vocab[i] for i in np.flatnonzero(result["pred"][local_k] > 0)]
        pred_rows.append({**feature_rows[idx], "true_objects": ";".join(true_objs), "predicted_objects": ";".join(pred_objs)})
    metrics = {
        "status": "pass",
        "task": "object_relevance",
        "task_display_name": task_display_name("object_relevance"),
        "model_family": "neural_mlp_metadata_multilabel",
        "source": "128_episode_qwen_jsonl_metadata",
        "num_train_windows": int(len(splits["train"])),
        "num_val_windows": int(len(splits["val"])),
        "num_test_windows": int(len(splits["test"])),
        "num_objects": int(len(vocab)),
        "history": result["history"],
        "device": result["device"],
        "splits": {"test": test_metrics},
        "primary_metric": "micro_f1",
        "primary_score": test_metrics["micro_f1"],
    }
    write_json(out_dir / "metrics.json", metrics)
    write_csv(out_dir / "predictions.csv", pred_rows)
    return metrics


def caption_grounding(rows: list[dict[str, Any]], episodes: dict[str, dict[str, Any]], splits: dict[str, np.ndarray], out_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    test_idx = splits["test"]
    candidate_texts = []
    query_texts = []
    for idx in test_idx:
        row = rows[int(idx)]
        ans = answer(row)
        episode = episodes.get(str(row.get("episode_id")))
        candidate_texts.append(row_text_features(row, episode))
        query_texts.append(" ".join([norm(ans.get("action")), norm(ans.get("subtask")), " ".join(norm(x) for x in ans.get("objects", []) or [])]))
    C = np.stack([hash_tokens(text, args.hash_dim, "caption_candidate") for text in candidate_texts])
    Q = np.stack([hash_tokens(text, args.hash_dim, "caption_query") for text in query_texts])
    scores = Q @ C.T
    ranks = []
    rank_rows = []
    for i in range(len(test_idx)):
        order = np.argsort(-scores[i])
        rank = int(np.flatnonzero(order == i)[0]) + 1
        row_id = rows[int(test_idx[i])].get("id")
        top1_row_id = rows[int(test_idx[int(order[0])])].get("id")
        ranks.append(rank)
        rank_rows.append(
            {
                "query_index": i,
                "row_id": row_id,
                "rank": rank,
                "top1_row_id": top1_row_id,
            }
        )
    ranks_arr = np.asarray(ranks, dtype=np.float32)
    metrics = {
        "status": "pass",
        "task": "caption_grounding",
        "task_display_name": task_display_name("caption_grounding"),
        "model_family": "simple_hashed_text_retrieval",
        "source": "128_episode_qwen_jsonl_metadata",
        "limitation": "query text is derived from held-out labels, while candidate representations exclude answer_json fields; this is a public-safe retrieval proxy, not a raw caption-to-sensor grounding model",
        "num_queries": int(len(test_idx)),
        "mrr": float(np.mean(1.0 / ranks_arr)) if len(ranks_arr) else 0.0,
        "median_rank": float(np.median(ranks_arr)) if len(ranks_arr) else 0.0,
        "mean_rank": float(np.mean(ranks_arr)) if len(ranks_arr) else 0.0,
        "top1_accuracy": float(np.mean(ranks_arr <= 1)) if len(ranks_arr) else 0.0,
        "top5_accuracy": float(np.mean(ranks_arr <= 5)) if len(ranks_arr) else 0.0,
        "top10_accuracy": float(np.mean(ranks_arr <= 10)) if len(ranks_arr) else 0.0,
        "primary_metric": "mrr",
        "primary_score": float(np.mean(1.0 / ranks_arr)) if len(ranks_arr) else 0.0,
    }
    out_dir = out_root / "caption_grounding"
    write_json(out_dir / "metrics.json", metrics)
    write_csv(out_dir / "ranks.csv", rank_rows)
    return {"simple": metrics, "neural": None}


def temporal_order(rows: list[dict[str, Any]], X: np.ndarray, splits: dict[str, np.ndarray], out_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    rng = np.random.default_rng(args.seed)

    def make_pairs(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
        by_episode: dict[str, list[int]] = {}
        for idx in indices:
            by_episode.setdefault(str(rows[int(idx)].get("episode_id")), []).append(int(idx))
        feats = []
        labels = []
        pair_rows = []
        for episode_id, items in by_episode.items():
            items = sorted(items, key=lambda k: int((rows[k].get("center_window") or {}).get("start_frame", 0) or 0))
            for a, b in zip(items, items[1:]):
                if rng.random() < 0.5:
                    first, second, label = a, b, 1
                else:
                    first, second, label = b, a, 0
                feats.append(np.concatenate([X[first], X[second], np.abs(X[first] - X[second])]))
                labels.append(label)
                pair_rows.append({"episode_id": episode_id, "first_id": rows[first].get("id"), "second_id": rows[second].get("id"), "true_order": label})
        return np.stack(feats).astype(np.float32), np.asarray(labels, dtype=np.int64), pair_rows

    X_train, y_train, _train_rows = make_pairs(splits["train"])
    X_test, y_test, test_pair_rows = make_pairs(splits["test"])
    X_all = np.concatenate([X_train, X_test], axis=0)
    mean, std = fit_scaler(X_train)
    Xs = (X_all - mean) / std
    train_idx = np.arange(len(X_train), dtype=np.int64)
    test_idx = np.arange(len(X_train), len(X_train) + len(X_test), dtype=np.int64)
    W, b, history = train_softmax_classifier(Xs[train_idx], y_train, 2, args.epochs, args.learning_rate, args.l2, True, args.seed)
    pred, prob = predict(Xs[test_idx], W, b)
    metrics, per_class, cm = compute_metrics(y_test, pred, ["reversed", "correct"])
    pred_rows = []
    for k, row in enumerate(test_pair_rows):
        pred_rows.append({**row, "predicted_order": int(pred[k]), "confidence": float(prob[k, int(pred[k])]), "correct": int(pred[k] == y_test[k])})
    payload = {
        "status": "pass",
        "task": "temporal_order",
        "task_display_name": task_display_name("temporal_order"),
        "model_family": "simple_softmax_metadata_pair",
        "source": "128_episode_qwen_jsonl_metadata",
        "limitation": "metadata-only pair features are weaker than the single-episode raw-feature temporal-order task",
        "num_train_samples": int(len(train_idx)),
        "num_test_samples": int(len(test_idx)),
        "history": history,
        **metrics,
        "primary_metric": "f1",
        "primary_score": metrics["macro_f1"],
    }
    out_dir = out_root / "temporal_order"
    write_json(out_dir / "metrics.json", payload)
    write_csv(out_dir / "predictions.csv", pred_rows)
    write_csv(out_dir / "per_class_metrics.csv", per_class)
    write_confusion(out_dir / "confusion_matrix.csv", cm, ["reversed", "correct"])
    return {"simple": payload, "neural": None}


def unsupported_record(task_id: str, out_root: Path, reason: str, primary_metric: str) -> dict[str, Any]:
    payload = {
        "status": "unsupported_without_raw_128_feature_blocks",
        "task": task_id,
        "task_display_name": task_display_name(task_id),
        "primary_metric": primary_metric,
        "primary_score": None,
        "reason": reason,
        "source": "128_episode_qwen_jsonl_metadata",
    }
    write_json(out_root / task_id / "metrics.json", payload)
    return {"simple": payload, "neural": None}


def build_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# 128-Episode Aligned Baselines",
        "",
        "These results align the earlier simple and neural baseline framing to the same selected 128-episode split used by the Qwen3-Omni pilot.",
        "",
        "The runner uses the derived Qwen JSONL export and public-safe metadata. It does not use raw Xperience-10M videos, HDF5 files, sensor NPZ blocks, Qwen weights, or LoRA weights.",
        "",
        "## Split",
        "",
        f"- Train windows: `{summary['split_counts']['train']}`",
        f"- Validation windows: `{summary['split_counts']['val']}`",
        f"- Test windows: `{summary['split_counts']['test']}`",
        f"- Exported episodes: `{summary['episode_counts']}`",
        "",
        "## Coverage",
        "",
        "| task | artifact id | simple status | simple primary | neural status | neural primary |",
        "| --- | --- | --- | ---: | --- | ---: |",
    ]
    for task in summary["tasks"]:
        simple = task.get("simple") or {}
        neural = task.get("neural") or {}
        simple_score = simple.get("primary_score")
        neural_score = neural.get("primary_score") if neural else None
        lines.append(
            "| {task} | `{artifact}` | {simple_status} | {simple_score} | {neural_status} | {neural_score} |".format(
                task=task.get("task_display_name") or task_display_name(task["task"]),
                artifact=task["task"],
                simple_status=simple.get("status", ""),
                simple_score="" if simple_score is None else f"{float(simple_score):.4f}",
                neural_status=neural.get("status", "not_run") if neural else "not_run",
                neural_score="" if neural_score is None else f"{float(neural_score):.4f}",
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The trainable scores are metadata/text baselines, not replacements for full raw-modality baselines. They are useful for checking split alignment, label difficulty, train/test label coverage, and whether the Qwen diagnostic run is being compared against the same 96/16/16 episode setup.",
            "",
            "Tasks marked `unsupported_without_raw_128_feature_blocks` still need the 128-run sensor feature NPZ blocks to reproduce the single-episode feature-level target exactly.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    if not args.dataset_jsonl.exists():
        raise FileNotFoundError(
            f"Missing 128 JSONL export: {args.dataset_jsonl}. Fetch the derived dataset.jsonl from the training host or rerun the 128 export."
        )
    if not args.verified_package.exists():
        raise FileNotFoundError(f"Missing verified package: {args.verified_package}")
    log(f"loading compact rows from {portable_path(args.dataset_jsonl)}")
    rows = load_jsonl(args.dataset_jsonl)
    log(f"loaded {len(rows)} rows")
    episodes = load_episode_context(args.verified_package)
    log("building metadata feature matrix")
    X, feature_rows = build_feature_matrix(rows, episodes, args.hash_dim)
    log(f"built feature matrix {X.shape[0]}x{X.shape[1]}")
    splits = split_indices(rows)
    split_counts = {key: int(len(value)) for key, value in splits.items()}
    expected = {"train": 2848, "val": 512, "test": 448}
    if split_counts != expected:
        raise ValueError(f"Dataset split mismatch: observed {split_counts}, expected {expected}")

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    log(f"writing shared artifacts to {portable_path(out)}")
    write_csv(out / "windows.csv", feature_rows)
    np.savez_compressed(
        out / "metadata_feature_matrix.npz",
        X=X,
        split=np.asarray([row.get("split") for row in rows], dtype=object),
        row_id=np.asarray([row.get("id") for row in rows], dtype=object),
    )

    task_results = []
    for task_id, (key, _metric) in CLASSIFICATION_TASKS.items():
        log(f"{task_id}: start")
        result = classification_baseline(task_id, rows, feature_rows, X, splits, lambda row, k=key: norm(answer(row).get(k)), out, args)
        task_results.append({"task": task_id, "task_display_name": task_display_name(task_id), **result})
    log("object_relevance: start")
    task_results.append({"task": "object_relevance", "task_display_name": task_display_name("object_relevance"), **simple_multilabel(rows, feature_rows, X, splits, out, args)})
    log("caption_grounding: start")
    task_results.append({"task": "caption_grounding", "task_display_name": task_display_name("caption_grounding"), **caption_grounding(rows, episodes, splits, out, args)})
    log("temporal_order: start")
    task_results.append({"task": "temporal_order", "task_display_name": task_display_name("temporal_order"), **temporal_order(rows, X, splits, out, args)})
    for task_id, spec in UNSUPPORTED_TASKS.items():
        log(f"{task_id}: recording unsupported status")
        task_results.append({"task": task_id, "task_display_name": task_display_name(task_id), **unsupported_record(task_id, out, spec["reason"], spec["primary_metric"])})

    task_results = sorted(task_results, key=lambda row: TASKS.index(row["task"]))
    episode_counts = Counter(str(row.get("split")) for row in episodes.values())
    summary = {
        "status": "pass",
        "run_id": "xperience10m_128_episode_aligned_task_baselines",
        "source_dataset_jsonl": portable_path(args.dataset_jsonl),
        "verified_package": portable_path(args.verified_package),
        "source_policy": "derived JSONL metadata only; fetched dataset JSONL is not committed",
        "num_rows": len(rows),
        "split_counts": split_counts,
        "episode_counts": dict(sorted(episode_counts.items())),
        "task_display_names": {task: task_display_name(task) for task in TASKS},
        "feature_contract": {
            "kind": "metadata_text_hash",
            "hash_dim": args.hash_dim,
            "numeric_dim": 10,
            "answer_json_used_as_input": False,
        },
        "run_config": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "l2": args.l2,
            "softmax_max_train_classes": args.softmax_max_train_classes,
            "neural_epochs": args.neural_epochs,
            "neural_hidden_dim": args.neural_hidden_dim,
            "neural_batch_size": args.neural_batch_size,
            "neural_learning_rate": args.neural_learning_rate,
            "neural_weight_decay": args.neural_weight_decay,
            "neural_dropout": args.neural_dropout,
            "neural_device": args.neural_device,
        },
        "tasks": task_results,
    }
    write_json(out / "summary_report.json", summary)
    write_csv(
        out / "task_metrics.csv",
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
            for row in task_results
        ],
    )
    (out / "BASELINE_ALIGNMENT_REPORT.md").write_text(build_markdown(summary), encoding="utf-8")
    print(json.dumps({"status": "pass", "output_dir": str(out), "split_counts": split_counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

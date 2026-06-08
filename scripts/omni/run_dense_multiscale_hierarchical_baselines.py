#!/usr/bin/env python3
"""Run compact baselines on dense/multiscale hierarchical 128-episode labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from neural_task_models import NeuralConfig, save_torch_model, train_classifier
from train_min_action_model import compute_metrics, fit_scaler, predict, train_softmax_classifier


DEFAULT_DATASET_DIR = ROOT / "results/omni_finetune/xperience10m_128ep_dense_multiscale_hierarchical_v1_20260608"
TARGETS = [
    "action_family",
    "action",
    "subtask_family",
    "subtask",
    "next_action_family",
    "next_action",
    "contact",
    "transition",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-jsonl", type=Path, default=DEFAULT_DATASET_DIR / "dense_multiscale_windows.jsonl")
    parser.add_argument("--dataset-manifest", type=Path, default=DEFAULT_DATASET_DIR / "dataset_manifest.json")
    parser.add_argument("--run-id", default="xperience10m_128ep_dense_multiscale_hierarchical_baselines_v1_20260608")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/omni_finetune")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--hash-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--learning-rate", type=float, default=0.16)
    parser.add_argument("--l2", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--softmax-max-train-classes", type=int, default=256)
    parser.add_argument("--include-neural", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--neural-epochs", type=int, default=45)
    parser.add_argument("--neural-hidden-dim", type=int, default=160)
    parser.add_argument("--neural-batch-size", type=int, default=1024)
    parser.add_argument("--neural-learning-rate", type=float, default=1e-3)
    parser.add_argument("--neural-weight-decay", type=float, default=1e-4)
    parser.add_argument("--neural-dropout", type=float, default=0.10)
    parser.add_argument("--neural-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[dense-hier-baselines] {message}", file=sys.stderr, flush=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def stable_hash(text: str, modulo: int) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % modulo


def hash_onehot(value: str, dim: int, namespace: str) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    idx = stable_hash(f"{namespace}:{value}", dim)
    sign = 1.0 if stable_hash(f"sign:{namespace}:{value}", 2) == 0 else -1.0
    vec[idx] = sign
    return vec


def build_features(rows: list[dict[str, Any]]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    max_end_by_episode: dict[str, int] = {}
    for row in rows:
        episode_id = str(row["episode_id"])
        end = int((row.get("center_window") or {}).get("end_frame", 0) or 0)
        max_end_by_episode[episode_id] = max(max_end_by_episode.get(episode_id, 0), end)
    scale_ids = sorted({str(row.get("scale_id")) for row in rows})
    scale_index = {scale: idx for idx, scale in enumerate(scale_ids)}

    features = []
    feature_rows = []
    for row in rows:
        window = row.get("center_window") or {}
        source = row.get("source_sparse_window") or {}
        episode_id = str(row["episode_id"])
        split = str(row.get("split"))
        scale = str(row.get("scale_id"))
        start = float(window.get("start_frame", 0) or 0)
        end = float(window.get("end_frame", start) or start)
        max_end = max(float(max_end_by_episode.get(episode_id, end)), 1.0)
        center = (start + end) / 2.0
        scale_vec = np.zeros(len(scale_ids), dtype=np.float32)
        scale_vec[scale_index[scale]] = 1.0
        provenance_vec = hash_onehot(str(source.get("kind", "")), 8, "provenance")
        media_vec = np.asarray(
            [
                float(bool(row.get("source_has_mosaic"))),
                float(bool(row.get("source_has_audio"))),
                float(len(row.get("source_media_names") or [])) / 6.0,
            ],
            dtype=np.float32,
        )
        numeric = np.asarray(
            [
                start / max_end,
                end / max_end,
                center / max_end,
                (end - start + 1.0) / max_end,
                float(row.get("window_frames", 0) or 0) / 100.0,
                float(row.get("stride_frames", 0) or 0) / 100.0,
                float(source.get("center_distance_frames", 0) or 0) / 100.0,
                float(source.get("overlap_fraction_of_dense", 0) or 0),
                float(source.get("overlap_fraction_of_source", 0) or 0),
                float(row.get("sensor_feature_dim", 0) or 0) / 10000.0,
            ],
            dtype=np.float32,
        )
        episode_hash = hash_onehot(episode_id, 32, "episode")
        split_hash = hash_onehot(split, 4, "split")
        # Episode/split hashes are included only for diagnostic baselines; the
        # held-out split remains episode-disjoint, so memorized train episodes
        # cannot directly cover validation/test episodes.
        feature = np.concatenate([numeric, media_vec, scale_vec, provenance_vec, episode_hash, split_hash]).astype(np.float32)
        features.append(feature)
        feature_rows.append({
            "id": row["id"],
            "episode_id": episode_id,
            "split": split,
            "scale_id": scale,
            "start_frame": int(start),
            "end_frame": int(end),
            "label_provenance": source.get("kind"),
        })
    return np.stack(features).astype(np.float32), feature_rows


def split_indices(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    splits = {"train": [], "val": [], "test": []}
    for idx, row in enumerate(rows):
        split = str(row.get("split"))
        if split in splits:
            splits[split].append(idx)
    return {split: np.asarray(indices, dtype=np.int64) for split, indices in splits.items()}


def encode_train_first(values: list[str], train_idx: np.ndarray) -> tuple[np.ndarray, list[str], int]:
    seen: OrderedDict[str, int] = OrderedDict()
    for idx in train_idx:
        value = values[int(idx)]
        if value not in seen:
            seen[value] = len(seen)
    train_class_count = len(seen)
    for value in values:
        if value not in seen:
            seen[value] = len(seen)
    return np.asarray([seen[value] for value in values], dtype=np.int64), list(seen), train_class_count


def probabilities_from_scores(scores: np.ndarray) -> np.ndarray:
    scores = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(scores)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


def fit_centroid_classifier(X: np.ndarray, y: np.ndarray, train_idx: np.ndarray, train_class_count: int) -> tuple[np.ndarray, np.ndarray]:
    centroids = []
    priors = []
    for class_id in range(train_class_count):
        members = train_idx[y[train_idx] == class_id]
        if len(members):
            centroids.append(X[members].mean(axis=0).astype(np.float32))
            priors.append(float(len(members)))
        else:
            centroids.append(np.zeros(X.shape[1], dtype=np.float32))
            priors.append(0.0)
    centroids_arr = np.stack(centroids).astype(np.float32)
    centroids_arr = centroids_arr / np.maximum(np.linalg.norm(centroids_arr, axis=1, keepdims=True), 1e-6)
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


def eval_split(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    metrics, per_class, cm = compute_metrics(y_true, y_pred, class_names)
    return metrics, per_class, cm


def write_confusion(path: Path, cm: np.ndarray, class_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["true\\pred", *class_names])
        for idx, name in enumerate(class_names):
            writer.writerow([name, *[int(v) for v in cm[idx]]])


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


def run_target(
    target: str,
    rows: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    X: np.ndarray,
    splits: dict[str, np.ndarray],
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    labels = [str((row.get("labels") or {}).get(target, "unknown") or "unknown") for row in rows]
    y, class_names, train_class_count = encode_train_first(labels, splits["train"])
    mean, std = fit_scaler(X[splits["train"]])
    Xs = (X - mean) / std
    target_dir = output_dir / "simple" / target
    target_dir.mkdir(parents=True, exist_ok=True)

    if train_class_count <= args.softmax_max_train_classes:
        model_kind = "softmax"
        log(f"{target}: softmax over {train_class_count} train classes")
        W, b, history = train_softmax_classifier(
            Xs[splits["train"]],
            y[splits["train"]],
            n_classes=train_class_count,
            epochs=args.epochs,
            lr=args.learning_rate,
            l2=args.l2,
            use_class_weights=True,
            seed=args.seed,
        )
        centroid = None
    else:
        model_kind = "centroid"
        log(f"{target}: centroid over {train_class_count} train classes")
        W, b = None, None
        centroid = fit_centroid_classifier(Xs, y, splits["train"], train_class_count)
        history = [{"method": "train_class_centroid_cosine", "reason": f"{train_class_count} train classes"}]

    split_metrics = {}
    predictions = []
    for split in ("val", "test", "train"):
        idx = splits[split]
        if len(idx) == 0:
            continue
        if model_kind == "softmax":
            pred, prob = predict(Xs[idx], W, b)
        else:
            assert centroid is not None
            pred, prob = predict_centroid(Xs[idx], centroid[0], centroid[1])
        metrics, per_class, cm = eval_split(y[idx], pred, class_names)
        split_metrics[split] = metrics
        if split == "test":
            write_csv(target_dir / "per_class_metrics.csv", per_class)
            write_confusion(target_dir / "confusion_matrix.csv", cm, class_names)
            for out_idx, global_idx in enumerate(idx):
                pred_id = int(pred[out_idx])
                true_id = int(y[int(global_idx)])
                predictions.append({
                    **feature_rows[int(global_idx)],
                    "target": target,
                    "true_label": class_names[true_id],
                    "predicted_label": class_names[pred_id],
                    "confidence": float(prob[out_idx, pred_id]),
                    "correct": int(pred_id == true_id),
                })
    unseen_test = sorted(set(int(v) for v in y[splits["test"]]) - set(int(v) for v in y[splits["train"]]))
    simple_metrics = {
        "status": "pass",
        "target": target,
        "model_family": f"simple_{model_kind}_dense_metadata",
        "input_features": "window position, scale id, provenance/media availability, and hashed episode/split diagnostics; target labels are excluded",
        "num_train_windows": int(len(splits["train"])),
        "num_val_windows": int(len(splits["val"])),
        "num_test_windows": int(len(splits["test"])),
        "num_classes": int(len(class_names)),
        "num_train_classes": int(train_class_count),
        "unseen_test_class_count": len(unseen_test),
        "unseen_test_classes": [class_names[item] for item in unseen_test],
        "primary_metric": "macro_f1",
        "primary_score": split_metrics.get("test", {}).get("macro_f1"),
        "splits": split_metrics,
        "history": history,
    }
    write_json(target_dir / "metrics.json", simple_metrics)
    write_csv(target_dir / "predictions.csv", predictions)
    if model_kind == "softmax":
        np.savez_compressed(target_dir / "model.npz", mean=mean, std=std, W=W, b=b, class_names=np.asarray(class_names, dtype=object))
    else:
        assert centroid is not None
        np.savez_compressed(target_dir / "model.npz", mean=mean, std=std, centroids=centroid[0], priors=centroid[1], class_names=np.asarray(class_names, dtype=object))

    neural_metrics = None
    if args.include_neural:
        neural_dir = output_dir / "neural_mlp" / target
        neural_dir.mkdir(parents=True, exist_ok=True)
        try:
            log(f"{target}: neural MLP")
            result = train_classifier(
                X,
                y,
                splits["train"],
                splits["test"],
                train_class_count,
                neural_config(args),
                use_class_weights=True,
            )
            pred = result["pred"]
            prob = result["prob"]
            metrics, per_class, cm = eval_split(y[splits["test"]], pred, class_names)
            pred_rows = []
            for out_idx, global_idx in enumerate(splits["test"]):
                pred_id = int(pred[out_idx])
                true_id = int(y[int(global_idx)])
                pred_rows.append({
                    **feature_rows[int(global_idx)],
                    "target": target,
                    "true_label": class_names[true_id],
                    "predicted_label": class_names[pred_id],
                    "confidence": float(prob[out_idx, pred_id]),
                    "correct": int(pred_id == true_id),
                })
            neural_metrics = {
                "status": "pass",
                "target": target,
                "model_family": "neural_mlp_dense_metadata",
                "device": result["device"],
                "num_train_windows": int(len(splits["train"])),
                "num_test_windows": int(len(splits["test"])),
                "num_classes": int(len(class_names)),
                "num_train_classes": int(train_class_count),
                "primary_metric": "macro_f1",
                "primary_score": metrics.get("macro_f1"),
                "test": metrics,
                "history": result["history"],
            }
            write_json(neural_dir / "metrics.json", neural_metrics)
            write_csv(neural_dir / "per_class_metrics.csv", per_class)
            write_csv(neural_dir / "predictions.csv", pred_rows)
            write_confusion(neural_dir / "confusion_matrix.csv", cm, class_names)
            save_torch_model(neural_dir / "model.pt", {
                "state_dict": result["state_dict"],
                "target": target,
                "class_names": class_names,
                "config": vars(args),
            })
        except Exception as exc:  # pragma: no cover - optional dependency/runtime guard.
            neural_metrics = {
                "status": "failed",
                "target": target,
                "model_family": "neural_mlp_dense_metadata",
                "primary_metric": "macro_f1",
                "primary_score": None,
                "error": str(exc),
            }
            write_json(neural_dir / "metrics.json", neural_metrics)
    return {"target": target, "simple": simple_metrics, "neural": neural_metrics}


def write_report(output_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Dense/Multiscale Hierarchical Baselines",
        "",
        f"Run id: `{summary['run_id']}`",
        "",
        f"Dataset: `{summary['dataset_manifest'].get('run_id', 'unknown')}`",
        f"Windows: `{summary['dataset_manifest'].get('num_samples')}`",
        "",
        "| target | simple status | simple test macro-F1 | neural status | neural test macro-F1 | train classes | test unseen |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for result in summary["targets"]:
        simple = result["simple"] or {}
        neural = result["neural"] or {}
        lines.append(
            "| `{target}` | {simple_status} | {simple_score} | {neural_status} | {neural_score} | {classes} | {unseen} |".format(
                target=result["target"],
                simple_status=simple.get("status"),
                simple_score="" if simple.get("primary_score") is None else f"{float(simple['primary_score']):.6f}",
                neural_status=neural.get("status", "not_run") if neural else "not_run",
                neural_score="" if not neural or neural.get("primary_score") is None else f"{float(neural['primary_score']):.6f}",
                classes=simple.get("num_train_classes", ""),
                unseen=simple.get("unseen_test_class_count", ""),
            )
        )
    output_dir.joinpath("RUN_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or (args.output_root / args.run_id)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise FileExistsError(f"Refusing to overwrite non-empty output dir without --force: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(args.dataset_jsonl)
    manifest = json.loads(args.dataset_manifest.read_text(encoding="utf-8")) if args.dataset_manifest.exists() else {}
    X, feature_rows = build_features(rows)
    splits = split_indices(rows)
    log(f"features={X.shape} splits={{{', '.join(f'{key}: {len(value)}' for key, value in splits.items())}}}")
    results = []
    for target in TARGETS:
        results.append(run_target(target, rows, feature_rows, X, splits, output_dir, args))
    summary = {
        "run_id": args.run_id,
        "dataset_jsonl": str(args.dataset_jsonl),
        "dataset_manifest": manifest,
        "feature_shape": list(X.shape),
        "split_counts": {split: int(len(idx)) for split, idx in splits.items()},
        "targets": results,
    }
    write_json(output_dir / "summary_report.json", summary)
    write_report(output_dir, summary)
    print(json.dumps(jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

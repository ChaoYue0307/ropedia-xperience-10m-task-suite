#!/usr/bin/env python3
"""Train/evaluate a Cosmos3-Nano future-window compatibility adapter.

This is the first runnable Cosmos branch for the shared 128-episode split. It
does not fine-tune Cosmos diffusion weights; instead it validates the
future-window data contract and produces world-model retrieval metrics plus
public-safe artifacts. Full Cosmos3 LoRA/diffusion fine-tuning can replace this
adapter once the remote environment has the Cosmos Diffusers stack installed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    workspace_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=workspace_default)
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    parser.add_argument("--run-id", default="xperience10m_cosmos3_nano_future_window_adapter")
    parser.add_argument("--eval-run-id")
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--eval-output-dir", type=Path)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--cosmos-model-dir", type=Path)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


class FeatureCache:
    def __init__(self) -> None:
        self._cache: dict[str, np.ndarray] = {}

    def get(self, path: str, index: int) -> np.ndarray:
        if path not in self._cache:
            with np.load(path, allow_pickle=True) as payload:
                self._cache[path] = np.asarray(payload["features"], dtype=np.float32)
        return np.asarray(self._cache[path][int(index)], dtype=np.float32)


def pair_features(rows: list[dict[str, Any]], cache: FeatureCache) -> tuple[np.ndarray, np.ndarray]:
    current = []
    future = []
    for row in rows:
        current.append(cache.get(str(row["sensor_feature_path"]), int(row["sensor_feature_index"])))
        future.append(cache.get(str(row["future_sensor_feature_path"]), int(row["future_sensor_feature_index"])))
    if not current:
        raise ValueError("No rows available for feature extraction.")
    return np.stack(current).astype(np.float32), np.stack(future).astype(np.float32)


def select_split(rows: list[dict[str, Any]], split: str, limit: int = 0) -> list[dict[str, Any]]:
    selected = [row for row in rows if row.get("split") == split]
    return selected[:limit] if limit > 0 else selected


def normalize_params(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def label(row: dict[str, Any], field: str) -> str:
    target = row.get("future_target") or {}
    return str(target.get(field, "unknown"))


def accuracy(rows: list[dict[str, Any]], predictions: list[dict[str, Any]], field: str) -> float:
    valid = [(row, pred) for row, pred in zip(rows, predictions) if label(row, field) != "unknown"]
    if not valid:
        return 0.0
    return sum(label(row, field) == str(pred.get(f"pred_{field}", "unknown")) for row, pred in valid) / len(valid)


def model_metadata(model_dir: Path | None) -> dict[str, Any]:
    if model_dir is None:
        return {"available": False, "reason": "cosmos model dir not provided"}
    model_dir = model_dir.expanduser()
    payload: dict[str, Any] = {"available": model_dir.exists(), "path": str(model_dir)}
    for filename in ("config.json", "model_index.json", "generation_config.json"):
        path = model_dir / filename
        payload[filename] = path.exists()
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if filename == "model_index.json":
            payload["pipeline_class"] = data.get("_class_name")
            payload["diffusers_version"] = data.get("_diffusers_version")
        if filename == "config.json":
            payload["architectures"] = data.get("architectures")
            cfg = ((data.get("model") or {}).get("config") or {})
            payload["lora_rank_default"] = cfg.get("lora_rank")
            payload["lora_enabled_default"] = cfg.get("lora_enabled")
            payload["resolution"] = cfg.get("resolution")
    return payload


def compute_retrieval(
    eval_rows: list[dict[str, Any]],
    y_hat: np.ndarray,
    candidate_y: np.ndarray,
    candidate_rows: list[dict[str, Any]],
    top_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    predictions: list[dict[str, Any]] = []
    rankings: list[dict[str, Any]] = []
    reciprocal_ranks = []
    recall_hits = 0
    reconstruction_errors = []
    same_episode_hits = 0
    dim = max(int(candidate_y.shape[1]), 1)

    for idx, (row, pred_vec) in enumerate(zip(eval_rows, y_hat)):
        distances = np.linalg.norm(candidate_y - pred_vec[None, :], axis=1)
        order = np.argsort(distances)
        true_rank_positions = np.where(order == idx)[0]
        rank = int(true_rank_positions[0]) + 1 if len(true_rank_positions) else len(order) + 1
        reciprocal_ranks.append(1.0 / rank)
        if rank <= top_k:
            recall_hits += 1
        top_indices = [int(item) for item in order[:top_k]]
        best = candidate_rows[top_indices[0]]
        same_episode_hits += int(best.get("episode_id") == row.get("episode_id"))
        true_error = float(np.linalg.norm(pred_vec - candidate_y[idx]) / math.sqrt(dim))
        reconstruction_errors.append(true_error)
        prediction = {
            "id": row.get("id"),
            "episode_id": row.get("episode_id"),
            "split": row.get("split"),
            "context_record_id": row.get("context_record_id"),
            "future_record_id": row.get("future_record_id"),
            "pred_future_record_id": best.get("future_record_id"),
            "rank": rank,
            "top_k_hit": rank <= top_k,
            "distance_to_true": float(distances[idx]),
            "distance_to_pred": float(distances[top_indices[0]]),
            "feature_reconstruction_error": true_error,
            "true_action": label(row, "action"),
            "pred_action": label(best, "action"),
            "true_contact": label(row, "contact"),
            "pred_contact": label(best, "contact"),
            "true_transition": label(row, "transition"),
            "pred_transition": label(best, "transition"),
        }
        predictions.append(prediction)
        for rank_pos, candidate_idx in enumerate(top_indices, start=1):
            candidate = candidate_rows[candidate_idx]
            rankings.append(
                {
                    "id": row.get("id"),
                    "rank": rank_pos,
                    "candidate_id": candidate.get("id"),
                    "candidate_future_record_id": candidate.get("future_record_id"),
                    "candidate_episode_id": candidate.get("episode_id"),
                    "distance": float(distances[candidate_idx]),
                    "is_true_future": candidate_idx == idx,
                }
            )

    n = max(len(eval_rows), 1)
    metrics = {
        "future_retrieval_mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        "future_retrieval_recall_at_5": float(recall_hits / n),
        "temporal_consistency": float(same_episode_hits / n),
        "feature_reconstruction_error": float(np.mean(reconstruction_errors)) if reconstruction_errors else 0.0,
    }
    return predictions, rankings, metrics


def temporal_rows(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        by_episode[str(row.get("episode_id"))].append(row)
    rows = []
    for episode_id, items in sorted(by_episode.items()):
        rows.append(
            {
                "episode_id": episode_id,
                "num_queries": len(items),
                "mrr": sum(1.0 / int(item["rank"]) for item in items) / max(len(items), 1),
                "recall_at_5": sum(bool(item["top_k_hit"]) for item in items) / max(len(items), 1),
                "mean_feature_reconstruction_error": sum(float(item["feature_reconstruction_error"]) for item in items) / max(len(items), 1),
            }
        )
    return rows


def qualitative_examples(eval_rows: list[dict[str, Any]], predictions: list[dict[str, Any]], limit: int = 20) -> dict[str, Any]:
    examples = []
    for row, pred in list(zip(eval_rows, predictions))[:limit]:
        examples.append(
            {
                "id": row.get("id"),
                "episode_id": row.get("episode_id"),
                "context_record_id": row.get("context_record_id"),
                "future_record_id": row.get("future_record_id"),
                "pred_future_record_id": pred.get("pred_future_record_id"),
                "rank": pred.get("rank"),
                "context_action": (row.get("conditioning") or {}).get("action"),
                "true_future_action": pred.get("true_action"),
                "pred_future_action": pred.get("pred_action"),
            }
        )
    return {
        "selection_policy": "first bounded examples from held-out eval rows; raw media omitted",
        "examples": examples,
    }


def main() -> int:
    args = parse_args()
    args.workspace = args.workspace.expanduser().resolve()
    root = args.workspace / "results" / "omni_finetune"
    eval_run_id = args.eval_run_id or f"{args.run_id}_eval"
    args.results_dir = args.results_dir or root / args.run_id
    args.eval_output_dir = args.eval_output_dir or root / eval_run_id
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.eval_output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.results_dir / "progress.jsonl"
    if progress_path.exists():
        progress_path.unlink()
    append_jsonl(progress_path, {"event": "setup_start", "run_id": args.run_id, "time": time.time()})

    rows = load_jsonl(args.dataset_jsonl)
    train_rows = select_split(rows, args.train_split, args.max_train_samples)
    val_rows = select_split(rows, args.val_split)
    eval_rows = select_split(rows, args.eval_split, args.max_eval_samples)
    if not train_rows or not eval_rows:
        raise ValueError(f"Need non-empty train/eval splits. train={len(train_rows)} eval={len(eval_rows)}")
    append_jsonl(
        progress_path,
        {
            "event": "setup_done",
            "run_id": args.run_id,
            "dataset_jsonl": str(args.dataset_jsonl),
            "num_train_samples": len(train_rows),
            "num_val_samples": len(val_rows),
            "num_eval_samples": len(eval_rows),
            "time": time.time(),
        },
    )

    cache = FeatureCache()
    x_train, y_train = pair_features(train_rows, cache)
    x_eval, y_eval = pair_features(eval_rows, cache)
    x_mean, x_std = normalize_params(x_train)
    y_mean, y_std = normalize_params(y_train)
    train_delta = standardize(y_train, y_mean, y_std) - standardize(x_train, x_mean, x_std)
    mean_delta = train_delta.mean(axis=0, keepdims=True).astype(np.float32)
    y_hat = (standardize(x_eval, x_mean, x_std) + mean_delta) * y_std + y_mean

    append_jsonl(
        progress_path,
        {
            "event": "adapter_built",
            "run_id": args.run_id,
            "feature_dim": int(x_train.shape[1]),
            "adapter": "standardized_current_plus_train_mean_future_delta",
            "time": time.time(),
        },
    )

    predictions, rankings, retrieval_metrics = compute_retrieval(eval_rows, y_hat, y_eval, eval_rows, args.top_k)
    metrics = {
        "eval_split": args.eval_split,
        "num_samples": len(eval_rows),
        "num_eval_episodes": len({row.get("episode_id") for row in eval_rows}),
        "held_out_episode_count": len({row.get("episode_id") for row in eval_rows}),
        **retrieval_metrics,
        "transition_accuracy": accuracy(eval_rows, predictions, "transition"),
        "contact_accuracy": accuracy(eval_rows, predictions, "contact"),
        "action_accuracy_from_retrieved_future": accuracy(eval_rows, predictions, "action"),
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
        "dataset_jsonl": str(args.dataset_jsonl),
        "cosmos_model": model_metadata(args.cosmos_model_dir),
    }

    write_json(args.eval_output_dir / "metrics.json", metrics)
    write_jsonl(args.eval_output_dir / "future_predictions.jsonl", predictions)
    write_csv(
        args.eval_output_dir / "retrieval_rankings.csv",
        rankings,
        ["id", "rank", "candidate_id", "candidate_future_record_id", "candidate_episode_id", "distance", "is_true_future"],
    )
    temporal = temporal_rows(predictions)
    write_csv(
        args.eval_output_dir / "temporal_consistency.csv",
        temporal,
        ["episode_id", "num_queries", "mrr", "recall_at_5", "mean_feature_reconstruction_error"],
    )
    write_json(args.eval_output_dir / "qualitative_examples.json", qualitative_examples(eval_rows, predictions))

    history = [
        {
            "epoch": 0,
            "train_loss": None,
            "val_loss": None,
            "note": "closed-form mean-delta adapter; no Cosmos diffusion weights fine-tuned in this compatibility run",
        }
    ]
    training_metadata = {
        "run_id": args.run_id,
        "backbone": "cosmos_world_model",
        "model_id": str(args.cosmos_model_dir) if args.cosmos_model_dir else None,
        "dataset_jsonl": str(args.dataset_jsonl),
        "checkpoint_dir": str(args.results_dir),
        "num_processes": 1,
        "num_train_samples": len(train_rows),
        "num_val_samples": len(val_rows),
        "history": history,
        "adapter": {
            "type": "standardized_current_plus_train_mean_future_delta",
            "feature_dim": int(x_train.shape[1]),
            "mean_delta_l2": float(np.linalg.norm(mean_delta)),
            "saved_weights": False,
        },
        "cosmos_model": model_metadata(args.cosmos_model_dir),
    }
    write_json(args.results_dir / "training_metadata.json", training_metadata)
    write_json(
        args.results_dir / "model_config.json",
        {
            "run_id": args.run_id,
            "backbone": "cosmos_world_model",
            "dataset_contract": "xperience10m_future_window_world_model_v0",
            "model_role": "Cosmos3-Nano compatibility branch before full diffusion LoRA fine-tuning",
            "adapter": training_metadata["adapter"],
            "normalization": {
                "x_mean_shape": list(x_mean.shape),
                "x_std_shape": list(x_std.shape),
                "y_mean_shape": list(y_mean.shape),
                "y_std_shape": list(y_std.shape),
            },
        },
    )
    write_json(
        args.results_dir / "checkpoint_manifest.json",
        {
            "run_id": args.run_id,
            "checkpoint_gate": "world_model_checkpoint_and_generation_config",
            "contains_base_model_weights": False,
            "contains_lora_weights": False,
            "contains_raw_media": False,
            "adapter_state": "closed_form_mean_delta_not_serialized",
            "eval_run_id": eval_run_id,
            "metrics": metrics,
        },
    )
    report = [
        "# Cosmos3-Nano Future-Window Compatibility Run",
        "",
        f"- Dataset: `{args.dataset_jsonl}`",
        f"- Train samples: `{len(train_rows)}`",
        f"- Validation samples: `{len(val_rows)}`",
        f"- Held-out test samples: `{len(eval_rows)}`",
        f"- Held-out episodes: `{metrics['held_out_episode_count']}`",
        f"- Future retrieval MRR: `{metrics['future_retrieval_mrr']:.6f}`",
        f"- Future retrieval recall@5: `{metrics['future_retrieval_recall_at_5']:.6f}`",
        f"- Temporal consistency: `{metrics['temporal_consistency']:.6f}`",
        f"- Feature reconstruction error: `{metrics['feature_reconstruction_error']:.6f}`",
        "",
        "This run validates the Cosmos3-Nano future-window contract on the same selected episode split.",
        "It does not fine-tune or publish Cosmos base weights; full Cosmos diffusion LoRA fine-tuning is the next step after the Cosmos Diffusers training stack is installed.",
    ]
    (args.eval_output_dir / "RUN_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    append_jsonl(progress_path, {"event": "eval_done", "run_id": args.run_id, "eval_run_id": eval_run_id, "metrics": str(args.eval_output_dir / "metrics.json"), "time": time.time()})
    append_jsonl(progress_path, {"event": "complete", "run_id": args.run_id, "time": time.time()})
    print(json.dumps({"status": "complete", "run_id": args.run_id, "eval_run_id": eval_run_id, "metrics": metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

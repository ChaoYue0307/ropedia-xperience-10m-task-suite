#!/usr/bin/env python3
"""Fill the 128-episode metadata task-19 cells with a documented sync proxy.

The public 128-episode metadata package does not contain paired per-camera
embeddings, so task 19 cannot be scored as true camera-view synchronization.
This runner uses the existing metadata feature matrix as the query side and the
staged same-window depth+audio feature block as the target side. The result is
therefore a compact synchronization proxy, matching the raw128 proxy policy.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "omni"))

from neural_task_models import NeuralConfig, save_torch_model, train_regressor
from run_128_sensor_block_completion_tasks import (
    FEATURE_BLOCKS,
    load_sensor_matrix,
    resolve_feature_path,
    split_indices,
)
from run_128_task_baselines import (
    TASKS,
    portable_path,
    retrieval_metrics_from_scores,
    ridge_regression_predict,
    task_display_name,
    write_json,
)


DEFAULT_DATASET = (
    ROOT
    / "results/omni_finetune/xperience10m_qwen3_omni_128ep_multiscale_cap96_v5_full8gpu_lora_dataset"
    / "dataset_a100_eval.jsonl"
)
DEFAULT_OUTPUT = ROOT / "results/omni_finetune/a100_128_metadata_task_baselines_20260616_v2"
TASK_ID = "camera_view_sync_retrieval"
PROXY_REASON = (
    "paired camera-view embeddings are absent from the 128 JSONL/feature export; "
    "metadata features retrieve the synchronized same-window depth/audio block as "
    "a documented compact synchronization proxy"
)
PROXY_TARGET = "same_window_depth_confidence_plus_audio_fisheye_cam0_aac"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-jsonl", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--feature-matrix", type=Path, default=None)
    parser.add_argument("--l2", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--include-neural", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--neural-epochs", type=int, default=35)
    parser.add_argument("--neural-hidden-dim", type=int, default=128)
    parser.add_argument("--neural-batch-size", type=int, default=256)
    parser.add_argument("--neural-learning-rate", type=float, default=1e-3)
    parser.add_argument("--neural-weight-decay", type=float, default=1e-4)
    parser.add_argument("--neural-dropout", type=float, default=0.10)
    parser.add_argument("--neural-device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[metadata-camera-sync-proxy] {message}", file=sys.stderr, flush=True)


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
        "split": row.get("split"),
        "center_window": window,
        "start_frame": int(window.get("start_frame", 0) or 0),
        "end_frame": int(window.get("end_frame", 0) or 0),
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


def sync_target_matrix(sensor_matrix: np.ndarray) -> np.ndarray:
    depth_start, depth_end = FEATURE_BLOCKS["depth_confidence"]
    audio_start, audio_end = FEATURE_BLOCKS["audio_fisheye_cam0_aac"]
    return np.concatenate(
        [sensor_matrix[:, depth_start:depth_end], sensor_matrix[:, audio_start:audio_end]],
        axis=1,
    ).astype(np.float32)


def normalize_rows(values: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(denom, 1e-6)


def feature_rows_from_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": row.get("id"),
            "episode_id": row.get("episode_id"),
            "split": row.get("split"),
            "start_frame": row.get("start_frame"),
            "end_frame": row.get("end_frame"),
        }
        for row in rows
    ]


def load_metadata_matrix(path: Path, rows: list[dict[str, Any]]) -> np.ndarray:
    with np.load(path, allow_pickle=True) as data:
        X = data["X"].astype(np.float32)
        row_ids = [str(value) for value in data["row_id"].tolist()] if "row_id" in data.files else []
    if X.shape[0] != len(rows):
        raise RuntimeError(f"metadata row count mismatch: X={X.shape[0]} rows={len(rows)}")
    if row_ids:
        for idx, (matrix_id, row) in enumerate(zip(row_ids, rows)):
            if matrix_id != str(row.get("id")):
                raise RuntimeError(f"metadata row id mismatch at {idx}: {matrix_id} != {row.get('id')}")
    return X


def simple_proxy(
    X: np.ndarray,
    Y: np.ndarray,
    rows: list[dict[str, Any]],
    splits: dict[str, np.ndarray],
    out_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    test_idx = splits["test"]
    pred, model = ridge_regression_predict(X[splits["train"]], Y[splits["train"]], X[test_idx], args.l2)
    scores = normalize_rows(pred) @ normalize_rows(Y[test_idx]).T
    retrieval_metrics, rank_rows = retrieval_metrics_from_scores(scores.astype(np.float32), rows, test_idx)
    payload = {
        "status": "pass",
        "task": TASK_ID,
        "task_display_name": task_display_name(TASK_ID),
        "model_family": "simple_ridge_metadata_sync_proxy",
        "source": "128_episode_qwen_jsonl_metadata_plus_staged_sensor_proxy_target",
        "scope": "multi_episode_128_aligned_metadata_baseline",
        "input_features": "frame/context metadata plus hashed prompt/options/main_task text; answer_json fields are excluded from inputs",
        "target_features": "same-window staged depth_confidence plus audio_fisheye_cam0_aac feature block",
        "split_policy": "train metadata-to-sync-target projection on train split, rank held-out same-window targets",
        "num_train_windows": int(len(splits["train"])),
        "num_val_windows": int(len(splits["val"])),
        "num_test_windows": int(len(test_idx)),
        **retrieval_metrics,
        "primary_metric": "mrr",
        "metric_direction": "higher",
        "primary_score": retrieval_metrics["mrr"],
        "proxy_completion": True,
        "proxy_reason": PROXY_REASON,
        "proxy_target": PROXY_TARGET,
    }
    write_json(out_dir / "metrics.json", payload)
    write_csv(out_dir / "ranks.csv", rank_rows)
    np.savez_compressed(out_dir / "model.npz", **model)
    return payload


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


def neural_proxy(
    X: np.ndarray,
    Y: np.ndarray,
    rows: list[dict[str, Any]],
    splits: dict[str, np.ndarray],
    out_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    test_idx = splits["test"]
    result = train_regressor(X.astype(np.float32), Y.astype(np.float32), splits["train"], test_idx, neural_config(args))
    scores = normalize_rows(result["pred"]) @ normalize_rows(Y[test_idx]).T
    retrieval_metrics, rank_rows = retrieval_metrics_from_scores(scores.astype(np.float32), rows, test_idx)
    payload = {
        "status": "pass",
        "task": TASK_ID,
        "task_display_name": task_display_name(TASK_ID),
        "model_family": "neural_mlp_metadata_sync_proxy",
        "source": "128_episode_qwen_jsonl_metadata_plus_staged_sensor_proxy_target",
        "scope": "multi_episode_128_aligned_metadata_baseline",
        "input_features": "frame/context metadata plus hashed prompt/options/main_task text; answer_json fields are excluded from inputs",
        "target_features": "same-window staged depth_confidence plus audio_fisheye_cam0_aac feature block",
        "split_policy": "train neural metadata-to-sync-target projection on train split, rank held-out same-window targets",
        "num_train_windows": int(len(splits["train"])),
        "num_val_windows": int(len(splits["val"])),
        "num_test_windows": int(len(test_idx)),
        "history": result["history"],
        "device": result["device"],
        **retrieval_metrics,
        "primary_metric": "mrr",
        "metric_direction": "higher",
        "primary_score": retrieval_metrics["mrr"],
        "proxy_completion": True,
        "proxy_reason": PROXY_REASON,
        "proxy_target": PROXY_TARGET,
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


def merge_summary(out_dir: Path, rows: list[dict[str, Any]], simple: dict[str, Any], neural: dict[str, Any] | None) -> None:
    summary_path = out_dir / "summary_report.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        split_counts = {key: len(value) for key, value in split_indices(rows).items()}
        summary = {
            "status": "pass",
            "run_id": "xperience10m_128_episode_aligned_task_baselines",
            "num_rows": len(rows),
            "split_counts": split_counts,
            "tasks": [],
        }

    existing = {item["task"]: item for item in summary.get("tasks", []) if "task" in item}
    existing[TASK_ID] = {"task": TASK_ID, "simple": simple, "neural": neural}
    summary["tasks"] = [existing[task_id] for task_id in TASKS if task_id in existing]
    summary["source_policy"] = (
        "derived JSONL metadata plus staged 128-episode sensor feature blocks for task cells whose targets are present; "
        "task 19 uses a documented compact same-window depth/audio synchronization proxy because paired camera-view embeddings are absent"
    )
    feature_contract = dict(summary.get("feature_contract") or {})
    sensor_block_completion = dict(feature_contract.get("sensor_block_completion") or {})
    completed = list(sensor_block_completion.get("completed_task_ids") or [])
    if TASK_ID not in completed:
        completed.append(TASK_ID)
    not_completed = [task for task in sensor_block_completion.get("not_completed_task_ids", []) if task != TASK_ID]
    sensor_block_completion.update(
        {
            "completed_task_ids": completed,
            "not_completed_task_ids": not_completed,
            "task19_proxy": {
                "proxy_completion": True,
                "proxy_target": PROXY_TARGET,
                "proxy_reason": PROXY_REASON,
            },
        }
    )
    feature_contract["sensor_block_completion"] = sensor_block_completion
    summary["feature_contract"] = feature_contract
    write_json(summary_path, summary)

    rows_out = []
    for item in summary["tasks"]:
        simple_item = item.get("simple") or {}
        neural_item = item.get("neural") or {}
        rows_out.append(
            {
                "task": item["task"],
                "task_display_name": task_display_name(item["task"]),
                "simple_status": simple_item.get("status"),
                "simple_primary_metric": simple_item.get("primary_metric"),
                "simple_primary_score": simple_item.get("primary_score"),
                "neural_status": neural_item.get("status") if neural_item else "not_run",
                "neural_primary_metric": neural_item.get("primary_metric") if neural_item else "",
                "neural_primary_score": neural_item.get("primary_score") if neural_item else "",
            }
        )
    write_csv(out_dir / "task_metrics.csv", rows_out)

    report_lines = [
        "# 128-Episode Metadata Camera-Sync Proxy",
        "",
        f"- Task: `{TASK_ID}` / {task_display_name(TASK_ID)}",
        f"- Proxy target: `{PROXY_TARGET}`",
        f"- Reason: {PROXY_REASON}",
        f"- Simple MRR: {simple.get('primary_score')}",
    ]
    if neural:
        report_lines.append(f"- Neural MRR: {neural.get('primary_score')}")
    report_lines.append("")
    (out_dir / "METADATA_CAMERA_SYNC_PROXY_REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    feature_matrix_path = args.feature_matrix or (args.output_dir / "metadata_feature_matrix.npz")
    if not args.dataset_jsonl.exists():
        raise FileNotFoundError(f"missing dataset JSONL: {args.dataset_jsonl}")
    if not feature_matrix_path.exists():
        raise FileNotFoundError(f"missing metadata feature matrix: {feature_matrix_path}")

    log(f"loading rows from {portable_path(args.dataset_jsonl)}")
    rows = load_rows(args.dataset_jsonl)
    splits = split_indices(rows)
    log(f"loading metadata matrix from {portable_path(feature_matrix_path)}")
    X = load_metadata_matrix(feature_matrix_path, rows)
    log("loading staged sensor matrix")
    sensor_matrix = load_sensor_matrix(rows, args.dataset_jsonl)
    Y = sync_target_matrix(sensor_matrix)
    log(f"query matrix {X.shape[0]}x{X.shape[1]}, target matrix {Y.shape[0]}x{Y.shape[1]}")

    simple = simple_proxy(X, Y, rows, splits, args.output_dir / TASK_ID, args)
    neural = None
    if args.include_neural:
        neural = neural_proxy(X, Y, rows, splits, args.output_dir / "neural_mlp" / TASK_ID, args)
    merge_summary(args.output_dir, rows, simple, neural)
    log(f"complete: simple MRR={simple.get('primary_score')} neural MRR={neural.get('primary_score') if neural else 'not_run'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

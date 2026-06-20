#!/usr/bin/env python3
"""Merge Qwen3 retrieval-task probe shards into one result package."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from eval_qwen3_omni_retrieval_task_probes import TASK_SPECS, score_task, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, nargs="+", required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def fake_args(run_id: str, first_metrics: dict[str, Any]) -> argparse.Namespace:
    return argparse.Namespace(
        run_id=run_id,
        model_id=first_metrics.get("model_id"),
        adapter_dir=Path(first_metrics.get("adapter_dir", "")),
        dataset_jsonl=Path(first_metrics.get("dataset_jsonl", "")),
        eval_split=first_metrics.get("eval_split", "test"),
        candidate_count=int(first_metrics.get("candidate_count", 4) or 4),
        future_frames=int(first_metrics.get("future_frames", 100) or 100),
        sample_offset=0,
        sample_stride=1,
    )


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    task_metrics: dict[str, dict[str, Any]] = {}
    first_metrics: dict[str, Any] | None = None
    duplicate_predictions: list[dict[str, Any]] = []

    for task_id, spec in TASK_SPECS.items():
        rows_by_id: dict[str, dict[str, Any]] = {}
        row_sources: dict[str, str] = {}
        for shard_dir in args.shard_dir:
            for row in read_jsonl(shard_dir / task_id / "predictions.jsonl"):
                key = str(row.get("prediction_id") or f"{task_id}::{row.get('id')}")
                if key in rows_by_id:
                    duplicate_predictions.append(
                        {
                            "task_id": task_id,
                            "prediction_id": key,
                            "kept_shard": row_sources.get(key),
                            "duplicate_shard": str(shard_dir),
                            "conflict": rows_by_id[key] != row,
                        }
                    )
                    continue
                rows_by_id[key] = row
                row_sources[key] = str(shard_dir)
            shard_metrics = read_json(shard_dir / task_id / "metrics.json")
            if shard_metrics and first_metrics is None:
                first_metrics = shard_metrics
        if not rows_by_id:
            continue
        ordered_rows = sorted(
            rows_by_id.values(),
            key=lambda row: (str(row.get("episode_id")), int(row.get("start_frame", 0)), str(row.get("id"))),
        )
        task_dir = args.output_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(task_dir / "predictions.jsonl", ordered_rows)
        metrics = score_task(task_id, spec, ordered_rows, args.output_dir, fake_args(args.run_id, first_metrics or {}))
        task_metrics[task_id] = metrics

    for shard_dir in args.shard_dir:
        if (shard_dir / "progress.jsonl").exists():
            shutil.copy2(shard_dir / "progress.jsonl", args.output_dir / f"{shard_dir.name}.progress.jsonl")

    summary = {
        "title": "Qwen3-Omni v6 Retrieval Task Probes",
        "status": "pass",
        "run_id": args.run_id,
        "shard_dirs": [str(path) for path in args.shard_dir],
        "duplicate_prediction_count": len(duplicate_predictions),
        "duplicate_prediction_conflict_count": sum(1 for row in duplicate_predictions if row["conflict"]),
        "duplicate_predictions": duplicate_predictions[:50],
        "tasks": {
            task_id: {
                "task_number": metrics["task_number"],
                "task_label": metrics["task_label"],
                "metric_key": metrics["metric_key"],
                "primary_score": metrics["primary_score"],
                "num_samples": metrics["num_samples"],
                "metrics_json": str(args.output_dir / task_id / "metrics.json"),
            }
            for task_id, metrics in task_metrics.items()
        },
    }
    write_json(args.output_dir / "summary.json", summary)
    report = [
        "# Qwen3-Omni v6 Retrieval Task Probes",
        "",
        f"- Run ID: `{args.run_id}`",
        f"- Shards: `{len(args.shard_dir)}`",
        "",
        "| Task | Metric | Score | Samples |",
        "| --- | --- | ---: | ---: |",
    ]
    for metrics in task_metrics.values():
        report.append(
            f"| {metrics['task_label']} | {metrics['metric_key']} | {metrics['primary_score']:.6f} | {metrics['num_samples']} |"
        )
    (args.output_dir / "RUN_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

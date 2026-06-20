#!/usr/bin/env python3
"""Merge sharded Cosmos3-Super task-15 interaction-text predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval_cosmos3_super_interaction_text_task import TASK_ID, score_rows, write_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--caption-manifest", type=Path, required=True)
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    parser.add_argument("--caption-jsonl", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--candidate-count", type=int, default=4)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    args = parse_args()
    merged: dict[str, dict[str, Any]] = {}
    for shard_dir in args.shard_dir:
        path = shard_dir / TASK_ID / "predictions.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"missing shard predictions: {path}")
        for row in read_jsonl(path):
            pred_id = row.get("prediction_id")
            if not pred_id:
                raise ValueError(f"prediction row missing prediction_id in {path}")
            if pred_id in merged:
                raise ValueError(f"duplicate prediction_id across shards: {pred_id}")
            merged[str(pred_id)] = row

    rows = sorted(merged.values(), key=lambda row: (str(row.get("episode_id")), int(row.get("start_frame", 0)), str(row.get("id"))))
    manifest = read_json(args.caption_manifest)
    score_args = argparse.Namespace(
        run_id=args.run_id,
        output_dir=args.output_dir,
        model=args.model,
        base_url=args.base_url,
        dataset_jsonl=args.dataset_jsonl,
        caption_jsonl=args.caption_jsonl,
        caption_manifest=args.caption_manifest,
        eval_split=args.eval_split,
        candidate_count=args.candidate_count,
        sample_offset=0,
        sample_stride=len(args.shard_dir),
    )
    metrics, _per_class, _confusion = score_rows(rows, score_args, manifest)
    write_outputs(rows, score_args, manifest)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "title": "Cosmos3-Super Reasoner Interaction Text Task-15 Probe",
                "status": "pass",
                "run_id": args.run_id,
                "shard_dirs": [str(path) for path in args.shard_dir],
                "task_metrics": {TASK_ID: metrics},
                "output_dir": str(args.output_dir),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

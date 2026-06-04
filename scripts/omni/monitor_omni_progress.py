#!/usr/bin/env python3
"""Print a compact progress snapshot for an omni fine-tuning run."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    workspace_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Monitor an omni fine-tuning run.")
    parser.add_argument("--workspace", type=Path, default=workspace_default)
    parser.add_argument("--run-id", default="xperience10m_qwen3_omni_32ep")
    parser.add_argument("--last", type=int, default=5)
    return parser.parse_args()


def read_jsonl(path: Path, limit: int) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows[-limit:]


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def nvidia_smi() -> str:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        return f"nvidia-smi unavailable: {exc}"


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def shard_export_summary(dataset_dir: Path) -> dict:
    shard_root = dataset_dir / "shards"
    if not shard_root.exists():
        return {}
    rows = []
    for shard_dir in sorted(shard_root.glob("shard_*")):
        media_count = sum(1 for _ in (shard_dir / "media").rglob("*") if _.is_file()) if (shard_dir / "media").exists() else 0
        sensor_count = sum(1 for _ in (shard_dir / "sensor_features").rglob("*") if _.is_file()) if (shard_dir / "sensor_features").exists() else 0
        manifest = shard_dir / "dataset_manifest.json"
        rows.append({
            "shard": shard_dir.name,
            "media_files": media_count,
            "sensor_files": sensor_count,
            "done": manifest.exists(),
            "samples": read_json(manifest).get("num_samples") if manifest.exists() else None,
        })
    return {
        "num_shards": len(rows),
        "done_shards": sum(1 for row in rows if row["done"]),
        "media_files": sum(row["media_files"] for row in rows),
        "sensor_files": sum(row["sensor_files"] for row in rows),
        "shards": rows,
    }


def dataset_summary(dataset_dir: Path) -> dict:
    manifest = read_json(dataset_dir / "dataset_manifest.json")
    if manifest:
        return {
            "path": str(dataset_dir / "dataset_manifest.json"),
            "num_samples": manifest.get("num_samples"),
            "num_episodes": manifest.get("num_episodes"),
            "split_counts": manifest.get("split_counts"),
            "skipped_episodes": len(manifest.get("skipped_episodes", [])),
        }
    dataset_jsonl = dataset_dir / "dataset.jsonl"
    if not dataset_jsonl.exists():
        return {}
    counts = Counter()
    episodes = set()
    with dataset_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            counts[row.get("split", "unspecified")] += 1
            episodes.add(row.get("episode_id"))
    return {
        "path": str(dataset_jsonl),
        "num_samples": sum(counts.values()),
        "num_episodes": len(episodes),
        "split_counts": dict(counts),
    }


def main() -> int:
    args = parse_args()
    root = args.workspace / "results" / "omni_finetune"
    run_dir = root / args.run_id
    dataset_dir = root / f"{args.run_id}_dataset"
    status_path = first_existing([
        run_dir / "status.jsonl",
        run_dir / "pipeline_status.jsonl",
        root / f"{args.run_id}_watch" / "status.jsonl",
    ])
    train_progress = root / f"{args.run_id}_lora" / "progress.jsonl"
    metrics = root / f"{args.run_id}_eval" / "metrics.json"
    log_path = first_existing([
        run_dir / "run.log",
        run_dir / "logs" / "pipeline.log",
        root / f"{args.run_id}.detached.log",
    ])

    print(f"Run: {args.run_id}")
    print(f"Status file: {status_path or 'not found'}")
    print(f"Pipeline log: {log_path or 'not found'}")
    print("\nGPU status: index, used MiB, total MiB, util %")
    print(nvidia_smi())

    print("\nRecent pipeline phases:")
    for row in read_jsonl(status_path, args.last) if status_path else []:
        print(json.dumps(row, ensure_ascii=False))

    print("\nExport summary:")
    export_summary = shard_export_summary(dataset_dir)
    if export_summary:
        compact = {key: export_summary[key] for key in ("num_shards", "done_shards", "media_files", "sensor_files")}
        print(json.dumps(compact, indent=2))
    else:
        print("No shard export directory found yet.")

    ds_summary = dataset_summary(dataset_dir)
    if ds_summary:
        print("\nDataset summary:")
        print(json.dumps(ds_summary, indent=2))

    print("\nRecent training progress:")
    for row in read_jsonl(train_progress, args.last):
        print(json.dumps(row, ensure_ascii=False))

    if metrics.exists():
        print("\nEval metrics:")
        payload = json.loads(metrics.read_text(encoding="utf-8"))
        keys = ["accuracy", "action_macro_f1", "json_validity_rate", "subtask_accuracy", "object_micro_f1"]
        print(json.dumps({key: payload.get(key) for key in keys}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

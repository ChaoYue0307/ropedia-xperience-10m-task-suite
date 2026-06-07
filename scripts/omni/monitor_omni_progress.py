#!/usr/bin/env python3
"""Print a compact progress snapshot for an omni fine-tuning run."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import Counter
from pathlib import Path


DEFAULT_RUN_ID = "xperience10m_qwen3_omni_32ep"


def parse_args() -> argparse.Namespace:
    workspace_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Monitor an omni fine-tuning run.")
    parser.add_argument("--workspace", type=Path, default=workspace_default)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--dataset-run-id", help="Run id that owns the episode manifest and exported dataset.")
    parser.add_argument("--train-run-id", help="Run id that owns training progress and checkpoint artifacts.")
    parser.add_argument("--eval-run-id", help="Run id that owns held-out evaluation metrics.")
    parser.add_argument("--watch-status-jsonl", type=Path, help="Explicit watcher status JSONL path.")
    parser.add_argument("--last", type=int, default=5)
    parser.add_argument("--stale-seconds", type=float, default=300.0, help="Warn when an eval log has not changed for this many seconds.")
    return parser.parse_args()


def read_jsonl(path: Path, limit: int = 0) -> list[dict]:
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
    return rows[-limit:] if limit > 0 else rows


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


def resolve_train_run_id(args: argparse.Namespace, root: Path, dataset_run_id: str) -> str:
    if args.train_run_id:
        return args.train_run_id

    if args.dataset_run_id:
        candidates = [
            args.run_id,
            f"{args.run_id}_lora",
            f"{dataset_run_id}_lora",
        ]
    else:
        candidates = [
            f"{args.run_id}_lora",
            args.run_id,
        ]

    checkpoint_root = root.parents[1] / "checkpoints"
    for candidate in candidates:
        if (root / candidate / "progress.jsonl").exists() or (checkpoint_root / candidate / "adapter_lora").exists():
            return candidate

    return args.run_id if args.dataset_run_id else f"{args.run_id}_lora"


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


def dataset_split_counts(dataset_dir: Path) -> dict[str, int]:
    manifest = read_json(dataset_dir / "dataset_manifest.json")
    split_counts = manifest.get("split_counts") if manifest else None
    if isinstance(split_counts, dict):
        return {str(key): int(value) for key, value in split_counts.items()}
    dataset_jsonl = dataset_dir / "dataset.jsonl"
    if not dataset_jsonl.exists():
        return {}
    counts = Counter()
    with dataset_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            counts[str(row.get("split", "unspecified"))] += 1
    return dict(counts)


def format_duration(seconds: float | None) -> str | None:
    if seconds is None:
        return None
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def training_progress_summary(rows: list[dict]) -> dict:
    if not rows:
        return {}
    setup = next((row for row in rows if row.get("event") == "setup_done"), {})
    train_loop = next((row for row in rows if row.get("event") == "train_loop_start"), {})
    train_steps = [row for row in rows if row.get("event") == "train_step"]
    latest = rows[-1]
    rank0_steps_per_epoch = int(
        setup.get("rank0_samples_per_epoch")
        or train_loop.get("rank_samples_per_epoch")
        or 0
    )
    epochs = int(train_loop.get("epochs") or 0)
    total_rank0_steps = rank0_steps_per_epoch * epochs if rank0_steps_per_epoch and epochs else rank0_steps_per_epoch
    summary = {
        "latest_event": latest.get("event"),
        "rows": len(rows),
        "num_processes": setup.get("num_processes"),
        "num_train_samples": setup.get("num_train_samples"),
        "rank0_samples_per_epoch": setup.get("rank0_samples_per_epoch"),
        "epochs": epochs or None,
    }
    if train_steps:
        last_step = train_steps[-1]
        current = int(last_step.get("global_step") or 0)
        rank0_seen = int(last_step.get("rank0_seen") or 0)
        epoch_percent = (
            round((rank0_seen / rank0_steps_per_epoch) * 100, 2)
            if rank0_steps_per_epoch
            else None
        )
        overall_percent = (
            round((current / total_rank0_steps) * 100, 2)
            if total_rank0_steps
            else None
        )
        summary.update({
            "current_epoch": last_step.get("epoch"),
            "global_step": current,
            "rank0_seen_this_epoch": rank0_seen,
            "rank0_steps_per_epoch": rank0_steps_per_epoch or None,
            "total_rank0_steps": total_rank0_steps or None,
            "percent_complete": overall_percent,
            "overall_percent_complete": overall_percent,
            "epoch_percent_complete": epoch_percent,
            "latest_rank0_loss": last_step.get("rank0_batch_loss"),
        })
        if len(train_steps) >= 2:
            first = train_steps[0]
            elapsed = float(last_step.get("timestamp", 0)) - float(first.get("timestamp", 0))
            step_delta = int(last_step.get("global_step", 0)) - int(first.get("global_step", 0))
            seconds_per_step = elapsed / step_delta if step_delta > 0 else None
            remaining_steps = max(0, total_rank0_steps - current) if total_rank0_steps else None
            remaining = remaining_steps * seconds_per_step if remaining_steps is not None and seconds_per_step else None
            summary["seconds_per_step"] = round(seconds_per_step, 3) if seconds_per_step else None
            summary["eta"] = format_duration(remaining)
    return summary


def eval_progress_summary(eval_dir: Path) -> dict:
    progress_path = eval_dir / "progress.jsonl"
    partial_path = eval_dir / "predictions.partial.jsonl"
    progress_rows = read_jsonl(progress_path)
    if not progress_rows and not partial_path.exists():
        return {}
    sample_events = [row for row in progress_rows if row.get("event") == "sample_done"]
    start = next((row for row in progress_rows if row.get("event") == "eval_start"), {})
    latest = progress_rows[-1] if progress_rows else {}
    completed = len(sample_events)
    if partial_path.exists():
        completed = max(completed, sum(1 for _ in partial_path.open("r", encoding="utf-8") if _.strip()))
    total = int(start.get("num_eval_samples") or latest.get("num_eval_samples") or 0)
    summary = {
        "latest_event": latest.get("event"),
        "progress_jsonl": str(progress_path),
        "partial_predictions": str(partial_path) if partial_path.exists() else None,
        "completed_samples": completed,
        "num_eval_samples": total or None,
        "percent_complete": round((completed / total) * 100, 2) if total else None,
    }
    if len(sample_events) >= 2 and total:
        first = sample_events[0]
        last = sample_events[-1]
        elapsed = float(last.get("timestamp", 0)) - float(first.get("timestamp", 0))
        sample_delta = int(last.get("completed_samples", 0)) - int(first.get("completed_samples", 0))
        seconds_per_sample = elapsed / sample_delta if sample_delta > 0 else None
        remaining = (total - completed) * seconds_per_sample if seconds_per_sample else None
        summary["seconds_per_sample"] = round(seconds_per_sample, 3) if seconds_per_sample else None
        summary["eta"] = format_duration(remaining)
    return summary


def legacy_eval_log_summary(run_dir: Path, eval_run_id: str, dataset_dir: Path, eval_split: str = "test", stale_seconds: float = 300.0) -> dict:
    log_path = run_dir / f"eval_{eval_run_id}.log"
    if not log_path.exists():
        return {}
    split_counts = dataset_split_counts(dataset_dir)
    total = split_counts.get(eval_split)
    completed = 0
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "Setting `pad_token_id`" in line or "Setting pad_token_id" in line:
                completed += 1
    stat = log_path.stat()
    modified_seconds_ago = time.time() - stat.st_mtime
    remaining = max(0, total - completed) if total is not None else None
    return {
        "source": "legacy_generation_log",
        "log": str(log_path),
        "health": "active" if modified_seconds_ago <= stale_seconds else "stale_log",
        "eval_split": eval_split,
        "completed_generations": completed,
        "num_eval_samples": total,
        "remaining_generations": remaining,
        "percent_complete": round((completed / total) * 100, 2) if total else None,
        "log_bytes": stat.st_size,
        "log_modified_seconds_ago": round(modified_seconds_ago, 1),
        "stale_seconds": stale_seconds,
    }


def main() -> int:
    args = parse_args()
    root = args.workspace / "results" / "omni_finetune"
    dataset_run_id = args.dataset_run_id or args.run_id
    train_run_id = resolve_train_run_id(args, root, dataset_run_id)
    eval_run_id = args.eval_run_id or f"{train_run_id}_eval"
    run_dir = root / dataset_run_id
    train_dir = root / train_run_id
    eval_dir = root / eval_run_id
    dataset_dir = root / f"{dataset_run_id}_dataset"
    status_path = first_existing([
        args.watch_status_jsonl if args.watch_status_jsonl else Path("__missing__"),
        run_dir / f"watch_{train_run_id}.jsonl",
        run_dir / "status.jsonl",
        run_dir / "pipeline_status.jsonl",
        root / f"{dataset_run_id}_watch" / "status.jsonl",
    ])
    train_progress = first_existing([
        train_dir / "progress.jsonl",
        root / f"{args.run_id}_lora" / "progress.jsonl",
        root / args.run_id / "progress.jsonl",
    ])
    metrics_candidates = [eval_dir / "metrics.json"]
    if not args.eval_run_id:
        metrics_candidates.extend([
            root / f"{train_run_id}_eval" / "metrics.json",
            root / f"{args.run_id}_eval" / "metrics.json",
        ])
    metrics = first_existing(metrics_candidates)
    log_path = first_existing([
        run_dir / "run.log",
        run_dir / "logs" / "pipeline.log",
        run_dir / f"train_{train_run_id}.log",
        root / f"{dataset_run_id}.detached.log",
    ])

    display_run_id = train_run_id if args.run_id == DEFAULT_RUN_ID and args.train_run_id else args.run_id
    print(f"Run: {display_run_id}")
    print(f"Dataset run: {dataset_run_id}")
    print(f"Train run: {train_run_id}")
    print(f"Eval run: {eval_run_id}")
    print(f"Status file: {status_path or 'not found'}")
    print(f"Training progress: {train_progress or 'not found'}")
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
    train_rows = read_jsonl(train_progress) if train_progress else []
    if train_rows:
        print(json.dumps(training_progress_summary(train_rows), indent=2))
    for row in train_rows[-args.last:]:
        print(json.dumps(row, ensure_ascii=False))

    if metrics and metrics.exists():
        print("\nEval metrics:")
        payload = json.loads(metrics.read_text(encoding="utf-8"))
        keys = ["accuracy", "action_macro_f1", "json_validity_rate", "subtask_accuracy", "object_micro_f1"]
        print(json.dumps({key: payload.get(key) for key in keys}, indent=2))
    else:
        eval_summary = eval_progress_summary(eval_dir)
        if not eval_summary:
            eval_summary = legacy_eval_log_summary(run_dir, eval_run_id, dataset_dir, stale_seconds=args.stale_seconds)
        if eval_summary:
            print("\nEval progress:")
            print(json.dumps(eval_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Wait for validated omni eval output, then create the verified public package."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_DATASET_RUN_ID = "xperience10m_qwen3_omni_128ep_fullsplit_fast8gpu"
DEFAULT_TRAIN_RUN_ID = "xperience10m_qwen3_omni_128ep_fullsplit_fast8gpu_lora_fsdp_full_train_noval_tail_logits_fullstatesave_v6"
DEFAULT_EVAL_RUN_ID = f"{DEFAULT_TRAIN_RUN_ID}_eval_test_full"


def parse_args() -> argparse.Namespace:
    workspace_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=workspace_default)
    parser.add_argument("--dataset-run-id", default=DEFAULT_DATASET_RUN_ID)
    parser.add_argument("--train-run-id", default=DEFAULT_TRAIN_RUN_ID)
    parser.add_argument("--eval-run-id", default=DEFAULT_EVAL_RUN_ID)
    parser.add_argument("--validation-json", type=Path)
    parser.add_argument("--watch-status-jsonl", type=Path)
    parser.add_argument("--package-output-dir", type=Path)
    parser.add_argument("--status-jsonl", type=Path)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--timeout-hours", type=float, default=12.0)
    parser.add_argument("--max-file-mb", type=float, default=50.0)
    parser.add_argument("--progress-event-seconds", type=float, default=300.0)
    parser.add_argument("--stale-seconds", type=float, default=300.0)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append({"event": "json_decode_error", "raw": line})
    return rows


def append_event(path: Path, event: str, **fields: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"event": event, "time": time.time(), **fields}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return sum(1 for line in handle if line.strip())


def dataset_split_counts(dataset_dir: Path) -> dict[str, int]:
    manifest = dataset_dir / "dataset_manifest.json"
    if manifest.exists():
        payload = read_json(manifest)
        split_counts = payload.get("split_counts")
        if isinstance(split_counts, dict):
            return {str(key): int(value) for key, value in split_counts.items()}
    dataset_jsonl = dataset_dir / "dataset.jsonl"
    counts: dict[str, int] = {}
    if not dataset_jsonl.exists():
        return counts
    with dataset_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            split = str(json.loads(line).get("split", "unspecified"))
            counts[split] = counts.get(split, 0) + 1
    return counts


def legacy_generation_count(log_path: Path) -> int:
    if not log_path.exists():
        return 0
    count = 0
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "Setting `pad_token_id`" in line or "Setting pad_token_id" in line:
                count += 1
    return count


def eval_progress(root: Path, dataset_run_id: str, eval_run_id: str, stale_seconds: float) -> dict[str, Any]:
    eval_dir = root / eval_run_id
    dataset_dir = root / f"{dataset_run_id}_dataset"
    progress = eval_dir / "progress.jsonl"
    partial = eval_dir / "predictions.partial.jsonl"
    final_predictions = eval_dir / "predictions.jsonl"
    log_path = root / dataset_run_id / f"eval_{eval_run_id}.log"
    split_counts = dataset_split_counts(dataset_dir)
    total = split_counts.get("test")

    source = None
    source_path: Path | None = None
    completed = 0
    if final_predictions.exists():
        source = "final_predictions"
        source_path = final_predictions
        completed = count_jsonl_rows(final_predictions)
    elif partial.exists():
        source = "partial_predictions"
        source_path = partial
        completed = count_jsonl_rows(partial)
    elif progress.exists():
        rows = read_jsonl(progress)
        sample_rows = [row for row in rows if row.get("event") == "sample_done"]
        source = "progress_jsonl"
        source_path = progress
        completed = len(sample_rows)
        starts = [row for row in rows if row.get("event") == "eval_start" and row.get("num_eval_samples")]
        if starts:
            total = int(starts[-1]["num_eval_samples"])
    elif log_path.exists():
        source = "legacy_generation_log"
        source_path = log_path
        completed = legacy_generation_count(log_path)

    if source is None:
        return {}

    modified_seconds_ago = time.time() - source_path.stat().st_mtime if source_path and source_path.exists() else None
    remaining = max(0, total - completed) if total is not None else None
    return {
        "source": source,
        "health": "active" if modified_seconds_ago is None or modified_seconds_ago <= stale_seconds else "stale",
        "completed": completed,
        "total": total,
        "remaining": remaining,
        "percent_complete": round((completed / total) * 100, 2) if total else None,
        "modified_seconds_ago": round(modified_seconds_ago, 1) if modified_seconds_ago is not None else None,
    }


def watcher_failure(watch_status: Path) -> dict[str, Any] | None:
    for row in reversed(read_jsonl(watch_status)):
        event = row.get("event")
        if event in {"eval_full_exit", "validation_eval_exit"} and int(row.get("returncode", 0)) != 0:
            return row
        if event == "watch_complete":
            return None
    return None


def package_cmd(args: argparse.Namespace, validation_json: Path, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        "scripts/omni/package_verified_omni_result.py",
        "--workspace",
        str(args.workspace),
        "--dataset-run-id",
        args.dataset_run_id,
        "--train-run-id",
        args.train_run_id,
        "--eval-run-id",
        args.eval_run_id,
        "--validation-json",
        str(validation_json),
        "--output-dir",
        str(output_dir),
        "--max-file-mb",
        str(args.max_file_mb),
    ]


def main() -> int:
    args = parse_args()
    args.workspace = args.workspace.expanduser().resolve()
    root = args.workspace / "results" / "omni_finetune"
    run_dir = root / args.dataset_run_id
    validation_json = args.validation_json or run_dir / f"validation_eval_{args.eval_run_id}.json"
    watch_status = args.watch_status_jsonl or run_dir / f"watch_{args.train_run_id}.jsonl"
    output_dir = args.package_output_dir or root / "verified_public" / args.eval_run_id
    status_path = args.status_jsonl or run_dir / f"package_watch_{args.eval_run_id}.jsonl"
    log_path = run_dir / f"package_verified_{args.eval_run_id}.log"

    append_event(
        status_path,
        "watch_start",
        dataset_run_id=args.dataset_run_id,
        train_run_id=args.train_run_id,
        eval_run_id=args.eval_run_id,
        validation_json=str(validation_json),
        output_dir=str(output_dir),
    )

    deadline = time.time() + args.timeout_hours * 3600
    last_state = None
    last_progress_event = 0.0
    while time.time() < deadline:
        failure = watcher_failure(watch_status)
        if failure:
            append_event(status_path, "blocked_by_eval_watcher", watcher_event=failure)
            return 1

        if validation_json.exists():
            validation = read_json(validation_json)
            state = validation.get("status")
            if state != last_state:
                append_event(status_path, "validation_observed", status=state, validation_json=str(validation_json))
                last_state = state
            if state == "pass":
                cmd = package_cmd(args, validation_json, output_dir)
                append_event(status_path, "package_start", command=cmd, log=str(log_path))
                with log_path.open("w", encoding="utf-8") as log:
                    proc = subprocess.run(cmd, cwd=args.workspace, stdout=log, stderr=subprocess.STDOUT, text=True)
                append_event(status_path, "package_exit", returncode=proc.returncode, log=str(log_path), output_dir=str(output_dir))
                return proc.returncode
            if state == "fail":
                append_event(status_path, "blocked_by_validation_failure", validation_json=str(validation_json), issues=validation.get("issues", []))
                return 1
        elif last_state != "waiting_for_validation":
            append_event(status_path, "waiting_for_validation", validation_json=str(validation_json), watch_status=str(watch_status))
            last_state = "waiting_for_validation"

        if time.time() - last_progress_event >= args.progress_event_seconds:
            progress = eval_progress(root, args.dataset_run_id, args.eval_run_id, args.stale_seconds)
            if progress:
                append_event(status_path, "eval_progress_observed", **progress)
                last_progress_event = time.time()

        time.sleep(args.poll_seconds)

    append_event(status_path, "timeout", validation_json=str(validation_json), timeout_hours=args.timeout_hours)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

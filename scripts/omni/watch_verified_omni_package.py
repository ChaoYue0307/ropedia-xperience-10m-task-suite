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

        time.sleep(args.poll_seconds)

    append_event(status_path, "timeout", validation_json=str(validation_json), timeout_hours=args.timeout_hours)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

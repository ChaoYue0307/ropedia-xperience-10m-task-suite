#!/usr/bin/env python3
"""Wait for launcher eval outputs, then validate, package, and audit them.

This is for launcher scripts that already perform training and full eval. It
does not rerun eval; it only waits for the eval artifact contract to appear and
then uses the existing validators/packagers.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from backbone_registry import load_registry


DEFAULT_REQUIRED_EVAL_FILES = [
    "metrics.json",
    "predictions.jsonl",
    "predictions.csv",
    "per_class_metrics.csv",
    "confusion_matrix.csv",
    "RUN_REPORT.md",
]


def parse_args() -> argparse.Namespace:
    workspace_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=workspace_default)
    parser.add_argument("--dataset-run-id", required=True)
    parser.add_argument("--train-run-id", required=True)
    parser.add_argument("--eval-run-id", required=True)
    parser.add_argument("--backbone", default="qwen3_omni_lora")
    parser.add_argument("--status-jsonl", type=Path)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--timeout-hours", type=float, default=48.0)
    parser.add_argument("--progress-event-seconds", type=float, default=300.0)
    parser.add_argument("--min-json-validity", type=float, default=1.0)
    parser.add_argument("--expected-train-episodes", type=int, default=96)
    parser.add_argument("--expected-val-episodes", type=int, default=16)
    parser.add_argument("--expected-test-episodes", type=int, default=16)
    parser.add_argument("--expected-dataset-train-episodes", type=int, default=89)
    parser.add_argument("--expected-dataset-val-episodes", type=int, default=16)
    parser.add_argument("--expected-dataset-test-episodes", type=int, default=14)
    parser.add_argument("--expected-num-processes", type=int, default=8)
    parser.add_argument("--max-file-mb", type=float, default=50.0)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
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
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def required_eval_files(workspace: Path, backbone_id: str) -> list[str]:
    registry = load_registry(workspace / "configs" / "omni_backbones")
    backbone = registry.get(backbone_id) or {}
    contract = backbone.get("artifact_contract") or {}
    files = contract.get("required_eval_files")
    return list(files) if isinstance(files, list) and files else list(DEFAULT_REQUIRED_EVAL_FILES)


def count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return sum(1 for line in handle if line.strip())


def eval_progress(eval_dir: Path) -> dict[str, Any]:
    progress_path = eval_dir / "progress.jsonl"
    partial_path = eval_dir / "predictions.partial.jsonl"
    final_path = eval_dir / "predictions.jsonl"
    metrics_path = eval_dir / "metrics.json"
    if metrics_path.exists():
        metrics = read_json(metrics_path)
        return {
            "source": "metrics",
            "completed": metrics.get("num_samples") or metrics.get("num_eval_samples"),
            "total": metrics.get("num_samples") or metrics.get("num_eval_samples"),
            "status": metrics.get("status", "metrics_present"),
        }
    if final_path.exists():
        return {"source": "predictions", "completed": count_jsonl_rows(final_path)}
    if partial_path.exists():
        return {"source": "partial_predictions", "completed": count_jsonl_rows(partial_path)}
    if progress_path.exists():
        rows = read_jsonl(progress_path)
        samples = [row for row in rows if row.get("event") == "sample_done"]
        starts = [row for row in rows if row.get("event") == "eval_start" and row.get("num_eval_samples")]
        total = starts[-1].get("num_eval_samples") if starts else None
        return {"source": "progress", "completed": len(samples), "total": total}
    return {}


def missing_files(eval_dir: Path, required_files: list[str]) -> list[str]:
    return [name for name in required_files if not (eval_dir / name).exists()]


def run_checked(cmd: list[str], *, cwd: Path, log_path: Path, status_path: Path, event: str) -> None:
    append_event(status_path, f"{event}_start", command=cmd, log=str(log_path))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, text=True)
    append_event(status_path, f"{event}_exit", returncode=proc.returncode, log=str(log_path))
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def validation_cmd(args: argparse.Namespace, output: Path) -> list[str]:
    return [
        sys.executable,
        "scripts/omni/validate_omni_finetune_run.py",
        "--workspace",
        str(args.workspace),
        "--run-id",
        args.dataset_run_id,
        "--dataset-run-id",
        args.dataset_run_id,
        "--train-run-id",
        args.train_run_id,
        "--eval-run-id",
        args.eval_run_id,
        "--backbone",
        args.backbone,
        "--require-stage",
        "eval",
        "--expected-train-episodes",
        str(args.expected_train_episodes),
        "--expected-val-episodes",
        str(args.expected_val_episodes),
        "--expected-test-episodes",
        str(args.expected_test_episodes),
        "--expected-dataset-train-episodes",
        str(args.expected_dataset_train_episodes),
        "--expected-dataset-val-episodes",
        str(args.expected_dataset_val_episodes),
        "--expected-dataset-test-episodes",
        str(args.expected_dataset_test_episodes),
        "--expected-num-processes",
        str(args.expected_num_processes),
        "--allow-zero-val-training",
        "--min-json-validity",
        str(args.min_json_validity),
        "--output",
        str(output),
    ]


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
        "--backbone",
        args.backbone,
        "--validation-json",
        str(validation_json),
        "--output-dir",
        str(output_dir),
        "--max-file-mb",
        str(args.max_file_mb),
    ]


def audit_cmd(output_dir: Path, output_json: Path, backbone: str) -> list[str]:
    return [
        sys.executable,
        "scripts/omni/audit_verified_omni_package.py",
        "--package-dir",
        str(output_dir),
        "--backbone",
        backbone,
        "--output",
        str(output_json),
    ]


def main() -> int:
    args = parse_args()
    args.workspace = args.workspace.expanduser().resolve()
    root = args.workspace / "results" / "omni_finetune"
    run_dir = root / args.dataset_run_id
    eval_dir = root / args.eval_run_id
    status_path = args.status_jsonl or run_dir / f"launcher_package_watch_{args.eval_run_id}.jsonl"
    validation_json = run_dir / f"validation_eval_{args.eval_run_id}.json"
    package_dir = root / "verified_public" / args.eval_run_id
    audit_json = run_dir / f"audit_verified_public_{args.eval_run_id}.json"
    required_files = required_eval_files(args.workspace, args.backbone)

    append_event(
        status_path,
        "watch_start",
        dataset_run_id=args.dataset_run_id,
        train_run_id=args.train_run_id,
        eval_run_id=args.eval_run_id,
        required_eval_files=required_files,
    )

    deadline = time.time() + args.timeout_hours * 3600
    last_missing: list[str] | None = None
    last_progress_event = 0.0
    while time.time() < deadline:
        missing = missing_files(eval_dir, required_files)
        if not missing:
            append_event(status_path, "eval_artifacts_ready", eval_dir=str(eval_dir))
            break
        if missing != last_missing:
            append_event(status_path, "waiting_for_eval_artifacts", eval_dir=str(eval_dir), missing=missing)
            last_missing = missing
        if time.time() - last_progress_event >= args.progress_event_seconds:
            progress = eval_progress(eval_dir)
            if progress:
                append_event(status_path, "eval_progress", **progress)
                last_progress_event = time.time()
        time.sleep(args.poll_seconds)
    else:
        append_event(status_path, "timeout", eval_dir=str(eval_dir), missing=last_missing or required_files)
        return 1

    run_checked(
        validation_cmd(args, validation_json),
        cwd=args.workspace,
        log_path=run_dir / f"validate_eval_{args.eval_run_id}.log",
        status_path=status_path,
        event="validation_eval",
    )
    run_checked(
        package_cmd(args, validation_json, package_dir),
        cwd=args.workspace,
        log_path=run_dir / f"package_verified_{args.eval_run_id}.log",
        status_path=status_path,
        event="package",
    )
    run_checked(
        audit_cmd(package_dir, audit_json, args.backbone),
        cwd=args.workspace,
        log_path=run_dir / f"audit_verified_public_{args.eval_run_id}.log",
        status_path=status_path,
        event="audit",
    )
    append_event(status_path, "watch_complete", package_dir=str(package_dir), audit_json=str(audit_json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

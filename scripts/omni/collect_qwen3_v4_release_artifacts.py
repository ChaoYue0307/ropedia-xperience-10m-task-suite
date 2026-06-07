#!/usr/bin/env python3
"""Collect Qwen3-Omni v4 release artifacts after remote validation passes.

This is a local handoff helper for the active remote run. It refuses to copy
anything until the remote public-safe package has a verified summary unless
``--allow-incomplete`` is passed for diagnostics. Adapter weights are copied
only when explicitly requested.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_REMOTE = os.environ.get("ROPEDIA_REMOTE", "")
DEFAULT_REMOTE_WORKSPACE = os.environ.get("ROPEDIA_REMOTE_WORKSPACE", "")
DEFAULT_DATASET_RUN_ID = "xperience10m_qwen3_omni_128ep_96train_16val_16test_valmon_20260605"
DEFAULT_TRAIN_RUN_ID = "xperience10m_qwen3_omni_128ep_structured_json_v4_4epoch_full8gpu_lora"
DEFAULT_EVAL_RUN_ID = f"{DEFAULT_TRAIN_RUN_ID}_eval_test_full"
DEFAULT_EVAL_SMOKE_RUN_ID = f"{DEFAULT_TRAIN_RUN_ID}_eval_smoke8"


def parse_args() -> argparse.Namespace:
    workspace_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=workspace_default)
    parser.add_argument("--remote", default=DEFAULT_REMOTE, required=not bool(DEFAULT_REMOTE))
    parser.add_argument("--remote-workspace", default=DEFAULT_REMOTE_WORKSPACE, required=not bool(DEFAULT_REMOTE_WORKSPACE))
    parser.add_argument("--dataset-run-id", default=DEFAULT_DATASET_RUN_ID)
    parser.add_argument("--train-run-id", default=DEFAULT_TRAIN_RUN_ID)
    parser.add_argument("--eval-run-id", default=DEFAULT_EVAL_RUN_ID)
    parser.add_argument("--eval-smoke-run-id", default=DEFAULT_EVAL_SMOKE_RUN_ID)
    parser.add_argument("--execute", action="store_true", help="run rsync; otherwise print the planned copy set")
    parser.add_argument("--include-adapter", action="store_true", help="also copy checkpoints/<train_run_id>/adapter_lora")
    parser.add_argument("--allow-incomplete", action="store_true", help="allow collection before verified package status is present")
    return parser.parse_args()


def run(argv: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, check=check, text=True, capture_output=True)


def remote_python_probe(args: argparse.Namespace) -> dict[str, Any]:
    script = f"""
import json
from pathlib import Path
root = Path({args.remote_workspace!r})
dataset = {args.dataset_run_id!r}
train = {args.train_run_id!r}
eval_run = {args.eval_run_id!r}
base = root / "results" / "omni_finetune"
package_dir = base / "verified_public" / eval_run
summary_path = package_dir / "verified_result_summary.json"
training_validation_path = base / dataset / f"validation_training_{{train}}.json"
eval_validation_path = base / dataset / f"validation_eval_{{eval_run}}.json"
package_watch = base / dataset / f"package_watch_{{eval_run}}.jsonl"
watch_status = base / dataset / f"watch_{{train}}.jsonl"
train_progress = base / train / "progress.jsonl"
def last_jsonl(path):
    if not path.exists():
        return None
    last = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            try:
                last = json.loads(line)
            except Exception:
                last = {{"event": "decode_error", "raw": line[:200]}}
    return last
summary = None
if summary_path.exists():
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
print(json.dumps({{
    "package_dir": str(package_dir),
    "package_exists": package_dir.exists(),
    "summary_path": str(summary_path),
    "summary_exists": summary_path.exists(),
    "summary_status": summary.get("status") if summary else None,
    "training_validation_path": str(training_validation_path),
    "training_validation_exists": training_validation_path.exists(),
    "eval_validation_path": str(eval_validation_path),
    "eval_validation_exists": eval_validation_path.exists(),
    "watch_status_last": last_jsonl(watch_status),
    "package_watch_last": last_jsonl(package_watch),
    "train_progress_last": last_jsonl(train_progress),
}}, indent=2))
"""
    proc = run(["ssh", args.remote, f"cd {args.remote_workspace} && .venv/bin/python - <<'PY'\n{script}\nPY"])
    return json.loads(proc.stdout)


def planned_paths(args: argparse.Namespace) -> list[tuple[str, Path]]:
    workspace = args.workspace.expanduser().resolve()
    remote_base = f"{args.remote}:{args.remote_workspace}"
    result_root = Path("results/omni_finetune")
    paths: list[tuple[str, Path]] = [
        (
            f"{remote_base}/results/omni_finetune/{args.dataset_run_id}/validation_training_{args.train_run_id}.json",
            workspace / result_root / args.dataset_run_id / f"validation_training_{args.train_run_id}.json",
        ),
        (
            f"{remote_base}/results/omni_finetune/{args.dataset_run_id}/validation_eval_{args.eval_run_id}.json",
            workspace / result_root / args.dataset_run_id / f"validation_eval_{args.eval_run_id}.json",
        ),
        (
            f"{remote_base}/results/omni_finetune/{args.dataset_run_id}/adapter_shape_check_{args.train_run_id}.json",
            workspace / result_root / args.dataset_run_id / f"adapter_shape_check_{args.train_run_id}.json",
        ),
        (
            f"{remote_base}/results/omni_finetune/{args.dataset_run_id}/validate_training_{args.train_run_id}.log",
            workspace / result_root / args.dataset_run_id / f"validate_training_{args.train_run_id}.log",
        ),
        (
            f"{remote_base}/results/omni_finetune/{args.dataset_run_id}/validate_eval_{args.eval_run_id}.log",
            workspace / result_root / args.dataset_run_id / f"validate_eval_{args.eval_run_id}.log",
        ),
        (
            f"{remote_base}/results/omni_finetune/{args.dataset_run_id}/eval_{args.eval_smoke_run_id}.log",
            workspace / result_root / args.dataset_run_id / f"eval_{args.eval_smoke_run_id}.log",
        ),
        (
            f"{remote_base}/results/omni_finetune/{args.dataset_run_id}/eval_{args.eval_run_id}.log",
            workspace / result_root / args.dataset_run_id / f"eval_{args.eval_run_id}.log",
        ),
        (
            f"{remote_base}/results/omni_finetune/{args.dataset_run_id}/watch_{args.train_run_id}.jsonl",
            workspace / result_root / args.dataset_run_id / f"watch_{args.train_run_id}.jsonl",
        ),
        (
            f"{remote_base}/results/omni_finetune/{args.dataset_run_id}/watch_{args.train_run_id}.log",
            workspace / result_root / args.dataset_run_id / f"watch_{args.train_run_id}.log",
        ),
        (
            f"{remote_base}/results/omni_finetune/{args.dataset_run_id}/package_watch_{args.eval_run_id}.jsonl",
            workspace / result_root / args.dataset_run_id / f"package_watch_{args.eval_run_id}.jsonl",
        ),
        (
            f"{remote_base}/results/omni_finetune/{args.dataset_run_id}/package_watch_{args.eval_run_id}.log",
            workspace / result_root / args.dataset_run_id / f"package_watch_{args.eval_run_id}.log",
        ),
        (
            f"{remote_base}/results/omni_finetune/{args.dataset_run_id}/audit_verified_public_{args.eval_run_id}.json",
            workspace / result_root / args.dataset_run_id / f"audit_verified_public_{args.eval_run_id}.json",
        ),
        (
            f"{remote_base}/results/omni_finetune/{args.train_run_id}/",
            workspace / result_root / args.train_run_id,
        ),
        (
            f"{remote_base}/results/omni_finetune/{args.eval_run_id}/",
            workspace / result_root / args.eval_run_id,
        ),
        (
            f"{remote_base}/results/omni_finetune/verified_public/{args.eval_run_id}/",
            workspace / result_root / "verified_public" / args.eval_run_id,
        ),
    ]
    if args.include_adapter:
        paths.append(
            (
                f"{remote_base}/checkpoints/{args.train_run_id}/adapter_lora/",
                workspace / "checkpoints" / args.train_run_id / "adapter_lora",
            )
        )
    return paths


def rsync_one(src: str, dst: Path, *, execute: bool) -> None:
    if src.endswith("/"):
        dst.mkdir(parents=True, exist_ok=True)
        dst_arg = str(dst) + "/"
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst_arg = str(dst)
    cmd = ["rsync", "-av", src, dst_arg]
    if not execute:
        print("DRY-RUN:", " ".join(cmd))
        return
    subprocess.run(cmd, check=True)


def audit_local_package(args: argparse.Namespace) -> int:
    package_dir = args.workspace / "results" / "omni_finetune" / "verified_public" / args.eval_run_id
    if not package_dir.exists():
        print(f"Local package not found after sync: {package_dir}", file=sys.stderr)
        return 1
    cmd = [
        sys.executable,
        "scripts/omni/audit_verified_omni_package.py",
        "--workspace",
        str(args.workspace),
        "--package-dir",
        str(package_dir),
        "--backbone",
        "qwen3_omni_lora",
    ]
    return subprocess.run(cmd, cwd=args.workspace).returncode


def main() -> int:
    args = parse_args()
    args.workspace = args.workspace.expanduser().resolve()
    probe = remote_python_probe(args)
    print(json.dumps({"remote_probe": probe}, indent=2))
    if probe.get("summary_status") != "verified" and not args.allow_incomplete:
        print("Remote verified public package is not ready; no files copied.", file=sys.stderr)
        return 2
    for src, dst in planned_paths(args):
        rsync_one(src, dst, execute=args.execute)
    if args.execute and probe.get("summary_status") == "verified":
        return audit_local_package(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

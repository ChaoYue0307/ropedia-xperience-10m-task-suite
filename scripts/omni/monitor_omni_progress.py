#!/usr/bin/env python3
"""Print a compact progress snapshot for an omni fine-tuning run."""

from __future__ import annotations

import argparse
import json
import subprocess
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


def main() -> int:
    args = parse_args()
    root = args.workspace / "results" / "omni_finetune"
    pipeline_status = root / args.run_id / "pipeline_status.jsonl"
    train_progress = root / f"{args.run_id}_lora" / "progress.jsonl"
    metrics = root / f"{args.run_id}_eval" / "metrics.json"
    log_path = root / args.run_id / "logs" / "pipeline.log"

    print(f"Run: {args.run_id}")
    print(f"Pipeline log: {log_path}")
    print("\nGPU status: index, used MiB, total MiB, util %")
    print(nvidia_smi())

    print("\nRecent pipeline phases:")
    for row in read_jsonl(pipeline_status, args.last):
        print(json.dumps(row, ensure_ascii=False))

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

#!/usr/bin/env python3
"""Wait for an omni training run, then validate and evaluate it.

The script is intentionally a gate runner: it does not invent metrics and it
does not mark a run successful until the training progress ends in `complete`,
the configured checkpoint check passes, and the existing validators pass.

The default checkpoint check is for PEFT/LoRA safetensors because Qwen3-Omni is
the first implemented branch. Future backbones can reuse the same wait/eval
sequence with a model-specific evaluator and a different checkpoint gate.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_DATASET_RUN_ID = "xperience10m_qwen3_omni_128ep_fullsplit_fast8gpu"
DEFAULT_TRAIN_RUN_ID = "xperience10m_qwen3_omni_128ep_fullsplit_fast8gpu_lora_fsdp_full_train_noval_tail_logits_fullstatesave_v6"
DEFAULT_MODEL_ID = str(Path.home() / "Ropedia/modelscope_models/Qwen__Qwen3-Omni-30B-A3B-Instruct")


def parse_args() -> argparse.Namespace:
    workspace_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Watch a training run and launch validated eval gates.")
    parser.add_argument("--workspace", type=Path, default=workspace_default)
    parser.add_argument("--dataset-run-id", default=DEFAULT_DATASET_RUN_ID)
    parser.add_argument("--train-run-id", default=DEFAULT_TRAIN_RUN_ID)
    parser.add_argument("--eval-run-id")
    parser.add_argument("--eval-smoke-run-id")
    parser.add_argument("--dataset-jsonl", type=Path)
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--eval-script", default="scripts/omni/eval_qwen3_omni_lora.py")
    parser.add_argument("--validation-script", default="scripts/omni/validate_omni_finetune_run.py")
    parser.add_argument("--adapter-check", choices=["lora_safetensors", "none"], default="lora_safetensors")
    parser.add_argument("--status-jsonl", type=Path)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--timeout-hours", type=float, default=12.0)
    parser.add_argument("--smoke-sample-limit", type=int, default=8)
    parser.add_argument("--full-eval-sample-limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--backbone", default="qwen3_omni_lora")
    parser.add_argument("--expected-train-episodes", type=int, default=96)
    parser.add_argument("--expected-val-episodes", type=int, default=16)
    parser.add_argument("--expected-test-episodes", type=int, default=16)
    parser.add_argument("--expected-dataset-train-episodes", type=int, default=89)
    parser.add_argument("--expected-dataset-val-episodes", type=int, default=16)
    parser.add_argument("--expected-dataset-test-episodes", type=int, default=14)
    parser.add_argument("--expected-num-processes", type=int, default=8)
    parser.add_argument("--skip-full-eval", action="store_true")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def append_event(path: Path, event: str, **fields: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"event": event, "time": time.time(), **fields}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run_checked(cmd: list[str], *, cwd: Path, log_path: Path, status_path: Path, event: str) -> None:
    append_event(status_path, f"{event}_start", command=cmd, log=str(log_path))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, text=True)
    append_event(status_path, f"{event}_exit", returncode=proc.returncode, log=str(log_path))
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def adapter_shape_check(adapter_dir: Path) -> dict[str, Any]:
    from safetensors.torch import load_file

    candidates = sorted(adapter_dir.glob("adapter_model*.safetensors"))
    if not candidates:
        raise FileNotFoundError(f"No adapter_model*.safetensors file in {adapter_dir}")
    state = load_file(str(candidates[0]))
    bad = [(name, tuple(tensor.shape), int(tensor.numel())) for name, tensor in state.items() if tensor.numel() == 0 or len(tensor.shape) != 2]
    shape_counts = Counter(str(tuple(tensor.shape)) for tensor in state.values())
    prefixes = Counter(
        name.split(".")[2]
        if name.startswith("base_model.model.") and len(name.split(".")) > 2
        else name.split(".")[0]
        for name in state
    )
    payload = {
        "status": "pass" if not bad else "fail",
        "adapter_file": str(candidates[0]),
        "num_tensors": len(state),
        "num_bad_tensors": len(bad),
        "bad_tensors": bad[:50],
        "total_bytes": sum(tensor.numel() * tensor.element_size() for tensor in state.values()),
        "shape_counts": dict(shape_counts),
        "prefixes": dict(prefixes),
    }
    if bad:
        raise ValueError(f"Adapter has invalid LoRA tensors: {bad[:5]}")
    return payload


def validation_cmd(args: argparse.Namespace, stage: str, output: Path, eval_run_id: str | None = None) -> list[str]:
    cmd = [
        sys.executable,
        args.validation_script,
        "--workspace",
        str(args.workspace),
        "--run-id",
        args.dataset_run_id,
        "--dataset-run-id",
        args.dataset_run_id,
        "--train-run-id",
        args.train_run_id,
        "--backbone",
        args.backbone,
        "--require-stage",
        stage,
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
        "--output",
        str(output),
    ]
    if eval_run_id:
        cmd.extend(["--eval-run-id", eval_run_id])
    return cmd


def eval_cmd(args: argparse.Namespace, run_id: str, sample_limit: int) -> list[str]:
    cmd = [
        sys.executable,
        args.eval_script,
        "--dataset-jsonl",
        str(args.dataset_jsonl),
        "--model-id",
        args.model_id,
        "--adapter-dir",
        str(args.adapter_dir),
        "--run-id",
        run_id,
        "--eval-split",
        "test",
        "--train-split",
        "train",
        "--max-new-tokens",
        str(args.max_new_tokens),
    ]
    if sample_limit > 0:
        cmd.extend(["--sample-limit", str(sample_limit)])
    if args.local_files_only:
        cmd.append("--local-files-only")
    return cmd


def wait_for_training_complete(args: argparse.Namespace, progress_path: Path, status_path: Path) -> None:
    deadline = time.time() + args.timeout_hours * 3600
    last_event = None
    while time.time() < deadline:
        rows = read_jsonl(progress_path)
        event = rows[-1].get("event") if rows else "missing_progress"
        if event != last_event:
            append_event(status_path, "training_progress", progress=str(progress_path), latest_event=event, rows=len(rows))
            last_event = event
        if event == "complete":
            return
        if event in {"failed", "error"}:
            raise RuntimeError(f"Training progress ended with {event}: {rows[-1]}")
        time.sleep(args.poll_seconds)
    raise TimeoutError(f"Timed out waiting for training complete: {progress_path}")


def main() -> int:
    args = parse_args()
    args.workspace = args.workspace.expanduser().resolve()
    root = args.workspace / "results" / "omni_finetune"
    run_dir = root / args.dataset_run_id
    train_dir = root / args.train_run_id
    args.dataset_jsonl = args.dataset_jsonl or root / f"{args.dataset_run_id}_dataset" / "dataset.jsonl"
    args.adapter_dir = args.adapter_dir or args.workspace / "checkpoints" / args.train_run_id / "adapter_lora"
    eval_run_id = args.eval_run_id or f"{args.train_run_id}_eval_test_full"
    eval_smoke_run_id = args.eval_smoke_run_id or f"{args.train_run_id}_eval_smoke{args.smoke_sample_limit}"
    status_path = args.status_jsonl or run_dir / f"watch_{args.train_run_id}.jsonl"
    progress_path = train_dir / "progress.jsonl"

    append_event(
        status_path,
        "watch_start",
        dataset_run_id=args.dataset_run_id,
        train_run_id=args.train_run_id,
        eval_smoke_run_id=eval_smoke_run_id,
        eval_run_id=eval_run_id,
    )
    wait_for_training_complete(args, progress_path, status_path)
    append_event(status_path, "training_complete_observed", progress=str(progress_path))

    if args.adapter_check == "lora_safetensors":
        shape_payload = adapter_shape_check(args.adapter_dir)
        shape_path = run_dir / f"adapter_shape_check_{args.train_run_id}.json"
        write_json(shape_path, shape_payload)
        append_event(status_path, "adapter_shape_check_done", output=str(shape_path), **{k: shape_payload[k] for k in ("status", "num_tensors", "num_bad_tensors", "total_bytes")})
    else:
        append_event(status_path, "adapter_shape_check_skipped", adapter_check=args.adapter_check)

    validation_training = run_dir / f"validation_training_{args.train_run_id}.json"
    run_checked(
        validation_cmd(args, "training", validation_training),
        cwd=args.workspace,
        log_path=run_dir / f"validate_training_{args.train_run_id}.log",
        status_path=status_path,
        event="validation_training",
    )

    run_checked(
        eval_cmd(args, eval_smoke_run_id, args.smoke_sample_limit),
        cwd=args.workspace,
        log_path=run_dir / f"eval_{eval_smoke_run_id}.log",
        status_path=status_path,
        event="eval_smoke",
    )

    if not args.skip_full_eval:
        run_checked(
            eval_cmd(args, eval_run_id, args.full_eval_sample_limit),
            cwd=args.workspace,
            log_path=run_dir / f"eval_{eval_run_id}.log",
            status_path=status_path,
            event="eval_full",
        )
        validation_eval = run_dir / f"validation_eval_{eval_run_id}.json"
        run_checked(
            validation_cmd(args, "eval", validation_eval, eval_run_id=eval_run_id),
            cwd=args.workspace,
            log_path=run_dir / f"validate_eval_{eval_run_id}.log",
            status_path=status_path,
            event="validation_eval",
        )

    append_event(status_path, "watch_complete", train_run_id=args.train_run_id, eval_run_id=eval_run_id)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        # Keep the failure visible in stdout/stderr for nohup logs.
        print(f"watch_omni_train_then_eval failed: {exc}", file=sys.stderr)
        raise

#!/usr/bin/env python3
"""Smoke-test verified result packaging for every configured omni backbone."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from backbone_registry import load_registry


FORBIDDEN_SUFFIXES = {
    ".hdf5",
    ".mp4",
    ".mov",
    ".rrd",
    ".safetensors",
    ".pt",
    ".pth",
    ".ckpt",
    ".bin",
    ".tar",
    ".gz",
    ".zip",
}


def parse_args() -> argparse.Namespace:
    workspace_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=workspace_default)
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_required_eval_file(path: Path, backbone: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.name == "metrics.json":
        metrics = {
            "eval_split": "test",
            "num_samples": 1,
            "num_eval_episodes": 1,
            "held_out_episode_count": 1,
        }
        for metric in backbone.get("primary_metrics", []):
            metrics.setdefault(str(metric), 1.0)
        write_json(path, metrics)
    elif path.suffix == ".jsonl":
        path.write_text(json.dumps({"id": "synthetic_0", "ok": True}) + "\n", encoding="utf-8")
    elif path.suffix == ".csv":
        path.write_text("id,value\nsynthetic_0,1\n", encoding="utf-8")
    elif path.suffix == ".md":
        path.write_text("# Synthetic Run Report\n\nContract smoke test artifact.\n", encoding="utf-8")
    elif path.suffix == ".json":
        write_json(path, {"status": "synthetic"})
    else:
        path.write_text("synthetic\n", encoding="utf-8")


def create_synthetic_run(temp_workspace: Path, backbone: dict[str, Any]) -> tuple[str, str, str]:
    backbone_id = str(backbone["id"])
    dataset_run_id = f"synthetic_{backbone_id}_dataset_run"
    train_run_id = f"synthetic_{backbone_id}_train"
    eval_run_id = f"synthetic_{backbone_id}_eval"
    root = temp_workspace / "results" / "omni_finetune"
    run_dir = root / dataset_run_id
    dataset_dir = root / f"{dataset_run_id}_dataset"
    train_dir = root / train_run_id
    eval_dir = root / eval_run_id

    write_json(
        run_dir / "episode_manifest.json",
        {"episodes": [{"episode_id": "synthetic_episode", "split": "test"}]},
    )
    write_json(
        dataset_dir / "dataset_manifest.json",
        {
            "num_samples": 1,
            "num_episodes": 1,
            "split_counts": {"test": 1},
            "skipped_episodes": [],
        },
    )
    write_json(
        train_dir / "training_metadata.json",
        {"num_processes": 1, "num_train_samples": 1, "num_val_samples": 0, "history": []},
    )
    train_dir.mkdir(parents=True, exist_ok=True)
    (train_dir / "progress.jsonl").write_text(json.dumps({"event": "complete"}) + "\n", encoding="utf-8")
    write_json(run_dir / f"validation_eval_{eval_run_id}.json", {"status": "pass", "summary": {"eval_dir": str(eval_dir)}})

    required_files = backbone["artifact_contract"]["required_eval_files"]
    for filename in required_files:
        write_required_eval_file(eval_dir / filename, backbone)
    return dataset_run_id, train_run_id, eval_run_id


def assert_public_safe(output_dir: Path) -> None:
    bad = [str(path.relative_to(output_dir)) for path in output_dir.rglob("*") if path.is_file() and path.suffix in FORBIDDEN_SUFFIXES]
    if bad:
        raise AssertionError(f"Forbidden files were packaged: {bad}")


def run_package(repo: Path, temp_workspace: Path, backbone: dict[str, Any]) -> dict[str, Any]:
    dataset_run_id, train_run_id, eval_run_id = create_synthetic_run(temp_workspace, backbone)
    cmd = [
        sys.executable,
        str(repo / "scripts" / "omni" / "package_verified_omni_result.py"),
        "--workspace",
        str(temp_workspace),
        "--dataset-run-id",
        dataset_run_id,
        "--train-run-id",
        train_run_id,
        "--eval-run-id",
        eval_run_id,
        "--backbone",
        str(backbone["id"]),
    ]
    subprocess.run(cmd, cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    output_dir = temp_workspace / "results" / "omni_finetune" / "verified_public" / eval_run_id
    summary = json.loads((output_dir / "verified_result_summary.json").read_text(encoding="utf-8"))
    audit_cmd = [
        sys.executable,
        str(repo / "scripts" / "omni" / "audit_verified_omni_package.py"),
        "--workspace",
        str(temp_workspace),
        "--package-dir",
        str(output_dir),
        "--backbone",
        str(backbone["id"]),
    ]
    audit_proc = subprocess.run(audit_cmd, cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    audit_payload = json.loads(audit_proc.stdout)
    assert summary["status"] == "verified"
    assert summary["required_eval_files"] == backbone["artifact_contract"]["required_eval_files"]
    assert set(summary["eval"]["primary_metrics"]) == set(backbone.get("primary_metrics", []))
    assert audit_payload["status"] == "pass"
    assert_public_safe(output_dir)
    return {
        "backbone": backbone["id"],
        "output_dir": str(output_dir),
        "audit_status": audit_payload["status"],
        "required_eval_files": summary["required_eval_files"],
        "primary_metrics": summary["eval"]["primary_metrics"],
    }


def main() -> int:
    args = parse_args()
    repo = args.workspace.expanduser().resolve()
    temp_root = Path(tempfile.mkdtemp(prefix="omni_packaging_contract_"))
    try:
        shutil.copytree(repo / "configs", temp_root / "configs")
        registry = load_registry(repo / "configs" / "omni_backbones")
        results = [run_package(repo, temp_root, registry[backbone_id]) for backbone_id in sorted(registry)]
        print(json.dumps({"status": "pass", "temp_workspace": str(temp_root), "backbones": results}, indent=2))
    finally:
        if args.keep_temp:
            print(f"Kept temp workspace: {temp_root}", file=sys.stderr)
        else:
            shutil.rmtree(temp_root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

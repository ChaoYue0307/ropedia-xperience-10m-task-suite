#!/usr/bin/env python3
"""Validate Xperience-10M omni fine-tuning run artifacts.

This is a run-completion gate, not a training script. It checks that the shared
episode split, exported dataset, LoRA training artifacts, and held-out eval
outputs agree with the declared backbone contract.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from backbone_registry import load_registry


STAGE_ORDER = {
    "manifest": 1,
    "dataset": 2,
    "training": 3,
    "eval": 4,
}

REQUIRED_EVAL_FILES = [
    "metrics.json",
    "predictions.jsonl",
    "predictions.csv",
    "per_class_metrics.csv",
    "confusion_matrix.csv",
    "RUN_REPORT.md",
]


def parse_args() -> argparse.Namespace:
    workspace_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Validate an omni fine-tuning run.")
    parser.add_argument("--workspace", type=Path, default=workspace_default)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--backbone", default="qwen3_omni_lora")
    parser.add_argument("--require-stage", choices=sorted(STAGE_ORDER), default="manifest")
    parser.add_argument("--expected-train-episodes", type=int, default=96)
    parser.add_argument("--expected-val-episodes", type=int, default=16)
    parser.add_argument("--expected-test-episodes", type=int, default=16)
    parser.add_argument("--expected-num-processes", type=int, default=8)
    parser.add_argument("--min-json-validity", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def add_issue(issues: list[dict[str, str]], stage: str, message: str, severity: str = "error") -> None:
    issues.append({"stage": stage, "severity": severity, "message": message})


def expected_counts(args: argparse.Namespace) -> dict[str, int]:
    return {
        "train": args.expected_train_episodes,
        "val": args.expected_val_episodes,
        "test": args.expected_test_episodes,
    }


def manifest_counts(manifest: dict[str, Any]) -> dict[str, int]:
    episodes = manifest.get("episodes", [])
    return dict(Counter(str(ep.get("split", "unspecified")) for ep in episodes))


def split_sets(rows: list[dict[str, Any]], key: str = "episode_id") -> dict[str, set[str]]:
    sets: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        sets[str(row.get("split", "unspecified"))].add(str(row.get(key, "")))
    return sets


def leakage_pairs(sets: dict[str, set[str]]) -> list[dict[str, Any]]:
    leaks = []
    splits = sorted(sets)
    for idx, left in enumerate(splits):
        for right in splits[idx + 1 :]:
            overlap = sorted(sets[left] & sets[right])
            if overlap:
                leaks.append({"left": left, "right": right, "overlap": overlap[:20], "count": len(overlap)})
    return leaks


def validate_manifest(args: argparse.Namespace, run_dir: Path, issues: list[dict[str, str]]) -> dict[str, Any]:
    path = run_dir / "episode_manifest.json"
    summary: dict[str, Any] = {"path": str(path)}
    try:
        manifest = read_json(path)
    except FileNotFoundError:
        add_issue(issues, "manifest", f"missing manifest: {path}")
        return summary

    episodes = manifest.get("episodes", [])
    counts = manifest_counts(manifest)
    summary.update({"episode_count": len(episodes), "split_counts": counts})
    expected = expected_counts(args)
    if counts != expected:
        add_issue(issues, "manifest", f"split counts {counts} do not match expected {expected}")
    ids = [str(ep.get("episode_id", "")) for ep in episodes]
    if len(ids) != len(set(ids)):
        add_issue(issues, "manifest", "duplicate episode_id values in manifest")
    by_split = split_sets(episodes)
    for leak in leakage_pairs(by_split):
        add_issue(issues, "manifest", f"episode leakage across splits: {leak}")

    session_sets: dict[str, set[str]] = defaultdict(set)
    for ep in episodes:
        session = ep.get("top_level_session") or ep.get("source_episode_id")
        if session:
            session_sets[str(ep.get("split", "unspecified"))].add(str(session))
    session_leaks = leakage_pairs(session_sets)
    summary["session_leakage"] = session_leaks
    for leak in session_leaks:
        add_issue(issues, "manifest", f"session leakage across splits: {leak}")
    return summary


def validate_dataset(dataset_dir: Path, issues: list[dict[str, str]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"dataset_dir": str(dataset_dir)}
    manifest_path = dataset_dir / "dataset_manifest.json"
    dataset_path = dataset_dir / "dataset.jsonl"
    try:
        dataset_manifest = read_json(manifest_path)
        rows = read_jsonl(dataset_path)
    except FileNotFoundError as exc:
        add_issue(issues, "dataset", f"missing dataset artifact: {exc}")
        return summary

    row_counts = Counter(str(row.get("split", "unspecified")) for row in rows)
    episode_sets = split_sets(rows)
    episode_counts = {split: len(values) for split, values in episode_sets.items()}
    summary.update({
        "manifest_path": str(manifest_path),
        "dataset_path": str(dataset_path),
        "manifest_num_samples": dataset_manifest.get("num_samples"),
        "row_count": len(rows),
        "sample_split_counts": dict(row_counts),
        "episode_split_counts": episode_counts,
        "skipped_episodes": len(dataset_manifest.get("skipped_episodes", [])),
    })
    if dataset_manifest.get("num_samples") != len(rows):
        add_issue(issues, "dataset", "dataset_manifest num_samples does not match dataset.jsonl rows")
    for split in ("train", "val", "test"):
        if row_counts.get(split, 0) <= 0:
            add_issue(issues, "dataset", f"no exported samples for split {split}")
    for leak in leakage_pairs(episode_sets):
        add_issue(issues, "dataset", f"episode leakage in exported records: {leak}")
    required_fields = {"id", "episode_id", "split", "media", "answer_json", "messages", "label_options"}
    for idx, row in enumerate(rows[:100]):
        missing = sorted(required_fields - set(row))
        if missing:
            add_issue(issues, "dataset", f"row {idx} missing fields: {missing}")
            break
    return summary


def validate_training(args: argparse.Namespace, workspace: Path, run_id: str, issues: list[dict[str, str]]) -> dict[str, Any]:
    train_dir = workspace / "results" / "omni_finetune" / f"{run_id}_lora"
    checkpoint_dir = workspace / "checkpoints" / f"{run_id}_lora" / "adapter_lora"
    summary: dict[str, Any] = {"train_dir": str(train_dir), "checkpoint_dir": str(checkpoint_dir)}
    try:
        metadata = read_json(train_dir / "training_metadata.json")
    except FileNotFoundError:
        add_issue(issues, "training", f"missing training metadata: {train_dir / 'training_metadata.json'}")
        return summary
    summary.update({
        "num_processes": metadata.get("num_processes"),
        "num_train_samples": metadata.get("num_train_samples"),
        "num_val_samples": metadata.get("num_val_samples"),
        "history_len": len(metadata.get("history", [])),
        "checkpoint_dir_recorded": metadata.get("checkpoint_dir"),
    })
    if int(metadata.get("num_processes", 0)) != args.expected_num_processes:
        add_issue(issues, "training", f"num_processes is {metadata.get('num_processes')}, expected {args.expected_num_processes}")
    if int(metadata.get("num_train_samples", 0)) <= 0 or int(metadata.get("num_val_samples", 0)) <= 0:
        add_issue(issues, "training", "training metadata has empty train or validation samples")
    for filename in ("adapter_config.json", "training_metadata.json"):
        if not (checkpoint_dir / filename).exists():
            add_issue(issues, "training", f"missing checkpoint artifact: {checkpoint_dir / filename}")
    if not any(checkpoint_dir.glob("adapter_model.*")):
        add_issue(issues, "training", f"missing adapter_model file in {checkpoint_dir}")
    progress = train_dir / "progress.jsonl"
    try:
        progress_rows = read_jsonl(progress)
    except FileNotFoundError:
        add_issue(issues, "training", f"missing training progress: {progress}")
        progress_rows = []
    if progress_rows and progress_rows[-1].get("event") != "complete":
        add_issue(issues, "training", "training progress does not end with complete")
    return summary


def validate_eval(args: argparse.Namespace, workspace: Path, run_id: str, issues: list[dict[str, str]]) -> dict[str, Any]:
    eval_dir = workspace / "results" / "omni_finetune" / f"{run_id}_eval"
    summary: dict[str, Any] = {"eval_dir": str(eval_dir)}
    for filename in REQUIRED_EVAL_FILES:
        if not (eval_dir / filename).exists():
            add_issue(issues, "eval", f"missing eval artifact: {eval_dir / filename}")
    try:
        metrics = read_json(eval_dir / "metrics.json")
    except FileNotFoundError:
        return summary
    predictions = []
    try:
        predictions = read_jsonl(eval_dir / "predictions.jsonl")
    except FileNotFoundError:
        pass
    summary.update({
        "eval_split": metrics.get("eval_split"),
        "num_eval_episodes": metrics.get("num_eval_episodes"),
        "json_validity_rate": metrics.get("json_validity_rate"),
        "action_macro_f1": metrics.get("action_macro_f1"),
        "prediction_rows": len(predictions),
    })
    if metrics.get("eval_split") != "test":
        add_issue(issues, "eval", f"eval_split is {metrics.get('eval_split')}, expected test")
    if int(metrics.get("num_eval_episodes", 0)) <= 0:
        add_issue(issues, "eval", "num_eval_episodes is empty")
    if float(metrics.get("json_validity_rate", 0.0)) < args.min_json_validity:
        add_issue(issues, "eval", f"json_validity_rate below threshold {args.min_json_validity}")
    if not predictions:
        add_issue(issues, "eval", "predictions.jsonl has no rows")
    return summary


def main() -> int:
    args = parse_args()
    workspace = args.workspace.expanduser().resolve()
    root = workspace / "results" / "omni_finetune"
    run_dir = root / args.run_id
    dataset_dir = root / f"{args.run_id}_dataset"

    issues: list[dict[str, str]] = []
    registry = load_registry(workspace / "configs" / "omni_backbones")
    if args.backbone not in registry:
        add_issue(issues, "backbone", f"unknown backbone: {args.backbone}")
        backbone = {}
    else:
        backbone = registry[args.backbone]

    summary: dict[str, Any] = {
        "run_id": args.run_id,
        "backbone": args.backbone,
        "backbone_status": backbone.get("status"),
        "required_stage": args.require_stage,
        "workspace": str(workspace),
    }
    summary["manifest"] = validate_manifest(args, run_dir, issues)
    if STAGE_ORDER[args.require_stage] >= STAGE_ORDER["dataset"]:
        summary["dataset"] = validate_dataset(dataset_dir, issues)
    if STAGE_ORDER[args.require_stage] >= STAGE_ORDER["training"]:
        summary["training"] = validate_training(args, workspace, args.run_id, issues)
    if STAGE_ORDER[args.require_stage] >= STAGE_ORDER["eval"]:
        summary["eval"] = validate_eval(args, workspace, args.run_id, issues)

    errors = [issue for issue in issues if issue["severity"] == "error"]
    payload = {
        "status": "pass" if not errors else "fail",
        "summary": summary,
        "issues": issues,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Audit a verified omni public package before publication updates."""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--backbone", help="Expected backbone id. Defaults to verified_result_summary.json.")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def add_issue(issues: list[dict[str, str]], stage: str, message: str, severity: str = "error") -> None:
    issues.append({"stage": stage, "severity": severity, "message": message})


def forbidden_files(package_dir: Path) -> list[str]:
    return [
        str(path.relative_to(package_dir))
        for path in package_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES
    ]


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return sum(1 for line in handle if line.strip())


def audit(args: argparse.Namespace) -> dict[str, Any]:
    workspace = args.workspace.expanduser().resolve()
    package_dir = args.package_dir.expanduser().resolve()
    try:
        package_label = package_dir.relative_to(workspace).as_posix()
    except ValueError:
        package_label = package_dir.name
    summary_path = package_dir / "verified_result_summary.json"
    issues: list[dict[str, str]] = []

    if not summary_path.exists():
        add_issue(issues, "summary", f"missing verified_result_summary.json: {summary_path}")
        return {"status": "fail", "package_dir": package_label, "issues": issues}

    summary = read_json(summary_path)
    backbone_id = args.backbone or summary.get("backbone")
    registry = load_registry(workspace / "configs" / "omni_backbones")
    if backbone_id not in registry:
        add_issue(issues, "backbone", f"unknown backbone: {backbone_id}")
        backbone = {}
    else:
        backbone = registry[backbone_id]

    if summary.get("status") != "verified":
        add_issue(issues, "summary", f"package status is {summary.get('status')}, expected verified")
    if summary.get("backbone") != backbone_id:
        add_issue(issues, "summary", f"summary backbone is {summary.get('backbone')}, expected {backbone_id}")

    eval_dir = package_dir / "eval"
    required_eval_files = list((backbone.get("artifact_contract") or {}).get("required_eval_files", []))
    included_files = set(summary.get("included_files", []))
    for filename in required_eval_files:
        rel = f"eval/{filename}"
        path = eval_dir / filename
        if not path.exists():
            add_issue(issues, "eval", f"missing required packaged eval file: {rel}")
        if rel not in included_files:
            add_issue(issues, "summary", f"required eval file missing from included_files: {rel}")

    eval_summary = summary.get("eval") or {}
    prediction_file = eval_summary.get("prediction_file")
    if prediction_file:
        prediction_path = eval_dir / str(prediction_file)
        if not prediction_path.exists():
            add_issue(issues, "eval", f"prediction file missing: eval/{prediction_file}")
        elif prediction_path.suffix == ".jsonl" and count_jsonl(prediction_path) <= 0:
            add_issue(issues, "eval", f"prediction file has no rows: eval/{prediction_file}")
    if int(eval_summary.get("prediction_rows") or 0) <= 0:
        add_issue(issues, "summary", "prediction_rows is empty")
    if int(eval_summary.get("held_out_episode_count") or eval_summary.get("num_eval_episodes") or 0) <= 0:
        add_issue(issues, "summary", "held-out episode count is empty")

    primary_metrics = eval_summary.get("primary_metrics") or {}
    for metric in backbone.get("primary_metrics", []):
        if metric not in primary_metrics:
            add_issue(issues, "metrics", f"missing primary metric in summary: {metric}")
        elif primary_metrics.get(metric) is None:
            add_issue(issues, "metrics", f"primary metric is null: {metric}")

    validation_path = package_dir / "validation" / "eval.json"
    if not validation_path.exists():
        add_issue(issues, "validation", "missing validation/eval.json")
    else:
        validation = read_json(validation_path)
        if validation.get("status") != "pass":
            add_issue(issues, "validation", f"validation status is {validation.get('status')}, expected pass")

    bad_files = forbidden_files(package_dir)
    for rel in bad_files:
        add_issue(issues, "public_safety", f"forbidden file in package: {rel}")

    errors = [issue for issue in issues if issue["severity"] == "error"]
    return {
        "status": "pass" if not errors else "fail",
        "package_dir": package_label,
        "backbone": backbone_id,
        "required_eval_files": required_eval_files,
        "primary_metrics": sorted(primary_metrics),
        "issues": issues,
    }


def main() -> int:
    args = parse_args()
    payload = audit(args)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Registry for Xperience-10M fine-tuning backbone contracts.

The registry keeps the shared data/split/evaluation spine separate from
backbone-specific train/eval code. Qwen3-Omni is currently implemented; world
model and policy branches are explicit extension contracts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_KEYS = {
    "id",
    "display_name",
    "status",
    "model_family",
    "dataset_contract",
    "training_objective",
    "split_policy",
    "modalities",
    "entrypoints",
    "primary_metrics",
    "artifact_contract",
    "extension_requirements",
}

IMPLEMENTED_STATUS = "implemented"
PLANNED_STATUS = "planned_adapter"
DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "omni_backbones"
ARTIFACT_CONTRACT_KEYS = {
    "checkpoint_gate",
    "required_eval_files",
    "required_training_files",
    "public_package_allowed",
    "public_package_forbidden",
}
REQUIRED_SPLITS = {"train", "val", "test"}
REQUIRED_PUBLIC_FORBIDDEN_TERMS = {
    "mp4",
    "hdf5",
    "rrd",
    "weights",
    "checkpoints",
    "archives",
}


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    missing = sorted(REQUIRED_KEYS - set(payload))
    if missing:
        raise ValueError(f"{path} is missing required keys: {missing}")
    artifact_contract = payload.get("artifact_contract", {})
    missing_artifact = sorted(ARTIFACT_CONTRACT_KEYS - set(artifact_contract))
    if missing_artifact:
        raise ValueError(f"{path} artifact_contract is missing required keys: {missing_artifact}")
    payload["_config_path"] = str(path)
    return payload


def load_registry(config_dir: Path = DEFAULT_CONFIG_DIR) -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {}
    for path in sorted(config_dir.glob("*.json")):
        payload = load_config(path)
        backbone_id = str(payload["id"])
        if backbone_id in configs:
            raise ValueError(f"Duplicate backbone id: {backbone_id}")
        configs[backbone_id] = payload
    if not configs:
        raise ValueError(f"No backbone configs found in {config_dir}")
    return configs


def implemented_entrypoints(config: dict[str, Any]) -> dict[str, str]:
    entrypoints = config.get("entrypoints", {})
    return {key: value for key, value in entrypoints.items() if value}


def summarize(config: dict[str, Any]) -> dict[str, Any]:
    entrypoints = implemented_entrypoints(config)
    return {
        "id": config["id"],
        "display_name": config["display_name"],
        "status": config["status"],
        "model_family": config["model_family"],
        "dataset_contract": config["dataset_contract"],
        "training_objective": config["training_objective"],
        "implemented_entrypoints": sorted(entrypoints),
        "missing_entrypoints": sorted(
            key for key, value in config.get("entrypoints", {}).items() if not value
        ),
        "primary_metrics": config["primary_metrics"],
        "checkpoint_gate": config["artifact_contract"]["checkpoint_gate"],
        "required_eval_file_count": len(config["artifact_contract"]["required_eval_files"]),
        "extension_requirement_count": len(config.get("extension_requirements", [])),
    }


def validate_implemented(config: dict[str, Any], workspace: Path) -> list[str]:
    issues: list[str] = []
    if config.get("status") != IMPLEMENTED_STATUS:
        return issues
    for name, rel_path in implemented_entrypoints(config).items():
        path = workspace / rel_path
        if not path.exists():
            issues.append(f"{config['id']} entrypoint {name} not found: {rel_path}")
    return issues


def text_contains_any(values: list[Any], term: str) -> bool:
    needle = term.lower()
    return any(needle in str(value).lower() for value in values)


def validate_contract(config: dict[str, Any], workspace: Path) -> list[str]:
    issues = validate_implemented(config, workspace)
    backbone_id = config["id"]
    status = config.get("status")
    if status not in {IMPLEMENTED_STATUS, PLANNED_STATUS}:
        issues.append(f"{backbone_id} has unsupported status: {status}")

    split_policy = config.get("split_policy") or {}
    default_counts = split_policy.get("default_counts") or {}
    missing_splits = sorted(REQUIRED_SPLITS - set(default_counts))
    if missing_splits:
        issues.append(f"{backbone_id} split_policy.default_counts missing splits: {missing_splits}")
    for split in sorted(REQUIRED_SPLITS & set(default_counts)):
        if int(default_counts.get(split) or 0) <= 0:
            issues.append(f"{backbone_id} split {split} has non-positive default count: {default_counts.get(split)}")
    if not split_policy.get("leakage_guard"):
        issues.append(f"{backbone_id} split_policy.leakage_guard is empty")

    primary_metrics = list(config.get("primary_metrics") or [])
    if not primary_metrics:
        issues.append(f"{backbone_id} primary_metrics is empty")
    if "held_out_episode_count" not in primary_metrics:
        issues.append(f"{backbone_id} primary_metrics must include held_out_episode_count")

    artifact_contract = config.get("artifact_contract") or {}
    required_eval = list(artifact_contract.get("required_eval_files") or [])
    required_training = list(artifact_contract.get("required_training_files") or [])
    if "metrics.json" not in required_eval:
        issues.append(f"{backbone_id} required_eval_files must include metrics.json")
    if "RUN_REPORT.md" not in required_eval:
        issues.append(f"{backbone_id} required_eval_files must include RUN_REPORT.md")
    if not any(str(filename).endswith(".jsonl") for filename in required_eval):
        issues.append(f"{backbone_id} required_eval_files must include a JSONL prediction file")
    if "training_metadata.json" not in required_training:
        issues.append(f"{backbone_id} required_training_files must include training_metadata.json")
    if "progress.jsonl" not in required_training:
        issues.append(f"{backbone_id} required_training_files must include progress.jsonl")

    public_allowed = list(artifact_contract.get("public_package_allowed") or [])
    public_forbidden = list(artifact_contract.get("public_package_forbidden") or [])
    if not public_allowed:
        issues.append(f"{backbone_id} public_package_allowed is empty")
    if not public_forbidden:
        issues.append(f"{backbone_id} public_package_forbidden is empty")
    for term in sorted(REQUIRED_PUBLIC_FORBIDDEN_TERMS):
        if not text_contains_any(public_forbidden, term):
            issues.append(f"{backbone_id} public_package_forbidden should mention {term}")
    return issues


def parse_args() -> argparse.Namespace:
    workspace_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Inspect configured omni fine-tuning backbones.")
    parser.add_argument("--workspace", type=Path, default=workspace_default)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--backbone", help="Print one backbone contract.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument("--validate", action="store_true", help="Validate backbone contracts and implemented entrypoint files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.workspace = args.workspace.expanduser().resolve()
    registry = load_registry(args.config_dir.expanduser().resolve())
    if args.backbone:
        if args.backbone not in registry:
            raise SystemExit(f"Unknown backbone {args.backbone}. Available: {', '.join(sorted(registry))}")
        payload: Any = registry[args.backbone]
    else:
        payload = {
            "config_dir": str(args.config_dir),
            "backbones": [summarize(config) for config in registry.values()],
        }

    issues: list[str] = []
    if args.validate:
        for config in registry.values():
            issues.extend(validate_contract(config, args.workspace))
        if isinstance(payload, dict):
            payload = {**payload, "validation_issues": issues}

    if args.json or args.backbone:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        for row in payload["backbones"]:
            print(f"{row['id']}: {row['status']} - {row['training_objective']}")
        if issues:
            print("\nValidation issues:")
            for issue in issues:
                print(f"- {issue}")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())

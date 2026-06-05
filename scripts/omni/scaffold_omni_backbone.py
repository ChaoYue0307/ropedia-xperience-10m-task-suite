#!/usr/bin/env python3
"""Create a validated omni backbone config from an existing contract template."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from backbone_registry import load_config, load_registry


def parse_args() -> argparse.Namespace:
    workspace_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=workspace_default)
    parser.add_argument("--template-backbone", default="policy_vla_branch")
    parser.add_argument("--id", required=True, help="New backbone id, e.g. cosmos_video2world_branch.")
    parser.add_argument("--display-name", required=True)
    parser.add_argument("--model-family", required=True)
    parser.add_argument("--dataset-contract", required=True)
    parser.add_argument("--training-objective", required=True)
    parser.add_argument("--checkpoint-gate", required=True)
    parser.add_argument("--local-model-env")
    parser.add_argument("--status", default="planned_adapter")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def config_filename(backbone_id: str) -> str:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_]*", backbone_id):
        raise ValueError("--id must use lowercase letters, digits, and underscores, and start with a letter or digit")
    return f"{backbone_id}.json"


def clone_entrypoints(template: dict[str, Any]) -> dict[str, Any]:
    entrypoints = dict(template.get("entrypoints", {}))
    for key in ("export", "train", "eval", "launcher", "upload", "watch"):
        if key in entrypoints:
            entrypoints[key] = None
    return entrypoints


def scaffold(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    workspace = args.workspace.expanduser().resolve()
    config_dir = args.output_dir or workspace / "configs" / "omni_backbones"
    config_dir = config_dir.expanduser().resolve()
    registry = load_registry(config_dir)
    if args.id in registry and not args.overwrite:
        raise FileExistsError(f"Backbone id already exists: {args.id}")
    if args.template_backbone not in registry:
        raise KeyError(f"Unknown template backbone {args.template_backbone}. Available: {', '.join(sorted(registry))}")

    template = registry[args.template_backbone]
    payload = json.loads(json.dumps(template))
    payload.pop("_config_path", None)
    payload.update({
        "id": args.id,
        "display_name": args.display_name,
        "status": args.status,
        "model_family": args.model_family,
        "default_model_id": None,
        "local_model_env": args.local_model_env or f"{args.id.upper()}_MODEL_DIR",
        "dataset_contract": args.dataset_contract,
        "training_objective": args.training_objective,
        "entrypoints": clone_entrypoints(template),
        "primary_metrics": ["held_out_episode_count"],
        "extension_requirements": [
            "Define the model-specific exporter for this dataset contract.",
            "Define the training launcher and checkpoint save format.",
            "Define the held-out evaluator and metric files.",
            "Define public-safe packaging rules before reporting public metrics.",
        ],
    })
    artifact_contract = dict(payload.get("artifact_contract") or {})
    artifact_contract["checkpoint_gate"] = args.checkpoint_gate
    payload["artifact_contract"] = artifact_contract

    output_path = config_dir / config_filename(args.id)
    load_config_payload_path = output_path
    if args.dry_run:
        load_config_payload_path = Path("/tmp") / output_path.name
        load_config_payload_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(output_path)
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    load_config(load_config_payload_path)
    if args.dry_run:
        load_config_payload_path.unlink(missing_ok=True)
    return payload, output_path


def main() -> int:
    args = parse_args()
    payload, output_path = scaffold(args)
    print(json.dumps({
        "status": "dry_run" if args.dry_run else "written",
        "output": str(output_path),
        "id": payload["id"],
        "template_backbone": args.template_backbone,
        "checkpoint_gate": payload["artifact_contract"]["checkpoint_gate"],
        "entrypoints": payload["entrypoints"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Audit whether a dataset can drive real Cosmos3-Super action fine-tuning.

The existing Cosmos3-Super Reasoner run evaluates base weights on structured
JSON QA. A true Cosmos3 Diffusers fine-tune is a different contract: the
transformer action path predicts continuous embodiment-domain action vectors,
not semantic JSON labels. This guard makes that distinction explicit and fails
closed until the exported Xperience-10M windows contain Cosmos-native action
targets.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

from qwen3_omni_dataset_utils import load_jsonl


REQUIRED_JSON_QA_FIELDS = {
    "action",
    "subtask",
    "objects",
    "contact",
    "transition",
    "next_action",
    "evidence_window",
}

ACTION_TARGET_KEYS = (
    "cosmos_action_target",
    "cosmos3_action_target",
    "cosmos_action_condition",
    "action_target",
)

REQUIRED_ACTION_TARGET_FIELDS = {
    "mode",
    "domain_name",
    "chunk_size",
}

ACTION_MODES = {"policy", "forward_dynamics", "inverse_dynamics"}

REQUIRED_SCHEMA = {
    "cosmos_action_target": {
        "mode": "policy|forward_dynamics|inverse_dynamics",
        "domain_name": "one Cosmos3 embodiment domain supported by CosmosActionCondition",
        "chunk_size": "positive integer action transition count",
        "raw_actions": "required for forward_dynamics; list[list[float]] with shape [T, raw_action_dim]",
        "video": "required for inverse_dynamics, or image/video conditioning for policy and forward_dynamics",
        "resolution_tier": "optional; one of 256, 480, 704, 720",
        "view_point": "optional; ego_view|third_person_view|wrist_view|concat_view",
    }
}


def parse_args() -> argparse.Namespace:
    workspace_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=workspace_default)
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument(
        "--backbone-config",
        type=Path,
        default=workspace_default / "configs" / "omni_backbones" / "cosmos3_super_reasoner.json",
    )
    parser.add_argument("--run-id", default="xperience10m_cosmos3_super_training_contract_audit")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument(
        "--require-trainable",
        action="store_true",
        help="Exit non-zero unless the dataset/model contract is ready for a real trainer launch.",
    )
    return parser.parse_args()


def read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")


def numeric_matrix(value: Any) -> tuple[bool, tuple[int, int] | None]:
    if not isinstance(value, list) or not value:
        return False, None
    width: int | None = None
    for row in value:
        if not isinstance(row, list) or not row:
            return False, None
        if width is None:
            width = len(row)
        elif len(row) != width:
            return False, None
        for item in row:
            if not isinstance(item, (int, float)) or not math.isfinite(float(item)):
                return False, None
    return True, (len(value), int(width or 0))


def find_action_target(row: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
    for key in ACTION_TARGET_KEYS:
        value = row.get(key)
        if isinstance(value, dict):
            return key, value
    return None, None


def media_has_video(row: dict[str, Any]) -> bool:
    media = row.get("media") if isinstance(row.get("media"), dict) else {}
    if media.get("mosaic_video_path") or row.get("primary_video_path"):
        return True
    video_paths = media.get("video_paths")
    return isinstance(video_paths, list) and any(isinstance(item, dict) and item.get("path") for item in video_paths)


def validate_action_target(target: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    missing = sorted(field for field in REQUIRED_ACTION_TARGET_FIELDS if field not in target)
    if missing:
        issues.append(f"missing fields: {missing}")
        return issues

    mode = str(target.get("mode"))
    if mode not in ACTION_MODES:
        issues.append(f"unsupported mode: {mode!r}")

    try:
        chunk_size = int(target.get("chunk_size"))
        if chunk_size < 1:
            issues.append("chunk_size must be >= 1")
    except Exception:
        issues.append("chunk_size must be an integer")
        chunk_size = 0

    if not str(target.get("domain_name") or "").strip():
        issues.append("domain_name is empty")

    raw_actions = target.get("raw_actions")
    if mode == "forward_dynamics":
        ok, shape = numeric_matrix(raw_actions)
        if not ok:
            issues.append("forward_dynamics requires numeric raw_actions shaped [T, raw_action_dim]")
        elif shape and shape[0] < 1:
            issues.append("raw_actions must include at least one action row")
    elif raw_actions is not None:
        ok, _ = numeric_matrix(raw_actions)
        if not ok:
            issues.append("raw_actions is present but is not a numeric matrix")

    return issues


def model_summary(model_dir: Path | None) -> dict[str, Any]:
    if model_dir is None:
        return {"provided": False}
    model_dir = model_dir.expanduser().resolve()
    config = read_json(model_dir / "config.json")
    transformer_config = read_json(model_dir / "transformer" / "config.json")
    inner = ((config.get("model") or {}).get("config") or {})
    return {
        "provided": True,
        "path": str(model_dir),
        "exists": model_dir.exists(),
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures"),
        "pipeline_class": read_json(model_dir / "model_index.json").get("_class_name"),
        "transformer_class": transformer_config.get("_class_name"),
        "action_gen": transformer_config.get("action_gen", inner.get("action_gen")),
        "action_dim": transformer_config.get("action_dim", inner.get("action_dim")),
        "lora_enabled_default": inner.get("lora_enabled"),
        "lora_rank_default": inner.get("lora_rank"),
        "lora_alpha_default": inner.get("lora_alpha"),
        "lora_target_modules_default": inner.get("lora_target_modules"),
        "rectified_flow_training_config_keys": sorted(
            ((inner.get("rectified_flow_training_config") or {}).keys())
        ),
    }


def dataset_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    split_counts = Counter(str(row.get("split", "unspecified")) for row in rows)
    episodes_by_split: dict[str, set[str]] = {}
    missing_json_answer = 0
    missing_json_fields = Counter()
    rows_with_video = 0
    rows_with_action_target = 0
    valid_action_targets = 0
    target_key_counts = Counter()
    target_mode_counts = Counter()
    target_issue_counts = Counter()
    examples: list[dict[str, Any]] = []

    for row in rows:
        split = str(row.get("split", "unspecified"))
        episodes_by_split.setdefault(split, set()).add(str(row.get("episode_id", "")))
        answer = row.get("answer_json") if isinstance(row.get("answer_json"), dict) else {}
        if not answer:
            missing_json_answer += 1
        for field in REQUIRED_JSON_QA_FIELDS:
            if field not in answer:
                missing_json_fields[field] += 1
        if media_has_video(row):
            rows_with_video += 1

        key, target = find_action_target(row)
        if target is None:
            continue
        rows_with_action_target += 1
        target_key_counts[str(key)] += 1
        target_mode_counts[str(target.get("mode", "missing"))] += 1
        issues = validate_action_target(target)
        if issues:
            for issue in issues:
                target_issue_counts[issue] += 1
            if len(examples) < 5:
                examples.append({"id": row.get("id"), "target_key": key, "issues": issues})
        else:
            valid_action_targets += 1

    return {
        "num_rows": len(rows),
        "split_counts": dict(split_counts),
        "episode_split_counts": {split: len(episodes) for split, episodes in sorted(episodes_by_split.items())},
        "rows_with_video": rows_with_video,
        "missing_json_answer": missing_json_answer,
        "missing_json_fields": dict(missing_json_fields),
        "rows_with_action_target": rows_with_action_target,
        "valid_action_targets": valid_action_targets,
        "target_key_counts": dict(target_key_counts),
        "target_mode_counts": dict(target_mode_counts),
        "target_issue_counts": dict(target_issue_counts),
        "target_issue_examples": examples,
    }


def decide(dataset: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []

    if dataset["num_rows"] <= 0:
        blockers.append("dataset has zero rows")
    if dataset["rows_with_video"] <= 0:
        blockers.append("dataset has no video conditioning paths")
    if dataset["missing_json_answer"] or dataset["missing_json_fields"]:
        warnings.append("dataset is not a complete JSON QA export")

    if model.get("provided"):
        if not model.get("exists"):
            blockers.append(f"model_dir does not exist: {model.get('path')}")
        if model.get("model_type") != "cosmos3_omni":
            warnings.append(f"model_type is not cosmos3_omni: {model.get('model_type')}")
        if model.get("action_gen") is not True:
            blockers.append("Cosmos3 transformer config does not advertise action_gen=True")
        if not model.get("action_dim"):
            blockers.append("Cosmos3 transformer config does not expose action_dim")
    else:
        warnings.append("model_dir not provided; model action_gen/action_dim could not be verified")

    if dataset["rows_with_action_target"] <= 0:
        blockers.append(
            "dataset has no cosmos_action_target/cosmos3_action_target/action_target records; "
            "semantic JSON labels cannot be used as Cosmos continuous action latents"
        )
    elif dataset["valid_action_targets"] != dataset["rows_with_action_target"]:
        blockers.append(
            "one or more action target records do not satisfy the CosmosActionCondition schema"
        )

    status = "ready_for_cosmos3_super_action_lora" if not blockers else "blocked_missing_cosmos_action_targets"
    if not blockers and dataset.get("target_mode_counts") == {"forward_dynamics": dataset["rows_with_action_target"]}:
        status = "ready_for_cosmos3_super_forward_dynamics_lora"
    return {
        "status": status,
        "weights_updated": False,
        "blockers": blockers,
        "warnings": warnings,
        "required_target_schema": REQUIRED_SCHEMA,
        "trainer_contract": {
            "diffusers_classes": [
                "Cosmos3OmniPipeline",
                "Cosmos3OmniTransformer",
                "CosmosActionCondition",
            ],
            "packing_helpers": [
                "Cosmos3OmniPipeline.prepare_latents",
                "Cosmos3OmniPipeline._prepare_text_segment",
                "Cosmos3OmniPipeline._prepare_vision_segment",
                "Cosmos3OmniPipeline._prepare_action_segment",
            ],
            "forward_outputs": "Cosmos3OmniTransformer.forward returns (preds_vision, preds_sound, preds_action). The current camera_pose forward_dynamics target uses raw actions as conditioning and should supervise preds_vision; supervised preds_action needs policy or inverse_dynamics targets.",
            "lora_targets": "use checkpoint-declared q_proj_moe_gen,k_proj_moe_gen,v_proj_moe_gen,o_proj_moe_gen unless a new audited config overrides them",
        },
        "next_steps": [
            "Run the one-sample action batch packer that calls Cosmos3OmniPipeline.prepare_latents and the static segment helpers, then records whether the current target supervises vision or action tokens.",
            "For the current camera_pose forward_dynamics target, implement a one-sample overfit with vision velocity/rectified-flow loss under action conditioning; add a policy/inverse target export before claiming supervised action-token prediction.",
            "Run a one-episode overfit before scheduling a 96/16/16 Super LoRA run; only publish a Cosmos model repo after new adapter/checkpoint weights exist.",
        ],
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    decision = payload["decision"]
    lines = [
        "# Cosmos3-Super Training Contract Audit",
        "",
        f"- Run id: `{payload['run_id']}`",
        f"- Dataset: `{payload['dataset_jsonl']}`",
        f"- Rows: `{payload['dataset']['num_rows']}`",
        f"- Rows with Cosmos action targets: `{payload['dataset']['rows_with_action_target']}`",
        f"- Valid Cosmos action targets: `{payload['dataset']['valid_action_targets']}`",
        f"- Status: `{decision['status']}`",
        f"- Weights updated: `{decision['weights_updated']}`",
        "",
        "## Blockers",
        "",
    ]
    if decision["blockers"]:
        lines.extend(f"- {item}" for item in decision["blockers"])
    else:
        lines.append("- None")
    lines.extend(["", "## Required Target Schema", "", "```json", json.dumps(REQUIRED_SCHEMA, indent=2), "```", ""])
    lines.extend(["## Next Steps", ""])
    lines.extend(f"- {item}" for item in decision["next_steps"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.workspace = args.workspace.expanduser().resolve()
    args.dataset_jsonl = args.dataset_jsonl.expanduser().resolve()
    if args.model_dir is not None:
        args.model_dir = args.model_dir.expanduser().resolve()
    output_dir = args.output_dir or args.workspace / "results" / "omni_finetune" / args.run_id
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"

    started = time.time()
    append_jsonl(progress_path, {"event": "start", "time": started, "run_id": args.run_id})
    rows = load_jsonl(args.dataset_jsonl)
    if args.sample_limit > 0:
        rows = rows[: args.sample_limit]
    append_jsonl(progress_path, {"event": "dataset_loaded", "time": time.time(), "rows": len(rows)})

    dataset = dataset_summary(rows)
    model = model_summary(args.model_dir)
    backbone = read_json(args.backbone_config)
    decision = decide(dataset, model)
    payload = {
        "run_id": args.run_id,
        "run_kind": "cosmos3_super_training_contract_audit",
        "started_at_unix": started,
        "finished_at_unix": time.time(),
        "elapsed_seconds": time.time() - started,
        "workspace": str(args.workspace),
        "dataset_jsonl": str(args.dataset_jsonl),
        "sample_limit": args.sample_limit,
        "backbone_config": str(args.backbone_config),
        "backbone": {
            "id": backbone.get("id"),
            "display_name": backbone.get("display_name"),
            "training_objective": backbone.get("training_objective"),
        },
        "model": model,
        "dataset": dataset,
        "decision": decision,
    }
    write_json(output_dir / "training_contract_audit.json", payload)
    write_json(output_dir / "training_metadata.json", {
        "run_id": args.run_id,
        "run_kind": payload["run_kind"],
        "weights_updated": False,
        "checkpoint_dir": None,
        "decision": decision,
    })
    write_report(output_dir / "RUN_REPORT.md", payload)
    append_jsonl(progress_path, {"event": "complete", "time": time.time(), "status": decision["status"]})
    print(json.dumps({"status": decision["status"], "output_dir": str(output_dir)}, indent=2))
    ready_statuses = {
        "ready_for_cosmos3_super_action_lora",
        "ready_for_cosmos3_super_forward_dynamics_lora",
    }
    return 1 if args.require_trainable and decision["status"] not in ready_statuses else 0


if __name__ == "__main__":
    raise SystemExit(main())

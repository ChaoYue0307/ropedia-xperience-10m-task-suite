#!/usr/bin/env python3
"""Audit Qwen3-Omni/Cosmos3 output readiness for all 20 tasks."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_PREDICTION_HINTS = {
    "qwen3_omni_v6_lora": {
        "train": [],
        "validation": [],
        "test": [
            "results/omni_finetune/verified_public/"
            "xperience10m_qwen3_omni_128ep_multiscale_cap96_v6_rank64_lr5e5_full8gpu_lora_eval_test_full/"
            "eval/predictions.jsonl",
            "results/omni_finetune/verified_public/"
            "xperience10m_qwen3_omni_128ep_multiscale_cap96_v6_rank64_lr5e5_full8gpu_lora_eval_test_full/"
            "eval/model_predictions.jsonl",
        ],
    },
    "cosmos3_super_reasoner": {
        "train": [],
        "validation": [],
        "test": [
            "results/omni_finetune/verified_public/"
            "xperience10m_cosmos3_super_reasoner_128ep_test_full_20260607/"
            "eval/predictions.jsonl",
            "results/omni_finetune/verified_public/"
            "xperience10m_cosmos3_super_reasoner_128ep_test_full_20260607/"
            "eval/model_predictions.jsonl",
        ],
    },
    "cosmos3_nano_future_window": {
        "train": [],
        "validation": [],
        "test": [
            "results/omni_finetune/verified_public/"
            "xperience10m_cosmos3_nano_128ep_future_window_h5_compat_adapter_eval_test_full/"
            "eval/predictions.jsonl",
            "results/omni_finetune/verified_public/"
            "xperience10m_cosmos3_nano_128ep_future_window_h5_compat_adapter_eval_test_full/"
            "eval/model_predictions.jsonl",
        ],
    },
}

REQUIRED_SPLITS = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root containing docs/data and results.",
    )
    parser.add_argument(
        "--matrix-json",
        type=Path,
        default=None,
        help="Task-method result matrix. Defaults to docs/data/task_method_20_result_matrix.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for readiness artifacts. Defaults to results/omni_finetune/model_output_probe_readiness.",
    )
    parser.add_argument(
        "--prediction",
        action="append",
        default=[],
        metavar="METHOD:SPLIT:PATH",
        help="Add a model-output file candidate, for example qwen3_omni_v6_lora:test:predictions.jsonl.",
    )
    return parser.parse_args()


def resolve_default(path: Path | None, workspace: Path, default: str) -> Path:
    return path if path is not None else workspace / default


def load_matrix(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_prediction_overrides(values: list[str]) -> dict[str, dict[str, list[str]]]:
    overrides: dict[str, dict[str, list[str]]] = {}
    for value in values:
        parts = value.split(":", 2)
        if len(parts) != 3:
            raise SystemExit(f"invalid --prediction value: {value}")
        method, split, path = parts
        if split not in REQUIRED_SPLITS:
            raise SystemExit(f"invalid split in --prediction value: {value}")
        overrides.setdefault(method, {name: [] for name in REQUIRED_SPLITS})[split].append(path)
    return overrides


def first_existing(workspace: Path, candidates: list[str]) -> dict:
    checked = []
    for candidate in candidates:
        path = Path(candidate)
        resolved = path if path.is_absolute() else workspace / path
        display_path = (
            resolved.relative_to(workspace).as_posix()
            if resolved.is_relative_to(workspace)
            else resolved.as_posix()
        )
        checked.append(display_path)
        if resolved.exists():
            return {
                "exists": True,
                "path": display_path,
                "bytes": resolved.stat().st_size,
                "checked": checked,
            }
    return {"exists": False, "path": None, "bytes": 0, "checked": checked}


def records_for_method(matrix: dict, method_id: str) -> list[dict]:
    return [row for row in matrix["records"] if row["series_id"] == method_id]


def build_readiness(workspace: Path, matrix: dict, overrides: dict[str, dict[str, list[str]]]) -> dict:
    methods = {}
    source_hints = DEFAULT_PREDICTION_HINTS.copy()
    matrix_complete = matrix.get("scored_method_task_count") == matrix.get("method_task_record_count")
    for method, split_map in overrides.items():
        target = source_hints.setdefault(method, {name: [] for name in REQUIRED_SPLITS})
        for split, paths in split_map.items():
            target.setdefault(split, []).extend(paths)

    for method_id, split_hints in sorted(source_hints.items()):
        split_status = {
            split: first_existing(workspace, split_hints.get(split, []))
            for split in REQUIRED_SPLITS
        }
        method_records = records_for_method(matrix, method_id)
        scored = [row for row in method_records if row.get("scored")]
        missing = [row for row in method_records if not row.get("scored")]
        ready_for_all_task_probe = all(split_status[split]["exists"] for split in REQUIRED_SPLITS)
        if matrix_complete and not missing:
            method_status = "superseded_by_completed_matrix"
            next_step = "No gap-filling action is required for the current 20-task matrix; use this script only for future replacement artifacts."
        else:
            method_status = "ready" if ready_for_all_task_probe else "missing_required_model_outputs"
            next_step = (
                "Run the all-task probe scorer against train/validation/test outputs."
                if ready_for_all_task_probe
                else "Collect or generate train, validation, and test prediction JSONL files first."
            )
        methods[method_id] = {
            "label": next((series["label"] for series in matrix["series"] if series["id"] == method_id), method_id),
            "matrix_scored_task_count": len(scored),
            "matrix_scoreless_task_count": len(missing),
            "required_splits": list(REQUIRED_SPLITS),
            "split_status": split_status,
            "ready_for_all_task_probe": ready_for_all_task_probe,
            "status": method_status,
            "scoreless_task_ids": [row["task_id"] for row in missing],
            "next_step": next_step,
        }

    ready_methods = [method for method, item in methods.items() if item["ready_for_all_task_probe"]]
    completion_state = "completed_matrix" if matrix_complete else "readiness_check"
    return {
        "title": "Model Output Probe Readiness",
        "status": "pass",
        "completion_state": completion_state,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_matrix": "docs/data/task_method_20_result_matrix.json",
        "scope": (
            "The current matrix is already complete. This artifact is retained as a "
            "guardrail for future replacement model-output probes and does not create "
            "or infer numeric scores."
            if matrix_complete
            else "This artifact checks readiness for extending verified Qwen3-Omni/Cosmos3 runs "
            "to all 20 task contracts. It does not create or infer numeric scores."
        ),
        "score_policy": (
            "The current matrix has zero scoreless cells. Future replacement scores "
            "must still come from task-specific held-out artifacts."
            if matrix_complete
            else "A scoreless Qwen3-Omni/Cosmos3 cell can become numeric only after the run "
            "emits the task target and the metric is computed against held-out labels."
        ),
        "ready_method_count": len(ready_methods),
        "methods": methods,
    }


def write_report(output_dir: Path, payload: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "model_output_probe_readiness.json"
    md_path = output_dir / "RUN_REPORT.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    rows = []
    for method_id, method in payload["methods"].items():
        split_bits = []
        for split, status in method["split_status"].items():
            split_bits.append(f"{split}: {'present' if status['exists'] else 'missing'}")
        rows.append(
            "| "
            + " | ".join(
                [
                    method["label"],
                    method_id,
                    f"{method['matrix_scored_task_count']}/20",
                    method["status"],
                    "; ".join(split_bits),
                    method["next_step"],
                ]
            )
            + " |"
        )

    intro = (
        "The 20-task matrix is already complete, so this readiness report is "
        "superseded for the current release. It remains a guardrail for future "
        "replacement model-output probes and does not assign new task scores."
        if payload.get("completion_state") == "completed_matrix"
        else "This report checks whether verified Qwen3-Omni/Cosmos3 runs have the prediction files\n"
        "needed to extend them to every 20-task contract. It is readiness evidence only;\n"
        "it does not assign new task scores."
    )

    report = f"""# Model Output Probe Readiness

Generated: `{payload['generated_at_utc']}`

{intro}

| Method | ID | Matrix scores | Status | Split files | Next step |
| --- | --- | --- | --- | --- | --- |
{chr(10).join(rows)}
"""
    md_path.write_text(report, encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


def main() -> None:
    args = parse_args()
    workspace = args.workspace.resolve()
    matrix_path = resolve_default(
        args.matrix_json, workspace, "docs/data/task_method_20_result_matrix.json"
    )
    output_dir = resolve_default(
        args.output_dir, workspace, "results/omni_finetune/model_output_probe_readiness"
    )
    overrides = parse_prediction_overrides(args.prediction)
    payload = build_readiness(workspace, load_matrix(matrix_path), overrides)
    write_report(output_dir, payload)


if __name__ == "__main__":
    main()

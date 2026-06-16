#!/usr/bin/env python3
"""Score task probes that are already present in verified model outputs.

This script does not run new model inference. It only derives task-specific
scores from committed held-out prediction JSONL files when the required target
and prediction fields are both already present.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qwen3_omni_dataset_utils import class_metrics


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "results/omni_finetune/model_output_task_probes_20260616"
MISSING_PRED_RELATION = "<missing_pred_relation>"

MODEL_SPECS = {
    "qwen3_omni_v6_lora": {
        "label": "Qwen3-Omni v6 LoRA",
        "prediction_jsonl": (
            "results/omni_finetune/verified_public/"
            "xperience10m_qwen3_omni_128ep_multiscale_cap96_v6_rank64_lr5e5_full8gpu_lora_eval_test_full/"
            "eval/predictions.jsonl"
        ),
    },
    "cosmos3_super_reasoner": {
        "label": "Cosmos3-Super Reasoner",
        "prediction_jsonl": (
            "results/omni_finetune/verified_public/"
            "xperience10m_cosmos3_super_reasoner_128ep_test_full_20260607/"
            "eval/predictions.jsonl"
        ),
    },
    "cosmos3_nano_future_window": {
        "label": "Cosmos3-Nano Future Window",
        "prediction_jsonl": (
            "results/omni_finetune/verified_public/"
            "xperience10m_cosmos3_nano_128ep_future_window_h5_compat_adapter_eval_test_full/"
            "eval/future_predictions.jsonl"
        ),
        "unsupported_reason": "verified future-window predictions do not contain object-set fields",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=ROOT,
        help="Repository root. Defaults to the current checkout.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for derived task-probe artifacts.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line_number, line in enumerate(fp, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def relpath(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def normalize_text(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    return text.strip("`'\". ")


def normalize_objects(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    seen: set[str] = set()
    objects: list[str] = []
    for item in value:
        obj = normalize_text(item).casefold()
        if not obj or obj in seen:
            continue
        seen.add(obj)
        objects.append(obj)
    return sorted(objects)


def relation_label(action: Any, objects: Any, *, missing_label: str | None = None) -> str | None:
    action_text = normalize_text(action)
    object_set = normalize_objects(objects)
    if not action_text or not object_set:
        return missing_label
    return f"{action_text} :: {' | '.join(object_set)}"


def score_action_object_relation(
    *,
    model_id: str,
    spec: dict[str, Any],
    prediction_jsonl: Path,
    output_dir: Path,
    workspace: Path,
) -> dict[str, Any]:
    rows = read_jsonl(prediction_jsonl)
    scored_rows: list[dict[str, Any]] = []
    y_true: list[str] = []
    y_pred: list[str] = []
    valid_pred_count = 0
    missing_pred_count = 0

    for row in rows:
        true_json = row.get("true_json") if isinstance(row.get("true_json"), dict) else {}
        pred_json = row.get("pred_json") if isinstance(row.get("pred_json"), dict) else {}
        true_action = row.get("true_label") or true_json.get("action")
        pred_action = row.get("predicted_label") or pred_json.get("action")
        true_relation = relation_label(true_action, true_json.get("objects"))
        if true_relation is None:
            continue
        pred_relation = relation_label(
            pred_action,
            pred_json.get("objects"),
            missing_label=MISSING_PRED_RELATION,
        )
        if pred_relation == MISSING_PRED_RELATION:
            missing_pred_count += 1
        else:
            valid_pred_count += 1
        y_true.append(true_relation)
        y_pred.append(pred_relation or MISSING_PRED_RELATION)
        scored_rows.append(
            {
                "id": row.get("id"),
                "split": row.get("split"),
                "episode_id": row.get("episode_id"),
                "center_window": json.dumps(row.get("center_window"), sort_keys=True),
                "true_action": true_action,
                "pred_action": pred_action,
                "true_objects": json.dumps(normalize_objects(true_json.get("objects")), ensure_ascii=False),
                "pred_objects": json.dumps(normalize_objects(pred_json.get("objects")), ensure_ascii=False),
                "true_relation": true_relation,
                "pred_relation": pred_relation,
                "correct": int(true_relation == pred_relation),
            }
        )

    if not y_true:
        raise RuntimeError(f"no action-object relation targets found in {prediction_jsonl}")

    label_options = sorted(set(y_true))
    metrics, per_class, _ = class_metrics(y_true, y_pred, label_options)
    metrics.update(
        {
            "title": f"{spec['label']} Action-Object Relation Probe",
            "status": "pass",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model_id": model_id,
            "model_label": spec["label"],
            "task_id": "action_object_relation",
            "task_number": 16,
            "task_label": "Action-Object Relation",
            "metric_key": "action_object_relation_macro_f1",
            "primary_metric": "action_object_relation_macro_f1",
            "primary_score": metrics["macro_f1"],
            "action_object_relation_macro_f1": metrics["macro_f1"],
            "action_object_relation_accuracy": metrics["accuracy"],
            "source_prediction_jsonl": relpath(prediction_jsonl, workspace),
            "scope": "held_out_test_existing_verified_prediction_json",
            "score_policy": (
                "Derived from existing verified held-out prediction JSON. No new model "
                "inference was run; rows without a predicted action/object relation are "
                "counted as missing predictions."
            ),
            "normalization_policy": (
                "The action component uses the verified predicted action label when "
                "present. The object component is a canonical casefolded set because "
                "task 16 is an action plus object-set relation."
            ),
            "total_prediction_rows": len(rows),
            "scored_rows": len(scored_rows),
            "excluded_rows_without_true_relation": len(rows) - len(scored_rows),
            "valid_pred_relation_count": valid_pred_count,
            "missing_pred_relation_count": missing_pred_count,
            "valid_pred_relation_rate": valid_pred_count / len(scored_rows),
            "artifact_files": {
                "metrics_json": relpath(output_dir / "metrics.json", workspace),
                "predictions_csv": relpath(output_dir / "predictions.csv", workspace),
                "per_class_metrics_csv": relpath(output_dir / "per_class_metrics.csv", workspace),
            },
        }
    )

    write_json(output_dir / "metrics.json", metrics)
    write_csv(
        output_dir / "predictions.csv",
        scored_rows,
        [
            "id",
            "split",
            "episode_id",
            "center_window",
            "true_action",
            "pred_action",
            "true_objects",
            "pred_objects",
            "true_relation",
            "pred_relation",
            "correct",
        ],
    )
    write_csv(
        output_dir / "per_class_metrics.csv",
        per_class,
        ["class_name", "support", "predicted", "precision", "recall", "f1"],
    )
    return metrics


def build_report(summary: dict[str, Any]) -> str:
    rows = []
    for model_id, result in summary["methods"].items():
        rows.append(
            "| "
            + " | ".join(
                [
                    result["label"],
                    model_id,
                    result["status"],
                    str(result.get("scored_rows", "n/a")),
                    (
                        f"{result.get('action_object_relation_macro_f1', 0.0):.6f}"
                        if result.get("action_object_relation_macro_f1") is not None
                        else "n/a"
                    ),
                    result.get("reason") or result.get("source_prediction_jsonl", ""),
                ]
            )
            + " |"
        )
    return f"""# Existing Model-Output Task Probes

Generated: `{summary['generated_at_utc']}`

This package scores only task targets already present in verified held-out
prediction JSON. It does not run new inference and does not infer targets that
are absent from a model branch.

| Method | ID | Status | Scored rows | Task 16 macro-F1 | Evidence |
| --- | --- | --- | ---: | ---: | --- |
{chr(10).join(rows)}
"""


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else workspace / args.output_dir
    task_dir = output_dir / "action_object_relation"
    methods: dict[str, Any] = {}

    for model_id, spec in MODEL_SPECS.items():
        prediction_path = workspace / spec["prediction_jsonl"]
        if spec.get("unsupported_reason"):
            methods[model_id] = {
                "label": spec["label"],
                "status": "unsupported_without_required_fields",
                "source_prediction_jsonl": spec["prediction_jsonl"],
                "reason": spec["unsupported_reason"],
            }
            continue
        if not prediction_path.exists():
            methods[model_id] = {
                "label": spec["label"],
                "status": "missing_prediction_jsonl",
                "source_prediction_jsonl": spec["prediction_jsonl"],
                "reason": "verified prediction JSONL was not found in the local checkout",
            }
            continue
        metrics = score_action_object_relation(
            model_id=model_id,
            spec=spec,
            prediction_jsonl=prediction_path,
            output_dir=task_dir / model_id,
            workspace=workspace,
        )
        methods[model_id] = {
            "label": spec["label"],
            "status": "scored",
            "source_prediction_jsonl": metrics["source_prediction_jsonl"],
            "source_metrics_json": metrics["artifact_files"]["metrics_json"],
            "scored_rows": metrics["scored_rows"],
            "valid_pred_relation_rate": metrics["valid_pred_relation_rate"],
            "action_object_relation_macro_f1": metrics["action_object_relation_macro_f1"],
            "action_object_relation_accuracy": metrics["action_object_relation_accuracy"],
        }

    summary = {
        "title": "Existing Model-Output Task Probes",
        "status": "pass",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scope": (
            "Task-specific scoring from existing verified held-out model outputs. "
            "No new model inference, training, or target backfilling is performed."
        ),
        "task_count_added_to_matrix": 1,
        "scored_method_task_count_added": sum(1 for item in methods.values() if item["status"] == "scored"),
        "methods": methods,
    }
    write_json(output_dir / "summary.json", summary)
    (output_dir / "RUN_REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(f"wrote {relpath(output_dir / 'summary.json', workspace)}")
    print(f"wrote {relpath(output_dir / 'RUN_REPORT.md', workspace)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

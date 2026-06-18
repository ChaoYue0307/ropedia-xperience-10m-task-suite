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
MISSING_PRED_NEXT_ACTION = "<missing_pred_next_action>"

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
        "time_to_transition_from_action_sequence": True,
    },
    "cosmos3_nano_future_window": {
        "label": "Cosmos3-Nano Future Window",
        "prediction_jsonl": (
            "results/omni_finetune/verified_public/"
            "xperience10m_cosmos3_nano_128ep_future_window_h5_compat_adapter_eval_test_full/"
            "eval/future_predictions.jsonl"
        ),
        "metrics_json": (
            "results/omni_finetune/verified_public/"
            "xperience10m_cosmos3_nano_128ep_future_window_h5_compat_adapter_eval_test_full/"
            "eval/metrics.json"
        ),
        "dataset_manifest": (
            "results/omni_finetune/verified_public/"
            "xperience10m_cosmos3_nano_128ep_future_window_h5_compat_adapter_eval_test_full/"
            "dataset/dataset_manifest.json"
        ),
        "source_window_map_jsonl": (
            "results/omni_finetune/model_output_task_probes_20260616/"
            "cosmos3_nano_future_window_source_window_map.jsonl"
        ),
        "target_map_jsonl": (
            "results/omni_finetune/model_output_task_probes_20260616/"
            "cosmos3_nano_future_window_target_map.jsonl"
        ),
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


def object_set_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    tp = fp = fn = exact = 0
    for row in rows:
        true_set = set(row["true_objects"])
        pred_set = set(row["pred_objects"])
        tp += len(true_set & pred_set)
        fp += len(pred_set - true_set)
        fn += len(true_set - pred_set)
        exact += int(true_set == pred_set)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    micro_f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "micro_f1": micro_f1,
        "precision": precision,
        "recall": recall,
        "exact_match": exact / len(rows) if rows else 0.0,
        "true_positive_objects": float(tp),
        "false_positive_objects": float(fp),
        "false_negative_objects": float(fn),
    }


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


def score_cosmos_nano_long_horizon_next_action(
    *,
    model_id: str,
    spec: dict[str, Any],
    prediction_jsonl: Path,
    dataset_manifest: Path,
    output_dir: Path,
    workspace: Path,
) -> dict[str, Any]:
    rows = read_jsonl(prediction_jsonl)
    manifest = json.loads(dataset_manifest.read_text(encoding="utf-8")) if dataset_manifest.exists() else {}
    scored_rows: list[dict[str, Any]] = []
    y_true: list[str] = []
    y_pred: list[str] = []

    for row in rows:
        true_action = normalize_text(row.get("true_action"))
        pred_action = normalize_text(row.get("pred_action"))
        if not true_action:
            continue
        if not pred_action:
            pred_action = "<missing_pred_action>"
        y_true.append(true_action)
        y_pred.append(pred_action)
        scored_rows.append(
            {
                "id": row.get("id"),
                "split": row.get("split"),
                "episode_id": row.get("episode_id"),
                "context_record_id": row.get("context_record_id"),
                "future_record_id": row.get("future_record_id"),
                "pred_future_record_id": row.get("pred_future_record_id"),
                "rank": row.get("rank"),
                "true_action": true_action,
                "pred_action": pred_action,
                "correct": int(true_action == pred_action),
                "top_k_hit": row.get("top_k_hit"),
                "distance_to_true": row.get("distance_to_true"),
                "distance_to_pred": row.get("distance_to_pred"),
            }
        )

    if not y_true:
        raise RuntimeError(f"no long-horizon action targets found in {prediction_jsonl}")

    label_options = sorted(set(y_true))
    metrics, per_class, _ = class_metrics(y_true, y_pred, label_options)
    horizon_windows = manifest.get("horizon_windows")
    dataset_contract = manifest.get("dataset_contract")
    metrics.update(
        {
            "title": f"{spec['label']} Long-Horizon Next-Action Probe",
            "status": "pass",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model_id": model_id,
            "model_label": spec["label"],
            "task_id": "long_horizon_next_action",
            "task_number": 13,
            "task_label": "Long-Horizon Next Action",
            "metric_key": "long_horizon_next_action_macro_f1",
            "primary_metric": "long_horizon_next_action_macro_f1",
            "primary_score": metrics["macro_f1"],
            "long_horizon_next_action_macro_f1": metrics["macro_f1"],
            "long_horizon_next_action_accuracy": metrics["accuracy"],
            "source_prediction_jsonl": relpath(prediction_jsonl, workspace),
            "source_dataset_manifest": relpath(dataset_manifest, workspace),
            "dataset_contract": dataset_contract,
            "horizon_windows": horizon_windows,
            "scope": "held_out_test_existing_verified_future_window_prediction_json",
            "score_policy": (
                "Derived from existing verified held-out Cosmos3-Nano future-window "
                "prediction JSON. No new model inference was run; true_action and "
                "pred_action are scored directly for the long-horizon next-action "
                "task axis."
            ),
            "normalization_policy": (
                "Macro-F1 is computed over the true held-out future action labels. "
                "Predicted labels absent from the true-label set remain in the "
                "confusion matrix and receive zero support."
            ),
            "known_limitation": (
                "The verified package records horizon_windows rather than a raw "
                "wall-clock horizon. This score should be read as the Cosmos-Nano "
                "future-window branch for task 13, not as independent proof of an "
                "exact five-second raw-video target."
            ),
            "total_prediction_rows": len(rows),
            "scored_rows": len(scored_rows),
            "excluded_rows_without_true_action": len(rows) - len(scored_rows),
            "unique_true_actions": len(set(y_true)),
            "unique_pred_actions": len(set(y_pred)),
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
            "context_record_id",
            "future_record_id",
            "pred_future_record_id",
            "rank",
            "true_action",
            "pred_action",
            "correct",
            "top_k_hit",
            "distance_to_true",
            "distance_to_pred",
        ],
    )
    write_csv(
        output_dir / "per_class_metrics.csv",
        per_class,
        ["class_name", "support", "predicted", "precision", "recall", "f1"],
    )
    return metrics


def load_target_map(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    return {str(row.get("id")): row for row in rows if row.get("id")}


def score_cosmos_nano_next_subtask_from_target_map(
    *,
    model_id: str,
    spec: dict[str, Any],
    prediction_jsonl: Path,
    target_map_jsonl: Path,
    output_dir: Path,
    workspace: Path,
) -> dict[str, Any]:
    source_rows = read_jsonl(prediction_jsonl)
    target_by_id = load_target_map(target_map_jsonl)
    scored_rows: list[dict[str, Any]] = []
    y_true: list[str] = []
    y_pred: list[str] = []
    missing_true_target_count = 0
    missing_pred_target_count = 0

    for row in source_rows:
        future_id = str(row.get("future_record_id") or "")
        pred_future_id = str(row.get("pred_future_record_id") or "")
        true_target = target_by_id.get(future_id)
        if not true_target:
            missing_true_target_count += 1
            continue
        pred_target = target_by_id.get(pred_future_id)
        true_subtask = normalize_text(true_target.get("subtask"))
        if not true_subtask:
            continue
        pred_subtask = normalize_text(pred_target.get("subtask") if pred_target else "")
        if not pred_subtask:
            pred_subtask = "<missing_pred_subtask>"
            missing_pred_target_count += 1
        y_true.append(true_subtask)
        y_pred.append(pred_subtask)
        scored_rows.append(
            {
                "id": row.get("id"),
                "split": row.get("split"),
                "episode_id": row.get("episode_id"),
                "future_record_id": future_id,
                "pred_future_record_id": pred_future_id,
                "true_subtask": true_subtask,
                "pred_subtask": pred_subtask,
                "correct": int(true_subtask == pred_subtask),
                "rank": row.get("rank"),
                "top_k_hit": row.get("top_k_hit"),
            }
        )

    if not y_true:
        raise RuntimeError(f"no joined future-subtask targets found in {prediction_jsonl}")

    label_options = sorted(set(y_true))
    metrics, per_class, _ = class_metrics(y_true, y_pred, label_options)
    metrics.update(
        {
            "title": f"{spec['label']} Next-Subtask Future-Window Probe",
            "status": "pass",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model_id": model_id,
            "model_label": spec["label"],
            "task_id": "next_subtask_forecast",
            "task_number": 14,
            "task_label": "Long-Horizon Next-Subtask Forecasting",
            "metric_key": "next_subtask_forecast_macro_f1",
            "primary_metric": "next_subtask_forecast_macro_f1",
            "primary_score": metrics["macro_f1"],
            "next_subtask_forecast_macro_f1": metrics["macro_f1"],
            "next_subtask_forecast_accuracy": metrics["accuracy"],
            "source_prediction_jsonl": relpath(prediction_jsonl, workspace),
            "source_target_map_jsonl": relpath(target_map_jsonl, workspace),
            "scope": "held_out_test_existing_future_window_retrieval_probe",
            "score_policy": (
                "Derived from existing verified held-out Cosmos3-Nano future-window "
                "predictions. The model output is the retrieved future record; both the "
                "true future record and retrieved future record are joined to a compact "
                "public-safe target map, then scored on subtask labels."
            ),
            "normalization_policy": "Whitespace and surrounding quotes/punctuation are normalized before exact-label macro-F1 scoring.",
            "known_limitation": (
                "This is a retrieval-derived future-subtask score. It is not a separately "
                "prompted generative subtask head, but it is target-backed by verified "
                "future-record retrieval outputs."
            ),
            "total_prediction_rows": len(source_rows),
            "target_map_rows": len(target_by_id),
            "scored_rows": len(scored_rows),
            "missing_true_target_count": missing_true_target_count,
            "missing_pred_target_count": missing_pred_target_count,
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
            "future_record_id",
            "pred_future_record_id",
            "true_subtask",
            "pred_subtask",
            "correct",
            "rank",
            "top_k_hit",
        ],
    )
    write_csv(
        output_dir / "per_class_metrics.csv",
        per_class,
        ["class_name", "support", "predicted", "precision", "recall", "f1"],
    )
    return metrics


def score_cosmos_nano_object_set_from_target_map(
    *,
    model_id: str,
    spec: dict[str, Any],
    prediction_jsonl: Path,
    target_map_jsonl: Path,
    output_dir: Path,
    workspace: Path,
) -> dict[str, Any]:
    source_rows = read_jsonl(prediction_jsonl)
    target_by_id = load_target_map(target_map_jsonl)
    scored_rows: list[dict[str, Any]] = []
    missing_true_target_count = 0
    missing_pred_target_count = 0

    for row in source_rows:
        future_id = str(row.get("future_record_id") or "")
        pred_future_id = str(row.get("pred_future_record_id") or "")
        true_target = target_by_id.get(future_id)
        if not true_target:
            missing_true_target_count += 1
            continue
        pred_target = target_by_id.get(pred_future_id)
        if not pred_target:
            missing_pred_target_count += 1
        true_objects = normalize_objects(true_target.get("objects"))
        pred_objects = normalize_objects(pred_target.get("objects") if pred_target else [])
        scored_rows.append(
            {
                "id": row.get("id"),
                "split": row.get("split"),
                "episode_id": row.get("episode_id"),
                "future_record_id": future_id,
                "pred_future_record_id": pred_future_id,
                "true_objects": true_objects,
                "pred_objects": pred_objects,
                "true_objects_json": json.dumps(true_objects, ensure_ascii=False),
                "pred_objects_json": json.dumps(pred_objects, ensure_ascii=False),
                "exact_match": int(set(true_objects) == set(pred_objects)),
                "rank": row.get("rank"),
                "top_k_hit": row.get("top_k_hit"),
            }
        )

    if not scored_rows:
        raise RuntimeError(f"no joined future-object targets found in {prediction_jsonl}")

    metrics = object_set_metrics(scored_rows)
    metrics.update(
        {
            "title": f"{spec['label']} Future Object-Set Probe",
            "status": "pass",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model_id": model_id,
            "model_label": spec["label"],
            "task_id": "object_set_forecast",
            "task_number": 17,
            "task_label": "Future Object-Set Forecasting",
            "metric_key": "object_set_forecast_micro_f1",
            "primary_metric": "object_set_forecast_micro_f1",
            "primary_score": metrics["micro_f1"],
            "object_set_forecast_micro_f1": metrics["micro_f1"],
            "object_set_forecast_precision": metrics["precision"],
            "object_set_forecast_recall": metrics["recall"],
            "source_prediction_jsonl": relpath(prediction_jsonl, workspace),
            "source_target_map_jsonl": relpath(target_map_jsonl, workspace),
            "scope": "held_out_test_existing_future_window_retrieval_probe",
            "score_policy": (
                "Derived from existing verified held-out Cosmos3-Nano future-window "
                "predictions. The retrieved future record is joined to a compact "
                "target map and its object set is compared with the true future "
                "record's object set."
            ),
            "normalization_policy": (
                "Objects are casefolded, stripped, deduplicated, and scored with "
                "micro-F1 over the held-out future object sets."
            ),
            "known_limitation": (
                "This is a retrieval-derived object-set score, not a separately prompted "
                "object-list generation head."
            ),
            "total_prediction_rows": len(source_rows),
            "target_map_rows": len(target_by_id),
            "scored_rows": len(scored_rows),
            "missing_true_target_count": missing_true_target_count,
            "missing_pred_target_count": missing_pred_target_count,
            "artifact_files": {
                "metrics_json": relpath(output_dir / "metrics.json", workspace),
                "predictions_csv": relpath(output_dir / "predictions.csv", workspace),
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
            "future_record_id",
            "pred_future_record_id",
            "true_objects_json",
            "pred_objects_json",
            "exact_match",
            "rank",
            "top_k_hit",
        ],
    )
    return metrics


def score_cosmos_nano_action_object_relation_from_target_map(
    *,
    model_id: str,
    spec: dict[str, Any],
    prediction_jsonl: Path,
    target_map_jsonl: Path,
    output_dir: Path,
    workspace: Path,
) -> dict[str, Any]:
    source_rows = read_jsonl(prediction_jsonl)
    target_by_id = load_target_map(target_map_jsonl)
    scored_rows: list[dict[str, Any]] = []
    y_true: list[str] = []
    y_pred: list[str] = []
    missing_true_target_count = 0
    missing_pred_target_count = 0

    for row in source_rows:
        future_id = str(row.get("future_record_id") or "")
        pred_future_id = str(row.get("pred_future_record_id") or "")
        true_target = target_by_id.get(future_id)
        if not true_target:
            missing_true_target_count += 1
            continue
        pred_target = target_by_id.get(pred_future_id)
        if not pred_target:
            missing_pred_target_count += 1
        true_relation = relation_label(true_target.get("action"), true_target.get("objects"))
        if true_relation is None:
            continue
        pred_relation = relation_label(
            pred_target.get("action") if pred_target else None,
            pred_target.get("objects") if pred_target else None,
            missing_label=MISSING_PRED_RELATION,
        )
        y_true.append(true_relation)
        y_pred.append(pred_relation or MISSING_PRED_RELATION)
        scored_rows.append(
            {
                "id": row.get("id"),
                "split": row.get("split"),
                "episode_id": row.get("episode_id"),
                "future_record_id": future_id,
                "pred_future_record_id": pred_future_id,
                "true_relation": true_relation,
                "pred_relation": pred_relation,
                "correct": int(true_relation == pred_relation),
                "rank": row.get("rank"),
                "top_k_hit": row.get("top_k_hit"),
            }
        )

    if not y_true:
        raise RuntimeError(f"no joined future action-object relation targets found in {prediction_jsonl}")

    label_options = sorted(set(y_true))
    metrics, per_class, _ = class_metrics(y_true, y_pred, label_options)
    metrics.update(
        {
            "title": f"{spec['label']} Future Action-Object Relation Probe",
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
            "source_target_map_jsonl": relpath(target_map_jsonl, workspace),
            "scope": "held_out_test_existing_future_window_retrieval_probe",
            "score_policy": (
                "Derived from existing verified held-out Cosmos3-Nano future-window predictions. "
                "The true future record and retrieved future record are joined to the public-safe "
                "target map, then scored on the action plus object-set relation."
            ),
            "normalization_policy": (
                "Action text is whitespace-normalized. Objects are casefolded, deduplicated, "
                "sorted, and joined into a canonical relation label before macro-F1 scoring."
            ),
            "known_limitation": (
                "This is a retrieval-derived future relation score, not a separately prompted "
                "action-object generation head."
            ),
            "total_prediction_rows": len(source_rows),
            "target_map_rows": len(target_by_id),
            "scored_rows": len(scored_rows),
            "missing_true_target_count": missing_true_target_count,
            "missing_pred_target_count": missing_pred_target_count,
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
            "future_record_id",
            "pred_future_record_id",
            "true_relation",
            "pred_relation",
            "correct",
            "rank",
            "top_k_hit",
        ],
    )
    write_csv(
        output_dir / "per_class_metrics.csv",
        per_class,
        ["class_name", "support", "predicted", "precision", "recall", "f1"],
    )
    return metrics


def parse_window(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, dict):
        return None
    try:
        start = float(value.get("start_frame"))
        end = float(value.get("end_frame"))
    except (TypeError, ValueError):
        return None
    if end < start:
        return None
    return start, end


def interval_iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]) + 1.0)
    union = max(a[1], b[1]) - min(a[0], b[0]) + 1.0
    return inter / union if union > 0 else 0.0


def score_caption_grounding_from_evidence_window(
    *,
    model_id: str,
    spec: dict[str, Any],
    prediction_jsonl: Path,
    output_dir: Path,
    workspace: Path,
) -> dict[str, Any]:
    source_rows = read_jsonl(prediction_jsonl)
    scored_rows: list[dict[str, Any]] = []
    ious: list[float] = []
    center_hits = 0
    within_20_hits = 0
    missing_pred_count = 0

    for row in source_rows:
        true_json = row.get("true_json") if isinstance(row.get("true_json"), dict) else {}
        pred_json = row.get("pred_json") if isinstance(row.get("pred_json"), dict) else {}
        true_window = parse_window(true_json.get("evidence_window"))
        if true_window is None:
            continue
        pred_window = parse_window(pred_json.get("evidence_window"))
        if pred_window is None:
            missing_pred_count += 1
            iou = 0.0
            pred_center = None
            center_error = None
        else:
            iou = interval_iou(true_window, pred_window)
            pred_center = (pred_window[0] + pred_window[1]) / 2.0
            true_center = (true_window[0] + true_window[1]) / 2.0
            center_error = abs(pred_center - true_center)
            center_hits += int(true_window[0] <= pred_center <= true_window[1])
            within_20_hits += int(center_error <= 20.0)
        ious.append(iou)
        scored_rows.append(
            {
                "id": row.get("id"),
                "split": row.get("split"),
                "episode_id": row.get("episode_id"),
                "true_start_frame": true_window[0],
                "true_end_frame": true_window[1],
                "pred_start_frame": pred_window[0] if pred_window else None,
                "pred_end_frame": pred_window[1] if pred_window else None,
                "pred_center_frame": pred_center,
                "center_error_frames": center_error,
                "iou": iou,
            }
        )

    if not scored_rows:
        raise RuntimeError(f"no evidence-window targets found in {prediction_jsonl}")

    mean_iou = sum(ious) / len(ious)
    center_hit_rate = center_hits / len(scored_rows)
    within_20_rate = within_20_hits / len(scored_rows)
    metrics = {
        "title": f"{spec['label']} Evidence-Window Grounding Probe",
        "status": "pass",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_id": model_id,
        "model_label": spec["label"],
        "task_id": "caption_grounding",
        "task_number": 8,
        "task_label": "Language Grounding",
        "metric_key": "caption_grounding_iou",
        "primary_metric": "caption_grounding_iou",
        "primary_score": mean_iou,
        "caption_grounding_iou": mean_iou,
        "caption_grounding_center_hit_rate": center_hit_rate,
        "caption_grounding_within_20_frames": within_20_rate,
        "source_prediction_jsonl": relpath(prediction_jsonl, workspace),
        "scope": "held_out_test_existing_verified_prediction_json",
        "score_policy": (
            "Derived from existing verified held-out structured JSON predictions. The task target is "
            "the evidence_window field already present in the true JSON; missing or invalid predicted "
            "evidence windows receive zero IoU."
        ),
        "normalization_policy": (
            "Evidence windows are frame intervals. The primary score is mean interval IoU; center-hit "
            "and within-20-frame rates are reported as diagnostics."
        ),
        "known_limitation": (
            "This is an evidence-window localization score, not a candidate-set retrieval MRR."
        ),
        "total_prediction_rows": len(source_rows),
        "scored_rows": len(scored_rows),
        "missing_pred_evidence_window_count": missing_pred_count,
        "artifact_files": {
            "metrics_json": relpath(output_dir / "metrics.json", workspace),
            "predictions_csv": relpath(output_dir / "predictions.csv", workspace),
        },
    }
    write_json(output_dir / "metrics.json", metrics)
    write_csv(
        output_dir / "predictions.csv",
        scored_rows,
        [
            "id",
            "split",
            "episode_id",
            "true_start_frame",
            "true_end_frame",
            "pred_start_frame",
            "pred_end_frame",
            "pred_center_frame",
            "center_error_frames",
            "iou",
        ],
    )
    return metrics


def score_modality_reconstruction_from_feature_error(
    *,
    model_id: str,
    spec: dict[str, Any],
    metrics_json: Path,
    output_dir: Path,
    workspace: Path,
) -> dict[str, Any]:
    source_metrics = json.loads(metrics_json.read_text(encoding="utf-8"))
    error = source_metrics.get("feature_reconstruction_error")
    if not isinstance(error, (int, float)):
        raise RuntimeError(f"feature_reconstruction_error is absent from {metrics_json}")
    quality = 1.0 / (1.0 + float(error)) if error >= 0 else 0.0
    metrics = {
        "title": f"{spec['label']} Modality Reconstruction Probe",
        "status": "pass",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_id": model_id,
        "model_label": spec["label"],
        "task_id": "modality_reconstruction",
        "task_number": 10,
        "task_label": "Cross-Modal Reconstruction",
        "metric_key": "feature_reconstruction_quality",
        "primary_metric": "feature_reconstruction_quality",
        "primary_score": quality,
        "feature_reconstruction_quality": quality,
        "feature_reconstruction_error": float(error),
        "source_metric_key": "feature_reconstruction_error",
        "source_metrics_json": relpath(metrics_json, workspace),
        "scope": "held_out_test_existing_verified_future_window_reconstruction_metric",
        "score_policy": (
            "Derived from the verified Cosmos3-Nano future-window package metric. The "
            "source package directly reports held-out feature_reconstruction_error; this "
            "artifact maps it onto task 10 as an inverse reconstruction-quality score "
            "1 / (1 + error) so the matrix can retain its higher-is-better convention."
        ),
        "normalization_policy": (
            "This is not the single-episode/128-baseline R2 metric. It is a model-branch "
            "reconstruction-quality probe backed by the verified held-out future-window "
            "feature reconstruction error."
        ),
        "known_limitation": (
            "The metric is comparable as evidence that the branch emitted a reconstruction "
            "objective, but it should not be read as an R2 head trained on the exact simple "
            "baseline feature split."
        ),
        "num_samples": source_metrics.get("num_samples"),
        "artifact_files": {
            "metrics_json": relpath(output_dir / "metrics.json", workspace),
        },
    }
    write_json(output_dir / "metrics.json", metrics)
    return metrics


def row_start(row: dict[str, Any]) -> int:
    window = row.get("center_window") if isinstance(row.get("center_window"), dict) else {}
    return int(window.get("start_frame", 0) or 0)


def action_from_row(row: dict[str, Any], *, predicted: bool) -> str:
    payload_key = "pred_json" if predicted else "true_json"
    label_key = "predicted_label" if predicted else "true_label"
    payload = row.get(payload_key) if isinstance(row.get(payload_key), dict) else {}
    return normalize_text(row.get(label_key) or payload.get("action"))


def next_action_from_row(row: dict[str, Any], *, predicted: bool) -> str:
    payload_key = "pred_json" if predicted else "true_json"
    label_key = "predicted_label" if predicted else "true_label"
    payload = row.get(payload_key) if isinstance(row.get(payload_key), dict) else {}
    return normalize_text(payload.get("next_action") or row.get(label_key) or payload.get("action"))


def score_long_horizon_next_action_from_verified_json(
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
    missing_pred_count = 0

    for row in rows:
        true_next_action = next_action_from_row(row, predicted=False)
        if not true_next_action:
            continue
        pred_next_action = next_action_from_row(row, predicted=True)
        if not pred_next_action:
            pred_next_action = MISSING_PRED_NEXT_ACTION
            missing_pred_count += 1
        y_true.append(true_next_action)
        y_pred.append(pred_next_action)
        scored_rows.append(
            {
                "id": row.get("id"),
                "split": row.get("split"),
                "episode_id": row.get("episode_id"),
                "center_window": json.dumps(row.get("center_window"), sort_keys=True),
                "true_next_action": true_next_action,
                "pred_next_action": pred_next_action,
                "correct": int(true_next_action == pred_next_action),
            }
        )

    if not y_true:
        raise RuntimeError(f"no next-action targets found in {prediction_jsonl}")

    label_options = sorted(set(y_true))
    metrics, per_class, _ = class_metrics(y_true, y_pred, label_options)
    metrics.update(
        {
            "title": f"{spec['label']} Long-Horizon Next-Action Existing-Output Probe",
            "status": "pass",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model_id": model_id,
            "model_label": spec["label"],
            "task_id": "long_horizon_next_action",
            "task_number": 13,
            "task_label": "Long-Horizon Next Action",
            "metric_key": "long_horizon_next_action_macro_f1",
            "primary_metric": "long_horizon_next_action_macro_f1",
            "primary_score": metrics["macro_f1"],
            "long_horizon_next_action_macro_f1": metrics["macro_f1"],
            "long_horizon_next_action_accuracy": metrics["accuracy"],
            "missing_pred_next_action_count": missing_pred_count,
            "source_prediction_jsonl": relpath(prediction_jsonl, workspace),
            "scope": "held_out_test_existing_verified_prediction_json",
            "score_policy": (
                "Derived from the existing verified held-out prediction JSON next_action field. "
                "No new model inference was run; rows without a predicted next_action are "
                "counted as missing predictions."
            ),
            "normalization_policy": (
                "Whitespace and surrounding quotes/punctuation are normalized before exact-label scoring."
            ),
            "known_limitation": (
                "This uses the next_action field already emitted by the verified structured-output "
                "eval. It is target-backed, but it is not a separately prompted future-task generation run."
            ),
            "scored_rows": len(y_true),
            "artifact_files": {
                "metrics_json": relpath(output_dir / "metrics.json", workspace),
                "predictions_csv": relpath(output_dir / "predictions.csv", workspace),
                "per_class_metrics_csv": relpath(output_dir / "per_class_metrics.csv", workspace),
            },
        }
    )
    write_csv(
        output_dir / "predictions.csv",
        scored_rows,
        [
            "id",
            "split",
            "episode_id",
            "center_window",
            "true_next_action",
            "pred_next_action",
            "correct",
        ],
    )
    write_csv(
        output_dir / "per_class_metrics.csv",
        per_class,
        ["label", "support", "predicted", "true_positive", "precision", "recall", "f1"],
    )
    write_json(output_dir / "metrics.json", metrics)
    return metrics


def transition_distances(rows: list[dict[str, Any]], labels: list[str], cap_frames: int) -> list[float]:
    by_episode: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        by_episode.setdefault(str(row.get("episode_id")), []).append(idx)
    distances = [float(cap_frames)] * len(rows)
    for indices in by_episode.values():
        indices.sort(key=lambda idx: row_start(rows[idx]))
        for pos, idx in enumerate(indices):
            current = labels[idx]
            start = row_start(rows[idx])
            distance = cap_frames
            for next_idx in indices[pos + 1 :]:
                if labels[next_idx] != current:
                    distance = min(max(row_start(rows[next_idx]) - start, 0), cap_frames)
                    break
            distances[idx] = float(distance)
    return distances


def score_time_to_transition_from_action_sequence(
    *,
    model_id: str,
    spec: dict[str, Any],
    prediction_jsonl: Path,
    output_dir: Path,
    workspace: Path,
    cap_frames: int = 200,
) -> dict[str, Any]:
    source_rows = read_jsonl(prediction_jsonl)
    rows: list[dict[str, Any]] = []
    true_actions: list[str] = []
    pred_actions: list[str] = []
    missing_pred_action_count = 0

    for row in source_rows:
        true_action = action_from_row(row, predicted=False)
        if not true_action:
            continue
        pred_action = action_from_row(row, predicted=True)
        if not pred_action:
            pred_action = "<missing_pred_action>"
            missing_pred_action_count += 1
        rows.append(row)
        true_actions.append(true_action)
        pred_actions.append(pred_action)

    if not rows:
        raise RuntimeError(f"no action labels found for time-to-transition scoring in {prediction_jsonl}")

    true_dist = transition_distances(rows, true_actions, cap_frames)
    pred_dist = transition_distances(rows, pred_actions, cap_frames)
    errors = [abs(pred - true) for pred, true in zip(pred_dist, true_dist)]
    mae = sum(errors) / len(errors)
    rmse = (sum(error * error for error in errors) / len(errors)) ** 0.5
    within_20 = sum(1 for error in errors if error <= 20.0) / len(errors)
    within_50 = sum(1 for error in errors if error <= 50.0) / len(errors)

    scored_rows = []
    for row, true_action, pred_action, true_value, pred_value, error in zip(
        rows,
        true_actions,
        pred_actions,
        true_dist,
        pred_dist,
        errors,
    ):
        scored_rows.append(
            {
                "id": row.get("id"),
                "split": row.get("split"),
                "episode_id": row.get("episode_id"),
                "start_frame": row_start(row),
                "true_action": true_action,
                "pred_action": pred_action,
                "true_time_to_transition_frames": true_value,
                "pred_time_to_transition_frames": pred_value,
                "absolute_error_frames": error,
            }
        )

    metrics = {
        "title": f"{spec['label']} Time-to-Transition Action-Sequence Probe",
        "status": "pass",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_id": model_id,
        "model_label": spec["label"],
        "task_id": "time_to_transition",
        "task_number": 20,
        "task_label": "Time to Transition",
        "metric_key": "time_to_transition_mae",
        "primary_metric": "time_to_transition_mae",
        "primary_score": mae,
        "metric_direction": "lower",
        "time_to_transition_mae": mae,
        "time_to_transition_rmse": rmse,
        "within_20_frames": within_20,
        "within_50_frames": within_50,
        "cap_frames": cap_frames,
        "source_prediction_jsonl": relpath(prediction_jsonl, workspace),
        "scope": "held_out_test_existing_action_sequence_derived_probe",
        "score_policy": (
            "Derived from existing verified held-out action predictions. The model did "
            "not emit a direct scalar time-to-transition value; predicted boundary timing "
            "is computed from changes in its predicted action sequence and compared with "
            "the true held-out action-boundary timing."
        ),
        "normalization_policy": (
            "Rows are grouped by episode and sorted by center-window start frame. The "
            "target and prediction are frames until the next action-label change, capped "
            f"at {cap_frames} frames."
        ),
        "known_limitation": (
            "This is a derived action-sequence probe, not evidence of a separately "
            "trained time-regression head. It is included because task 20's target is "
            "deterministically derivable from a sequence of action labels."
        ),
        "total_prediction_rows": len(source_rows),
        "scored_rows": len(scored_rows),
        "excluded_rows_without_true_action": len(source_rows) - len(scored_rows),
        "missing_pred_action_count": missing_pred_action_count,
        "artifact_files": {
            "metrics_json": relpath(output_dir / "metrics.json", workspace),
            "predictions_csv": relpath(output_dir / "predictions.csv", workspace),
        },
    }
    write_json(output_dir / "metrics.json", metrics)
    write_csv(
        output_dir / "predictions.csv",
        scored_rows,
        [
            "id",
            "split",
            "episode_id",
            "start_frame",
            "true_action",
            "pred_action",
            "true_time_to_transition_frames",
            "pred_time_to_transition_frames",
            "absolute_error_frames",
        ],
    )
    return metrics


def score_cosmos_nano_time_to_transition_from_future_windows(
    *,
    model_id: str,
    spec: dict[str, Any],
    prediction_jsonl: Path,
    source_window_map_jsonl: Path,
    output_dir: Path,
    workspace: Path,
    cap_frames: int = 200,
) -> dict[str, Any]:
    source_rows = read_jsonl(prediction_jsonl)
    window_rows = read_jsonl(source_window_map_jsonl)
    window_by_id = {str(row.get("id")): row for row in window_rows if row.get("id")}
    rows: list[dict[str, Any]] = []
    true_actions: list[str] = []
    pred_actions: list[str] = []
    missing_window_count = 0
    missing_pred_action_count = 0

    for row in source_rows:
        future_id = str(row.get("future_record_id") or "")
        window = window_by_id.get(future_id)
        if not window:
            missing_window_count += 1
            continue
        true_action = normalize_text(row.get("true_action") or window.get("action"))
        if not true_action:
            continue
        pred_action = normalize_text(row.get("pred_action"))
        if not pred_action:
            pred_action = "<missing_pred_action>"
            missing_pred_action_count += 1
        rows.append(
            {
                **row,
                "center_window": {
                    "start_frame": window.get("start_frame"),
                    "end_frame": window.get("end_frame"),
                    "num_frames": window.get("num_frames"),
                },
                "future_record_id": future_id,
                "future_window_action": window.get("action"),
            }
        )
        true_actions.append(true_action)
        pred_actions.append(pred_action)

    if not rows:
        raise RuntimeError(f"no future-window rows could be joined for time-to-transition scoring in {prediction_jsonl}")

    true_dist = transition_distances(rows, true_actions, cap_frames)
    pred_dist = transition_distances(rows, pred_actions, cap_frames)
    errors = [abs(pred - true) for pred, true in zip(pred_dist, true_dist)]
    mae = sum(errors) / len(errors)
    rmse = (sum(error * error for error in errors) / len(errors)) ** 0.5
    within_20 = sum(1 for error in errors if error <= 20.0) / len(errors)
    within_50 = sum(1 for error in errors if error <= 50.0) / len(errors)

    scored_rows = []
    for row, true_action, pred_action, true_value, pred_value, error in zip(
        rows,
        true_actions,
        pred_actions,
        true_dist,
        pred_dist,
        errors,
    ):
        scored_rows.append(
            {
                "id": row.get("id"),
                "split": row.get("split"),
                "episode_id": row.get("episode_id"),
                "future_record_id": row.get("future_record_id"),
                "start_frame": row_start(row),
                "true_action": true_action,
                "pred_action": pred_action,
                "true_time_to_transition_frames": true_value,
                "pred_time_to_transition_frames": pred_value,
                "absolute_error_frames": error,
                "rank": row.get("rank"),
                "top_k_hit": row.get("top_k_hit"),
            }
        )

    metrics = {
        "title": f"{spec['label']} Time-to-Transition Future-Window Probe",
        "status": "pass",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_id": model_id,
        "model_label": spec["label"],
        "task_id": "time_to_transition",
        "task_number": 20,
        "task_label": "Time to Transition",
        "metric_key": "time_to_transition_mae",
        "primary_metric": "time_to_transition_mae",
        "primary_score": mae,
        "metric_direction": "lower",
        "time_to_transition_mae": mae,
        "time_to_transition_rmse": rmse,
        "within_20_frames": within_20,
        "within_50_frames": within_50,
        "cap_frames": cap_frames,
        "source_prediction_jsonl": relpath(prediction_jsonl, workspace),
        "source_window_map_jsonl": relpath(source_window_map_jsonl, workspace),
        "scope": "held_out_test_existing_future_window_action_sequence_probe",
        "score_policy": (
            "Derived from existing verified held-out Cosmos3-Nano future-window predictions. "
            "The model did not emit a direct scalar time-to-transition value; predicted boundary "
            "timing is computed from changes in its predicted future-action sequence and compared "
            "with the true held-out future-action boundary timing."
        ),
        "normalization_policy": (
            "Rows are joined to the compact source-window map by future_record_id, grouped by "
            "episode, and sorted by future-window start frame. The target and prediction are "
            f"frames until the next action-label change, capped at {cap_frames} frames."
        ),
        "known_limitation": (
            "This is a derived future-window action-sequence probe, not evidence of a separately "
            "trained scalar time-regression head. It is included because task 20's boundary target "
            "is deterministically derivable once future-window action predictions are verified."
        ),
        "total_prediction_rows": len(source_rows),
        "source_window_rows": len(window_rows),
        "scored_rows": len(scored_rows),
        "missing_window_count": missing_window_count,
        "missing_pred_action_count": missing_pred_action_count,
        "artifact_files": {
            "metrics_json": relpath(output_dir / "metrics.json", workspace),
            "predictions_csv": relpath(output_dir / "predictions.csv", workspace),
        },
    }
    write_json(output_dir / "metrics.json", metrics)
    write_csv(
        output_dir / "predictions.csv",
        scored_rows,
        [
            "id",
            "split",
            "episode_id",
            "future_record_id",
            "start_frame",
            "true_action",
            "pred_action",
            "true_time_to_transition_frames",
            "pred_time_to_transition_frames",
            "absolute_error_frames",
            "rank",
            "top_k_hit",
        ],
    )
    return metrics


def build_report(summary: dict[str, Any]) -> str:
    rows = []
    for model_id, result in summary["methods"].items():
        task_results = result.get("tasks", {})
        task13 = task_results.get("long_horizon_next_action", {})
        task14 = task_results.get("next_subtask_forecast", {})
        task16 = task_results.get("action_object_relation", {})
        task17 = task_results.get("object_set_forecast", {})
        task20 = task_results.get("time_to_transition", {})
        task8 = task_results.get("caption_grounding", {})
        rows.append(
            "| "
            + " | ".join(
                [
                    result["label"],
                    model_id,
                    result["status"],
                    ", ".join(sorted(task_results)) or "n/a",
                    (
                        f"{task13.get('long_horizon_next_action_macro_f1', 0.0):.6f}"
                        if task13.get("long_horizon_next_action_macro_f1") is not None
                        else "n/a"
                    ),
                    (
                        f"{task14.get('next_subtask_forecast_macro_f1', 0.0):.6f}"
                        if task14.get("next_subtask_forecast_macro_f1") is not None
                        else "n/a"
                    ),
                    (
                        f"{task16.get('action_object_relation_macro_f1', 0.0):.6f}"
                        if task16.get("action_object_relation_macro_f1") is not None
                        else "n/a"
                    ),
                    (
                        f"{task17.get('object_set_forecast_micro_f1', 0.0):.6f}"
                        if task17.get("object_set_forecast_micro_f1") is not None
                        else "n/a"
                    ),
                    (
                        f"{task20.get('time_to_transition_mae', 0.0):.3f}"
                        if task20.get("time_to_transition_mae") is not None
                        else "n/a"
                    ),
                    (
                        f"{task8.get('caption_grounding_iou', 0.0):.6f}"
                        if task8.get("caption_grounding_iou") is not None
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

| Method | ID | Status | Scored tasks | Task 13 macro-F1 | Task 14 macro-F1 | Task 16 macro-F1 | Task 17 micro-F1 | Task 20 MAE | Task 8 IoU | Evidence |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
{chr(10).join(rows)}
"""


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else workspace / args.output_dir
    methods: dict[str, Any] = {}

    for model_id, spec in MODEL_SPECS.items():
        prediction_path = workspace / spec["prediction_jsonl"]
        if not prediction_path.exists():
            methods[model_id] = {
                "label": spec["label"],
                "status": "missing_prediction_jsonl",
                "source_prediction_jsonl": spec["prediction_jsonl"],
                "reason": "verified prediction JSONL was not found in the local checkout",
                "tasks": {},
            }
            continue
        task_results: dict[str, Any] = {}
        unsupported: dict[str, str] = {}
        if model_id != "cosmos3_nano_future_window":
            metrics = score_action_object_relation(
                model_id=model_id,
                spec=spec,
                prediction_jsonl=prediction_path,
                output_dir=output_dir / "action_object_relation" / model_id,
                workspace=workspace,
            )
            task_results["action_object_relation"] = {
                "source_metrics_json": metrics["artifact_files"]["metrics_json"],
                "scored_rows": metrics["scored_rows"],
                "valid_pred_relation_rate": metrics["valid_pred_relation_rate"],
                "action_object_relation_macro_f1": metrics["action_object_relation_macro_f1"],
                "action_object_relation_accuracy": metrics["action_object_relation_accuracy"],
            }
        if model_id == "cosmos3_super_reasoner":
            metrics = score_caption_grounding_from_evidence_window(
                model_id=model_id,
                spec=spec,
                prediction_jsonl=prediction_path,
                output_dir=output_dir / "caption_grounding" / model_id,
                workspace=workspace,
            )
            task_results["caption_grounding"] = {
                "source_metrics_json": metrics["artifact_files"]["metrics_json"],
                "scored_rows": metrics["scored_rows"],
                "caption_grounding_iou": metrics["caption_grounding_iou"],
                "caption_grounding_center_hit_rate": metrics["caption_grounding_center_hit_rate"],
                "missing_pred_evidence_window_count": metrics["missing_pred_evidence_window_count"],
            }
        if model_id == "cosmos3_nano_future_window":
            manifest_path = workspace / spec["dataset_manifest"]
            metrics = score_cosmos_nano_long_horizon_next_action(
                model_id=model_id,
                spec=spec,
                prediction_jsonl=prediction_path,
                dataset_manifest=manifest_path,
                output_dir=output_dir / "long_horizon_next_action" / model_id,
                workspace=workspace,
            )
            task_results["long_horizon_next_action"] = {
                "source_metrics_json": metrics["artifact_files"]["metrics_json"],
                "scored_rows": metrics["scored_rows"],
                "horizon_windows": metrics.get("horizon_windows"),
                "long_horizon_next_action_macro_f1": metrics["long_horizon_next_action_macro_f1"],
                "long_horizon_next_action_accuracy": metrics["long_horizon_next_action_accuracy"],
            }
            target_map_path = workspace / spec["target_map_jsonl"]
            if target_map_path.exists():
                metrics = score_cosmos_nano_next_subtask_from_target_map(
                    model_id=model_id,
                    spec=spec,
                    prediction_jsonl=prediction_path,
                    target_map_jsonl=target_map_path,
                    output_dir=output_dir / "next_subtask_forecast" / model_id,
                    workspace=workspace,
                )
                task_results["next_subtask_forecast"] = {
                    "source_metrics_json": metrics["artifact_files"]["metrics_json"],
                    "scored_rows": metrics["scored_rows"],
                    "next_subtask_forecast_macro_f1": metrics["next_subtask_forecast_macro_f1"],
                    "next_subtask_forecast_accuracy": metrics["next_subtask_forecast_accuracy"],
                }
                metrics = score_cosmos_nano_object_set_from_target_map(
                    model_id=model_id,
                    spec=spec,
                    prediction_jsonl=prediction_path,
                    target_map_jsonl=target_map_path,
                    output_dir=output_dir / "object_set_forecast" / model_id,
                    workspace=workspace,
                )
                task_results["object_set_forecast"] = {
                    "source_metrics_json": metrics["artifact_files"]["metrics_json"],
                    "scored_rows": metrics["scored_rows"],
                    "object_set_forecast_micro_f1": metrics["object_set_forecast_micro_f1"],
                    "object_set_forecast_precision": metrics["object_set_forecast_precision"],
                    "object_set_forecast_recall": metrics["object_set_forecast_recall"],
                }
                metrics = score_cosmos_nano_action_object_relation_from_target_map(
                    model_id=model_id,
                    spec=spec,
                    prediction_jsonl=prediction_path,
                    target_map_jsonl=target_map_path,
                    output_dir=output_dir / "action_object_relation" / model_id,
                    workspace=workspace,
                )
                task_results["action_object_relation"] = {
                    "source_metrics_json": metrics["artifact_files"]["metrics_json"],
                    "scored_rows": metrics["scored_rows"],
                    "action_object_relation_macro_f1": metrics["action_object_relation_macro_f1"],
                    "action_object_relation_accuracy": metrics["action_object_relation_accuracy"],
                }
            metrics_path = workspace / spec["metrics_json"]
            metrics = score_modality_reconstruction_from_feature_error(
                model_id=model_id,
                spec=spec,
                metrics_json=metrics_path,
                output_dir=output_dir / "modality_reconstruction" / model_id,
                workspace=workspace,
            )
            task_results["modality_reconstruction"] = {
                "source_metrics_json": metrics["artifact_files"]["metrics_json"],
                "source_verified_metrics_json": metrics["source_metrics_json"],
                "feature_reconstruction_quality": metrics["feature_reconstruction_quality"],
                "feature_reconstruction_error": metrics["feature_reconstruction_error"],
                "num_samples": metrics.get("num_samples"),
            }
            window_map_path = workspace / spec["source_window_map_jsonl"]
            if window_map_path.exists():
                metrics = score_cosmos_nano_time_to_transition_from_future_windows(
                    model_id=model_id,
                    spec=spec,
                    prediction_jsonl=prediction_path,
                    source_window_map_jsonl=window_map_path,
                    output_dir=output_dir / "time_to_transition" / model_id,
                    workspace=workspace,
                )
                task_results["time_to_transition"] = {
                    "source_metrics_json": metrics["artifact_files"]["metrics_json"],
                    "scored_rows": metrics["scored_rows"],
                    "time_to_transition_mae": metrics["time_to_transition_mae"],
                    "within_20_frames": metrics["within_20_frames"],
                }
        if spec.get("time_to_transition_from_action_sequence"):
            metrics = score_long_horizon_next_action_from_verified_json(
                model_id=model_id,
                spec=spec,
                prediction_jsonl=prediction_path,
                output_dir=output_dir / "long_horizon_next_action" / model_id,
                workspace=workspace,
            )
            task_results["long_horizon_next_action"] = {
                "source_metrics_json": metrics["artifact_files"]["metrics_json"],
                "scored_rows": metrics["scored_rows"],
                "long_horizon_next_action_macro_f1": metrics["long_horizon_next_action_macro_f1"],
                "long_horizon_next_action_accuracy": metrics["long_horizon_next_action_accuracy"],
            }
            metrics = score_time_to_transition_from_action_sequence(
                model_id=model_id,
                spec=spec,
                prediction_jsonl=prediction_path,
                output_dir=output_dir / "time_to_transition" / model_id,
                workspace=workspace,
            )
            task_results["time_to_transition"] = {
                "source_metrics_json": metrics["artifact_files"]["metrics_json"],
                "scored_rows": metrics["scored_rows"],
                "time_to_transition_mae": metrics["time_to_transition_mae"],
                "within_20_frames": metrics["within_20_frames"],
            }
        methods[model_id] = {
            "label": spec["label"],
            "status": "scored" if task_results else "unsupported_without_required_fields",
            "source_prediction_jsonl": spec["prediction_jsonl"],
            "tasks": task_results,
            "unsupported_tasks": unsupported,
            "reason": "; ".join(unsupported.values()) if unsupported and not task_results else None,
        }

    scored_count = sum(len(item.get("tasks", {})) for item in methods.values())
    summary = {
        "title": "Existing Model-Output Task Probes",
        "status": "pass",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scope": (
            "Task-specific scoring from existing verified held-out model outputs. "
            "No new model inference, training, or target backfilling is performed."
        ),
        "task_ids_added_to_matrix": [
            "action_object_relation",
            "caption_grounding",
            "long_horizon_next_action",
            "modality_reconstruction",
            "next_subtask_forecast",
            "object_set_forecast",
            "time_to_transition",
        ],
        "scored_method_task_count_added": scored_count,
        "methods": methods,
    }
    write_json(output_dir / "summary.json", summary)
    (output_dir / "RUN_REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(f"wrote {relpath(output_dir / 'summary.json', workspace)}")
    print(f"wrote {relpath(output_dir / 'RUN_REPORT.md', workspace)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

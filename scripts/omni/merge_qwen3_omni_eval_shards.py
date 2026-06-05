#!/usr/bin/env python3
"""Merge Qwen3-Omni held-out eval shards and recompute final metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from qwen3_omni_dataset_utils import (
    class_metrics,
    json_validity_rate,
    label_counts,
    load_jsonl,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    workspace_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, action="append", required=True)
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--model-id", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--run-id", default="qwen_lora_eval_merged")
    parser.add_argument("--workspace", type=Path, default=workspace_default)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def field_accuracy(rows: list[dict], field: str) -> float | None:
    valid_rows = [row for row in rows if row["true_json"].get(field) != "unknown"]
    if not valid_rows:
        return None
    return sum(row["pred_json"].get(field) == row["true_json"].get(field) for row in valid_rows) / len(valid_rows)


def object_micro_f1(rows: list[dict]) -> float | None:
    tp = fp = fn = 0
    for row in rows:
        true_objects = set(row["true_json"].get("objects") or [])
        pred_objects = set(row["pred_json"].get("objects") or [])
        tp += len(true_objects & pred_objects)
        fp += len(pred_objects - true_objects)
        fn += len(true_objects - pred_objects)
    if tp + fp + fn == 0:
        return None
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def load_shard_predictions(shard_dirs: list[Path]) -> tuple[list[dict], list[dict]]:
    rows_by_id: dict[str, dict] = {}
    issues = []
    for shard_dir in shard_dirs:
        path = shard_dir / "predictions.jsonl"
        if not path.exists():
            issues.append({"stage": "load", "message": f"missing predictions: {path}"})
            continue
        for row in load_jsonl(path):
            sample_id = str(row.get("id", ""))
            if not sample_id:
                issues.append({"stage": "load", "message": f"prediction row without id in {path}"})
                continue
            if sample_id in rows_by_id:
                issues.append({"stage": "load", "message": f"duplicate prediction id {sample_id}"})
                continue
            rows_by_id[sample_id] = row
    return list(rows_by_id.values()), issues


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_jsonl(args.dataset_jsonl)
    eval_samples = [sample for sample in samples if sample.get("split") == args.eval_split]
    if not eval_samples:
        raise ValueError("No evaluation samples selected.")

    expected_ids = [sample["id"] for sample in eval_samples]
    expected_id_set = set(expected_ids)
    rows, issues = load_shard_predictions(args.shard_dir)
    rows = [row for row in rows if row.get("id") in expected_id_set]
    rows_by_id = {row["id"]: row for row in rows}
    missing_ids = [sample_id for sample_id in expected_ids if sample_id not in rows_by_id]
    if missing_ids:
        issues.append({"stage": "coverage", "message": f"missing {len(missing_ids)} eval predictions", "examples": missing_ids[:20]})
    if issues and not args.allow_missing:
        raise RuntimeError(json.dumps({"issues": issues}, indent=2))

    ordered_rows = [rows_by_id[sample_id] for sample_id in expected_ids if sample_id in rows_by_id]
    train_labels = {
        sample.get("answer_json", {}).get("action", sample.get("label", "unknown"))
        for sample in samples
        if sample.get("split") == args.train_split
    }
    eval_labels = {
        sample.get("answer_json", {}).get("action", sample.get("label", "unknown"))
        for sample in eval_samples
    }
    unseen_labels = sorted(eval_labels - train_labels)
    label_options = eval_samples[0]["label_options"]
    metrics, per_class, cm = class_metrics(
        [row["true_label"] for row in ordered_rows],
        [row["predicted_label"] for row in ordered_rows],
        label_options,
    )
    seen_rows = [row for row in ordered_rows if row.get("true_label_seen_in_train")]
    unseen_rows = [row for row in ordered_rows if not row.get("true_label_seen_in_train")]
    metrics.update({
        "run_id": args.run_id,
        "model_id": args.model_id,
        "adapter_dir": str(args.adapter_dir) if args.adapter_dir else None,
        "dataset_jsonl": str(args.dataset_jsonl),
        "eval_split": args.eval_split,
        "train_split": args.train_split,
        "num_eval_episodes": len({row["episode_id"] for row in ordered_rows}),
        "unseen_eval_labels": unseen_labels,
        "num_unseen_label_samples": len(unseen_rows),
        "seen_label_accuracy": sum(row["correct"] for row in seen_rows) / len(seen_rows) if seen_rows else None,
        "unseen_label_accuracy": sum(row["correct"] for row in unseen_rows) / len(unseen_rows) if unseen_rows else None,
        "eval_label_counts": label_counts(eval_samples),
        "json_validity_rate": json_validity_rate([row["raw_prediction"] for row in ordered_rows]),
        "action_macro_f1": metrics["macro_f1"],
        "subtask_accuracy": field_accuracy(ordered_rows, "subtask"),
        "transition_accuracy": field_accuracy(ordered_rows, "transition"),
        "next_action_accuracy": field_accuracy(ordered_rows, "next_action"),
        "contact_accuracy": field_accuracy(ordered_rows, "contact"),
        "object_micro_f1": object_micro_f1(ordered_rows),
        "shard_dirs": [str(path) for path in args.shard_dir],
        "coverage": {
            "expected_eval_samples": len(expected_ids),
            "merged_prediction_rows": len(ordered_rows),
            "missing_prediction_rows": len(missing_ids),
        },
        "issues": issues,
    })

    write_jsonl(args.output_dir / "predictions.jsonl", ordered_rows)
    write_csv(
        args.output_dir / "predictions.csv",
        ordered_rows,
        ["id", "target", "split", "episode_id", "center_window", "true_label", "raw_prediction", "predicted_label", "correct", "true_label_seen_in_train"],
    )
    write_csv(args.output_dir / "per_class_metrics.csv", per_class, ["class_name", "support", "predicted", "precision", "recall", "f1"])
    labels = metrics["labels"]
    with (args.output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["true\\pred"] + labels)
        for label, row in zip(labels, cm):
            writer.writerow([label] + row)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report = [
        "# Qwen3-Omni LoRA Sharded Evaluation",
        "",
        f"- Dataset: `{args.dataset_jsonl}`",
        f"- Eval split: `{args.eval_split}`",
        f"- Expected eval samples: `{len(expected_ids)}`",
        f"- Merged predictions: `{len(ordered_rows)}`",
        f"- Held-out episodes: `{metrics['num_eval_episodes']}`",
        f"- Accuracy: `{metrics['accuracy']:.4f}`",
        f"- Macro-F1: `{metrics['macro_f1']:.4f}`",
        f"- JSON validity: `{metrics['json_validity_rate']:.4f}`",
        "",
        "Artifacts include `metrics.json`, `predictions.csv`, `per_class_metrics.csv`, and `confusion_matrix.csv`.",
    ]
    (args.output_dir / "RUN_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

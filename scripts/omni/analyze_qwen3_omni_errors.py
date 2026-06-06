#!/usr/bin/env python3
"""Analyze public-safe Qwen3-Omni held-out prediction errors.

The script consumes a verified public package, not raw Xperience-10M data. It
summarizes where the diagnostic pilot fails by episode, train-seen status,
coarse action family, object category, parsed prediction state, and
required-modality state. The outputs are small derived CSV/JSON/Markdown
artifacts suitable for the public package.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_PACKAGE = (
    Path(__file__).resolve().parents[2]
    / "results/omni_finetune/verified_public/"
    / "xperience10m_qwen3_omni_128ep_96train_16val_16test_valmon_20260605_eval"
)

ACTION_FAMILIES = [
    ("phone_use", ("phone", "smartphone", "watch", "screen")),
    ("paper_cardboard_craft", ("paper", "cardboard", "fold", "cut", "draw", "mark", "ruler", "scissors", "lantern", "star")),
    ("retail_stocking", ("shelf", "product", "can", "canned", "container", "box", "grocery", "stock")),
    ("small_object_sorting", ("bead", "button", "tile", "mahjong", "puzzle", "piece")),
    ("cleaning", ("clean", "wipe", "wash", "vacuum", "sweep", "trash")),
    ("locomotion", ("walk", "approach", "enter", "move through", "arrive", "leave")),
    ("food_kitchen", ("kettle", "rice", "saucepan", "kitchen", "bottle", "jar", "lid")),
]

OBJECT_CATEGORIES = [
    ("phone_device", ("phone", "smartphone", "watch", "charger", "cable", "power bank", "earbud")),
    ("paper_cardboard", ("paper", "cardboard", "lantern", "origami", "star", "ribbon")),
    ("tool_stationery", ("scissors", "knife", "ruler", "marker", "pen", "stapler", "glue", "tape")),
    ("retail_container", ("shelf", "container", "product", "box", "can", "canned", "package", "bag")),
    ("furniture_room", ("table", "chair", "desk", "counter", "sink", "door", "wall", "floor")),
    ("food_kitchen", ("kettle", "rice", "saucepan", "jar", "bottle", "food", "kitchen")),
    ("craft_small_object", ("bead", "button", "tile", "mahjong", "puzzle", "foam", "piece")),
    ("cleaning", ("vacuum", "broom", "cloth", "towel", "trash")),
]

REQUIRED_VIDEO_FILES = {
    "fisheye_cam0.mp4",
    "fisheye_cam1.mp4",
    "fisheye_cam2.mp4",
    "fisheye_cam3.mp4",
    "stereo_left.mp4",
    "stereo_right.mp4",
}

REQUIRED_HDF5_MODALITIES = {
    "calibration",
    "slam_pose",
    "slam_point_cloud",
    "depth",
    "depth_confidence",
    "hand_mocap",
    "body_mocap",
    "contacts",
    "imu",
    "caption",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-examples", type=int, default=12)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def norm(value: Any) -> str:
    return str(value or "").strip().lower()


def family_for(text: str, families: list[tuple[str, tuple[str, ...]]], fallback: str = "other") -> str:
    low = norm(text)
    for name, keywords in families:
        if any(keyword in low for keyword in keywords):
            return name
    return fallback


def object_categories(objects: list[Any]) -> set[str]:
    categories: set[str] = set()
    for obj in objects:
        categories.add(family_for(str(obj), OBJECT_CATEGORIES, "other_object"))
    return categories or {"no_object_label"}


def f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def bool_metric(row: dict[str, Any], key: str) -> bool:
    true_json = row.get("true_json") or {}
    pred_json = row.get("pred_json") or {}
    return norm(true_json.get(key)) == norm(pred_json.get(key)) and bool(pred_json)


def object_overlap(row: dict[str, Any]) -> tuple[int, int, int]:
    true_objects = {norm(item) for item in (row.get("true_json") or {}).get("objects", []) if norm(item)}
    pred_objects = {norm(item) for item in (row.get("pred_json") or {}).get("objects", []) if norm(item)}
    return len(true_objects & pred_objects), len(pred_objects), len(true_objects)


def modality_state(episode: dict[str, Any] | None) -> tuple[str, list[str]]:
    if not episode:
        return "episode_manifest_missing", ["episode_manifest_missing"]
    missing: list[str] = []
    files = {str(item.get("name")): bool(item.get("exists")) for item in episode.get("files", [])}
    for filename in sorted(REQUIRED_VIDEO_FILES):
        if not files.get(filename):
            missing.append(filename)
    hdf5 = episode.get("hdf5_modalities") or {}
    for modality in sorted(REQUIRED_HDF5_MODALITIES):
        if not hdf5.get(modality):
            missing.append(modality)
    if missing:
        return "missing_required_modalities", missing
    if files.get("visualization.rrd") is False:
        return "rrd_missing_only_required_modalities_present", ["visualization.rrd"]
    return "required_modalities_present", []


def add_row_stats(bucket: dict[str, Any], row: dict[str, Any]) -> None:
    bucket["samples"] += 1
    valid = bool(row.get("pred_json"))
    bucket["parsed_predictions"] += int(valid)
    bucket["action_exact"] += int(bool_metric(row, "action"))
    bucket["subtask_exact"] += int(bool_metric(row, "subtask"))
    bucket["transition_exact"] += int(bool_metric(row, "transition"))
    bucket["next_action_exact"] += int(bool_metric(row, "next_action"))
    bucket["contact_exact"] += int(bool_metric(row, "contact"))
    matched, pred_count, true_count = object_overlap(row)
    bucket["object_matched"] += matched
    bucket["object_predicted"] += pred_count
    bucket["object_true"] += true_count


def empty_bucket() -> dict[str, Any]:
    return {
        "samples": 0,
        "parsed_predictions": 0,
        "action_exact": 0,
        "subtask_exact": 0,
        "transition_exact": 0,
        "next_action_exact": 0,
        "contact_exact": 0,
        "object_matched": 0,
        "object_predicted": 0,
        "object_true": 0,
    }


def finalize_bucket(name: str, bucket: dict[str, Any]) -> dict[str, Any]:
    samples = max(int(bucket["samples"]), 1)
    precision = bucket["object_matched"] / bucket["object_predicted"] if bucket["object_predicted"] else 0.0
    recall = bucket["object_matched"] / bucket["object_true"] if bucket["object_true"] else 0.0
    return {
        "group": name,
        "samples": bucket["samples"],
        "parsed_prediction_rate": bucket["parsed_predictions"] / samples,
        "action_exact_rate": bucket["action_exact"] / samples,
        "subtask_exact_rate": bucket["subtask_exact"] / samples,
        "transition_exact_rate": bucket["transition_exact"] / samples,
        "next_action_exact_rate": bucket["next_action_exact"] / samples,
        "contact_exact_rate": bucket["contact_exact"] / samples,
        "object_precision": precision,
        "object_recall": recall,
        "object_f1": f1(precision, recall),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def top_rows(groups: dict[str, dict[str, Any]], *, min_samples: int = 1, reverse: bool = False) -> list[dict[str, Any]]:
    rows = [finalize_bucket(name, bucket) for name, bucket in groups.items() if bucket["samples"] >= min_samples]
    return sorted(rows, key=lambda row: (row["parsed_prediction_rate"], row["action_exact_rate"], row["samples"]), reverse=reverse)


def markdown_table(rows: list[dict[str, Any]], columns: list[str], limit: int = 8) -> list[str]:
    selected = rows[:limit]
    if not selected:
        return ["No rows."]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in selected:
        values = []
        for col in columns:
            value = row.get(col)
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def main() -> int:
    args = parse_args()
    package_dir = args.package_dir.expanduser().resolve()
    output_dir = args.output_dir or package_dir / "analysis"
    output_dir = output_dir.expanduser().resolve()

    predictions = load_jsonl(package_dir / "eval" / "predictions.jsonl")
    metrics = load_json(package_dir / "eval" / "metrics.json")
    episode_manifest = load_json(package_dir / "dataset" / "episode_manifest.json")
    episodes = {episode.get("episode_id"): episode for episode in episode_manifest.get("episodes", [])}

    overall = empty_bucket()
    by_episode: dict[str, dict[str, Any]] = defaultdict(empty_bucket)
    by_family: dict[str, dict[str, Any]] = defaultdict(empty_bucket)
    by_seen: dict[str, dict[str, Any]] = defaultdict(empty_bucket)
    by_modality: dict[str, dict[str, Any]] = defaultdict(empty_bucket)
    by_object_category: dict[str, dict[str, Any]] = defaultdict(empty_bucket)
    invalid_examples = []
    overgenerated_examples = []
    modality_missing_by_episode: dict[str, list[str]] = {}

    for row in predictions:
        episode_id = str(row.get("episode_id"))
        true_json = row.get("true_json") or {}
        pred_json = row.get("pred_json") or {}
        add_row_stats(overall, row)
        add_row_stats(by_episode[episode_id], row)
        add_row_stats(by_family[family_for(str(true_json.get("action")), ACTION_FAMILIES)], row)
        add_row_stats(by_seen["seen_in_train" if row.get("true_label_seen_in_train") else "unseen_in_train"], row)
        state, missing = modality_state(episodes.get(episode_id))
        modality_missing_by_episode.setdefault(episode_id, missing)
        add_row_stats(by_modality[state], row)
        for category in object_categories(true_json.get("objects", [])):
            add_row_stats(by_object_category[category], row)
        if not pred_json and len(invalid_examples) < args.max_examples:
            invalid_examples.append({
                "id": row.get("id"),
                "episode_id": episode_id,
                "true_action": true_json.get("action"),
                "raw_prediction_prefix": str(row.get("raw_prediction", ""))[:240],
            })
        pred_objects = pred_json.get("objects", []) if isinstance(pred_json, dict) else []
        if len(pred_objects) > 20 and len(overgenerated_examples) < args.max_examples:
            overgenerated_examples.append({
                "id": row.get("id"),
                "episode_id": episode_id,
                "true_action": true_json.get("action"),
                "predicted_object_count": len(pred_objects),
                "first_predicted_objects": pred_objects[:20],
            })

    episode_rows = top_rows(by_episode)
    family_rows = top_rows(by_family)
    seen_rows = top_rows(by_seen)
    modality_rows = top_rows(by_modality)
    object_rows = top_rows(by_object_category)

    write_csv(output_dir / "episode_error_analysis.csv", episode_rows)
    write_csv(output_dir / "action_family_error_analysis.csv", family_rows)
    write_csv(output_dir / "train_seen_error_analysis.csv", seen_rows)
    write_csv(output_dir / "missing_modality_error_analysis.csv", modality_rows)
    write_csv(output_dir / "object_category_error_analysis.csv", object_rows)

    summary = {
        "status": "pass",
        "source_package": package_dir.name,
        "source_prediction_rows": len(predictions),
        "metrics_json_validity_rate": metrics.get("json_validity_rate"),
        "computed": finalize_bucket("overall", overall),
        "worst_episode_groups": episode_rows[:8],
        "action_family_groups": family_rows,
        "train_seen_groups": seen_rows,
        "missing_modality_groups": modality_rows,
        "object_category_groups": object_rows,
        "invalid_json_examples": invalid_examples,
        "object_overgeneration_examples": overgenerated_examples,
        "modality_missing_by_episode": modality_missing_by_episode,
        "interpretation": (
            "The diagnostic pilot is dominated by invalid or weak structured outputs and exact-label failures. "
            "These tables identify where to tighten JSON constraints, action/subtask target formatting, object vocabularies, "
            "and missing-modality robustness before claiming stronger model quality."
        ),
    }
    (output_dir / "error_analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    report = [
        "# Qwen3-Omni Held-Out Error Analysis",
        "",
        "This report is computed from the verified public package predictions. It contains only derived metrics and sanitized examples.",
        "",
        "## Overall",
        "",
        f"- Prediction rows: `{len(predictions)}`",
        f"- JSON validity from `metrics.json`: `{summary['metrics_json_validity_rate']:.4f}`",
        f"- Parsed prediction rate from public rows: `{summary['computed']['parsed_prediction_rate']:.4f}`",
        f"- Action exact rate: `{summary['computed']['action_exact_rate']:.4f}`",
        f"- Subtask exact rate: `{summary['computed']['subtask_exact_rate']:.4f}`",
        f"- Contact exact rate: `{summary['computed']['contact_exact_rate']:.4f}`",
        f"- Object F1: `{summary['computed']['object_f1']:.4f}`",
        "",
        "## Weakest Episode Groups",
        "",
        *markdown_table(episode_rows, ["group", "samples", "parsed_prediction_rate", "action_exact_rate", "object_f1"]),
        "",
        "## Action Families",
        "",
        *markdown_table(family_rows, ["group", "samples", "parsed_prediction_rate", "action_exact_rate", "subtask_exact_rate", "object_f1"]),
        "",
        "## Train-Seen Split",
        "",
        *markdown_table(seen_rows, ["group", "samples", "parsed_prediction_rate", "action_exact_rate", "next_action_exact_rate"]),
        "",
        "## Required-Modality State",
        "",
        *markdown_table(modality_rows, ["group", "samples", "parsed_prediction_rate", "action_exact_rate", "object_f1"]),
        "",
        "## Object Categories",
        "",
        *markdown_table(object_rows, ["group", "samples", "object_precision", "object_recall", "object_f1"]),
        "",
        "## Interpretation",
        "",
        summary["interpretation"],
        "",
        "Generated files:",
        "",
        "- `error_analysis_summary.json`",
        "- `episode_error_analysis.csv`",
        "- `action_family_error_analysis.csv`",
        "- `train_seen_error_analysis.csv`",
        "- `missing_modality_error_analysis.csv`",
        "- `object_category_error_analysis.csv`",
    ]
    (output_dir / "ERROR_ANALYSIS.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "output_dir": str(output_dir), "prediction_rows": len(predictions)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

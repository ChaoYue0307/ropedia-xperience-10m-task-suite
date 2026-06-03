#!/usr/bin/env python3
"""Audit semantic content labels after Xperience-10M annotations are staged."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py


CATEGORY_RULES = {
    "food_and_drink": ["cook", "coffee", "drink", "food", "kitchen", "meal", "pour", "cup", "bottle"],
    "dressing_and_hygiene": ["sock", "shoe", "dress", "wear", "bathroom", "toilet", "wash", "brush"],
    "packing_and_organizing": ["pack", "organize", "bin", "box", "storage", "arrange", "sort"],
    "shopping_and_retail": ["retail", "shop", "shelf", "aisle", "product", "store"],
    "cleaning_and_housework": ["clean", "wipe", "sweep", "wash", "laundry", "trash"],
    "navigation_and_locomotion": ["walk", "move through", "navigate", "stair", "hallway", "corridor"],
    "tool_or_device_use": ["tool", "device", "phone", "computer", "laptop", "button", "switch"],
    "object_manipulation": ["pick", "place", "grasp", "hold", "open", "close", "move", "lift"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--selection-json", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=Path("results/omni_finetune/staged_content_audit.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("results/omni_finetune/staged_content_audit.csv"))
    parser.add_argument("--report-output", type=Path, default=Path("results/omni_finetune/STAGED_CONTENT_AUDIT.md"))
    return parser.parse_args()


def load_selection(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {ep["episode_path"]: ep for ep in payload.get("selected_episodes", [])}


def parse_caption(annotation: Path) -> dict[str, Any]:
    with h5py.File(annotation, "r") as h5:
        if "caption" not in h5:
            return {"parse_status": "missing"}
        raw = h5["caption"][()]
    text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
    try:
        data = json.loads(text)
    except Exception as exc:
        return {"parse_status": "failed", "error": str(exc), "json_bytes": len(text.encode("utf-8"))}

    config = data.get("config", {}) if isinstance(data, dict) else {}
    segments = data.get("segments", []) if isinstance(data, dict) else []
    if not isinstance(segments, list):
        segments = []

    subtasks: list[str] = []
    actions: list[str] = []
    objects: list[str] = []
    interactions: list[str] = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        if segment.get("Sub Task"):
            subtasks.append(str(segment["Sub Task"]))
        current_actions = segment.get("Current Action", [])
        if isinstance(current_actions, list):
            for action in current_actions:
                if isinstance(action, dict):
                    if action.get("label"):
                        actions.append(str(action["label"]))
                    if action.get("description"):
                        interactions.append(str(action["description"]))
        object_map = segment.get("objects", {})
        if isinstance(object_map, dict):
            for names in object_map.values():
                if isinstance(names, list):
                    objects.extend(str(name) for name in names)
        interaction_map = segment.get("interaction", {})
        if isinstance(interaction_map, dict):
            interactions.extend(str(value) for value in interaction_map.values())

    main_task = str(config.get("Main Task", ""))
    global_summary = str(data.get("global_summary", "")) if isinstance(data, dict) else ""
    return {
        "parse_status": "ok",
        "json_bytes": len(text.encode("utf-8")),
        "main_task": main_task,
        "global_summary": global_summary,
        "segment_count": len(segments),
        "subtasks": sorted(set(subtasks)),
        "actions": sorted(set(actions)),
        "objects": sorted(set(objects)),
        "interaction_preview": " ".join(interactions)[:500],
    }


def derive_category(record: dict[str, Any]) -> str:
    text = " ".join(
        [
            record.get("main_task", ""),
            record.get("global_summary", ""),
            " ".join(record.get("subtasks", [])),
            " ".join(record.get("actions", [])),
            " ".join(record.get("objects", [])),
            record.get("interaction_preview", ""),
        ]
    ).lower()
    scores = {
        category: sum(1 for keyword in keywords if keyword in text)
        for category, keywords in CATEGORY_RULES.items()
    }
    category, score = max(scores.items(), key=lambda item: item[1])
    return category if score > 0 else "uncategorized"


def infer_episode_key(annotation: Path, data_root: Path) -> str:
    parent = annotation.parent
    try:
        return parent.relative_to(data_root).as_posix()
    except ValueError:
        return parent.as_posix()


def md_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return lines


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "episode_path",
        "split",
        "size_band",
        "category",
        "main_task",
        "segment_count",
        "actions",
        "objects",
        "annotation_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "episode_path": row["episode_path"],
                    "split": row.get("split", ""),
                    "size_band": row.get("size_band", ""),
                    "category": row["category"],
                    "main_task": row.get("main_task", ""),
                    "segment_count": row.get("segment_count", 0),
                    "actions": "; ".join(row.get("actions", [])),
                    "objects": "; ".join(row.get("objects", [])),
                    "annotation_path": row["annotation_path"],
                }
            )


def main() -> int:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    selection = load_selection(args.selection_json)
    annotations = sorted(data_root.rglob("annotation.hdf5"))
    rows: list[dict[str, Any]] = []
    for annotation in annotations:
        episode_path = infer_episode_key(annotation, data_root)
        parsed = parse_caption(annotation)
        selected_meta = selection.get(episode_path, {})
        record = {
            "episode_path": episode_path,
            "annotation_path": str(annotation),
            "split": selected_meta.get("split", ""),
            "size_band": selected_meta.get("size_band", ""),
            **parsed,
        }
        record["category"] = derive_category(record) if parsed.get("parse_status") == "ok" else "unparsed"
        rows.append(record)

    category_counts = Counter(row["category"] for row in rows)
    split_category_counts: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        split_category_counts[row.get("split", "")][row["category"]] += 1

    payload = {
        "status": "pass",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_root": str(data_root),
        "selection_json": str(args.selection_json) if args.selection_json else None,
        "episode_count": len(rows),
        "category_counts": dict(category_counts.most_common()),
        "split_category_counts": {split or "unknown": dict(counts.most_common()) for split, counts in split_category_counts.items()},
        "rows": rows,
        "note": "Categories are keyword-derived from caption text and should be reviewed before final training claims.",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_csv(args.output_csv, rows)

    report = [
        "# Xperience-10M Staged Content Audit",
        "",
        "This report parses staged `annotation.hdf5` files and derives coarse content categories from caption text.",
        "",
        f"- Data root: `{data_root}`",
        f"- Episodes parsed: {len(rows)}",
        "",
        "## Category Counts",
        "",
        *md_table(["Category", "Episodes"], [[cat, count] for cat, count in category_counts.most_common()]),
        "",
        "## Split x Category",
        "",
    ]
    categories = sorted(category_counts)
    report.extend(
        md_table(
            ["Split", *categories],
            [
                [split or "unknown", *[counts.get(category, 0) for category in categories]]
                for split, counts in sorted(split_category_counts.items())
            ],
        )
    )
    report.extend(
        [
            "",
            "## Next Action",
            "",
            "If one category dominates train, val, or test, swap episodes from the staged pool before starting model fine-tuning.",
        ]
    )
    args.report_output.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"episode_count": len(rows), "category_counts": payload["category_counts"]}, indent=2))
    print(f"PASS: wrote {args.output_json}")
    print(f"PASS: wrote {args.output_csv}")
    print(f"PASS: wrote {args.report_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

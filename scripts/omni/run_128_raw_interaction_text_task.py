#!/usr/bin/env python3
"""Score 128-episode task 15 from extracted raw annotation interaction text.

The public 128 JSONL does not contain the original ``annotation.hdf5`` caption
``interaction`` strings.  This runner consumes the compact JSONL emitted by
``extract_xperience10m_annotation_captions.py``, aligns those raw interaction
strings back to the 128-episode window timeline, and writes the expected
metadata baseline artifacts:

* results/omni_finetune/a100_128_metadata_task_baselines_20260616_v2/
  interaction_text_prediction/metrics.json
* results/omni_finetune/a100_128_metadata_task_baselines_20260616_v2/
  neural_mlp/interaction_text_prediction/metrics.json

It intentionally refuses partial extraction manifests unless --allow-partial is
used for smoke testing.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "omni"))

from run_128_task_baselines import (  # noqa: E402
    build_feature_matrix,
    classification_baseline,
    load_jsonl,
    split_indices,
    write_csv,
    write_json,
)


DEFAULT_DATASET = (
    ROOT
    / "results/omni_finetune/xperience10m_qwen3_omni_128ep_multiscale_cap96_v5_full8gpu_lora_dataset"
    / "dataset_a100_eval.jsonl"
)
DEFAULT_CAPTION_DIR = ROOT / "results/omni_finetune/xperience10m_128_raw_caption_interactions_task15_20260619_full"
DEFAULT_OUTPUT = ROOT / "results/omni_finetune/a100_128_metadata_task_baselines_20260616_v2"
TASK_ID = "interaction_text_prediction"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-jsonl", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--caption-jsonl", type=Path, default=DEFAULT_CAPTION_DIR / "caption_interactions.jsonl")
    parser.add_argument("--caption-manifest", type=Path, default=DEFAULT_CAPTION_DIR / "caption_interactions_manifest.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--hash-dim", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--learning-rate", type=float, default=0.16)
    parser.add_argument("--l2", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--softmax-max-train-classes", type=int, default=256)
    parser.add_argument("--include-neural", action="store_true", default=True)
    parser.add_argument("--neural-epochs", type=int, default=35)
    parser.add_argument("--neural-hidden-dim", type=int, default=128)
    parser.add_argument("--neural-batch-size", type=int, default=256)
    parser.add_argument("--neural-learning-rate", type=float, default=1e-3)
    parser.add_argument("--neural-weight-decay", type=float, default=1e-4)
    parser.add_argument("--neural-dropout", type=float, default=0.10)
    parser.add_argument("--neural-device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--max-neural-classes",
        type=int,
        default=12000,
        help="Keep the raw text classifier exact unless the train label space is extremely large.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def log(message: str) -> None:
    print(f"[128-raw-interaction-task] {message}", file=sys.stderr, flush=True)


def episode_id_from_annotation_path(path_text: str) -> str:
    parts = Path(path_text).parts
    if len(parts) >= 2:
        return f"{parts[-3]}__{parts[-2]}" if parts[-1] == "annotation.hdf5" and len(parts) >= 3 else f"{parts[-2]}__{parts[-1]}"
    return path_text.replace("/", "__")


def normalized_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def load_caption_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = normalized_text(row.get("interaction_text"))
            if not text:
                continue
            annotation_path = str(row.get("annotation_path") or "")
            episode_id = episode_id_from_annotation_path(annotation_path)
            raw_frame = row.get("interaction_frame_sort")
            if raw_frame is None:
                raw_frame = row.get("interaction_frame")
            try:
                frame_value = float(raw_frame)
            except (TypeError, ValueError):
                continue
            rows.append(
                {
                    "episode_id": episode_id,
                    "annotation_path": annotation_path,
                    "raw_frame": frame_value,
                    "interaction_text": text,
                    "sub_task": row.get("sub_task"),
                    "main_task": row.get("main_task"),
                }
            )
    return rows


def max_window_end(rows: list[dict[str, Any]]) -> dict[str, int]:
    values: dict[str, int] = defaultdict(int)
    for row in rows:
        episode_id = str(row.get("episode_id"))
        window = row.get("center_window") or {}
        try:
            end_frame = int(window.get("end_frame", 0) or 0)
        except (TypeError, ValueError):
            end_frame = 0
        values[episode_id] = max(values[episode_id], end_frame)
    return dict(values)


def build_episode_interactions(
    caption_rows: list[dict[str, Any]],
    dataset_rows: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    dataset_max = max_window_end(dataset_rows)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in caption_rows:
        grouped[row["episode_id"]].append(row)

    aligned: dict[str, list[dict[str, Any]]] = {}
    summaries: list[dict[str, Any]] = []
    for episode_id, items in grouped.items():
        if episode_id not in dataset_max:
            summaries.append({"episode_id": episode_id, "status": "caption_only", "interaction_count": len(items)})
            continue
        raw = np.asarray([float(item["raw_frame"]) for item in items], dtype=np.float64)
        raw_min = float(raw.min())
        raw_max = float(raw.max())
        span = max(raw_max - raw_min, 1.0)
        max_end = max(int(dataset_max[episode_id]), 1)
        episode_items: list[dict[str, Any]] = []
        for item in items:
            mapped = int(round((float(item["raw_frame"]) - raw_min) / span * max_end))
            episode_items.append({**item, "mapped_frame": mapped})
        episode_items.sort(key=lambda item: (int(item["mapped_frame"]), item["interaction_text"]))
        aligned[episode_id] = episode_items
        summaries.append(
            {
                "episode_id": episode_id,
                "status": "aligned",
                "interaction_count": len(items),
                "raw_frame_min": raw_min,
                "raw_frame_max": raw_max,
                "dataset_max_end_frame": max_end,
            }
        )
    return aligned, summaries


def row_center(row: dict[str, Any]) -> int:
    window = row.get("center_window") or {}
    start = int(window.get("start_frame", 0) or 0)
    end = int(window.get("end_frame", start) or start)
    return int(round((start + end) / 2.0))


def assign_interaction_labels(
    rows: list[dict[str, Any]],
    interactions: dict[str, list[dict[str, Any]]],
) -> tuple[list[str], list[dict[str, Any]]]:
    labels: list[str] = []
    assigned_rows: list[dict[str, Any]] = []
    for row in rows:
        episode_id = str(row.get("episode_id"))
        candidates = interactions.get(episode_id) or []
        if not candidates:
            labels.append("")
            assigned_rows.append({})
            continue
        center = row_center(row)
        best = min(candidates, key=lambda item: (abs(int(item["mapped_frame"]) - center), int(item["mapped_frame"])))
        labels.append(str(best["interaction_text"]))
        assigned_rows.append(
            {
                "assigned_interaction_text": best["interaction_text"],
                "assigned_interaction_frame": int(best["mapped_frame"]),
                "assigned_raw_interaction_frame": best["raw_frame"],
                "assigned_sub_task": best.get("sub_task"),
            }
        )
    return labels, assigned_rows


def write_label_audit(
    path: Path,
    feature_rows: list[dict[str, Any]],
    labels: list[str],
    assigned_rows: list[dict[str, Any]],
    splits: dict[str, np.ndarray],
) -> None:
    rows = []
    split_by_index = {}
    for split, indices in splits.items():
        for idx in indices:
            split_by_index[int(idx)] = split
    for idx, label in enumerate(labels):
        if not label:
            continue
        rows.append(
            {
                **feature_rows[idx],
                **assigned_rows[idx],
                "split": split_by_index.get(idx, feature_rows[idx].get("split")),
                "label": label,
            }
        )
    write_csv(path, rows)


def labeled_split_counts(rows: list[dict[str, Any]], labels: list[str]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row, label in zip(rows, labels):
        if label:
            counts[str(row.get("split"))] += 1
    return dict(sorted(counts.items()))


def check_manifest(args: argparse.Namespace) -> dict[str, Any]:
    if not args.caption_manifest.exists():
        raise FileNotFoundError(f"caption manifest is missing: {args.caption_manifest}")
    manifest = read_json(args.caption_manifest)
    if manifest.get("status") != "pass" and not args.allow_partial:
        raise SystemExit(
            f"Caption extraction is not complete: status={manifest.get('status')} "
            f"processed={manifest.get('processed_file_count')}/{manifest.get('requested_file_count')}. "
            "Use --allow-partial only for smoke tests."
        )
    return manifest


def main() -> None:
    args = parse_args()
    manifest = check_manifest(args)
    if not args.dataset_jsonl.exists():
        raise FileNotFoundError(f"dataset JSONL is missing: {args.dataset_jsonl}")
    if not args.caption_jsonl.exists():
        raise FileNotFoundError(f"caption interaction JSONL is missing: {args.caption_jsonl}")

    log(f"loading dataset rows from {args.dataset_jsonl}")
    rows = load_jsonl(args.dataset_jsonl)
    log(f"loading raw interaction rows from {args.caption_jsonl}")
    caption_rows = load_caption_rows(args.caption_jsonl)
    interactions, episode_summaries = build_episode_interactions(caption_rows, rows)
    labels, assigned_rows = assign_interaction_labels(rows, interactions)
    labeled_count = sum(1 for label in labels if label)
    if labeled_count == 0:
        raise RuntimeError("no dataset windows could be assigned raw interaction labels")
    split_counts = labeled_split_counts(rows, labels)
    if not split_counts.get("train") or not split_counts.get("test"):
        raise RuntimeError(
            "raw interaction labels do not cover both train and test splits; "
            f"labeled_split_counts={split_counts}. Wait for more annotation files before scoring."
        )

    log(f"building metadata/text features for {len(rows)} rows; {labeled_count} rows have raw interaction labels")
    X, feature_rows = build_feature_matrix(rows, {}, args.hash_dim)
    splits = split_indices(rows)

    def label_getter(row: dict[str, Any]) -> str:
        idx = row_index[id(row)]
        return labels[idx]

    # classification_baseline receives row objects, so id(row) is stable for
    # this in-process call and avoids mutating the compact JSONL rows.
    row_index = {id(row): idx for idx, row in enumerate(rows)}
    result = classification_baseline(TASK_ID, rows, feature_rows, X, splits, label_getter, args.output_dir, args)

    write_label_audit(args.output_dir / TASK_ID / "raw_interaction_label_audit.csv", feature_rows, labels, assigned_rows, splits)
    summary = {
        "status": "pass",
        "task": TASK_ID,
        "source_caption_manifest": str(args.caption_manifest),
        "source_caption_jsonl": str(args.caption_jsonl),
        "source_dataset_jsonl": str(args.dataset_jsonl),
        "allow_partial": bool(args.allow_partial),
        "caption_manifest_status": manifest.get("status"),
        "requested_annotation_file_count": manifest.get("requested_file_count"),
        "processed_annotation_file_count": manifest.get("processed_file_count"),
        "caption_interaction_row_count": len(caption_rows),
        "dataset_window_count": len(rows),
        "labeled_window_count": labeled_count,
        "labeled_split_counts": split_counts,
        "episode_alignment_policy": (
            "For each episode, raw annotation interaction timestamps are linearly mapped from "
            "that episode's observed raw interaction timestamp range onto the 0-based 128-JSONL "
            "window frame range; each window receives the nearest mapped raw interaction text."
        ),
        "aligned_episode_count": len(interactions),
        "episode_alignment": episode_summaries,
        "simple": result.get("simple"),
        "neural": result.get("neural"),
    }
    write_json(args.output_dir / TASK_ID / "raw_interaction_task_summary.json", summary)
    log("done")


if __name__ == "__main__":
    main()

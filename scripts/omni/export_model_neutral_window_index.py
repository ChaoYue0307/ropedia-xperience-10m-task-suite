#!/usr/bin/env python3
"""Convert exported omni JSONL records into a backbone-neutral window index."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a model-neutral Xperience-10M window index.")
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--output-jsonl", type=Path)
    parser.add_argument("--output-manifest", type=Path)
    parser.add_argument("--run-id", default="xperience10m_window_index")
    return parser.parse_args()


def load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def compact_media(media: dict[str, Any]) -> dict[str, Any]:
    return {
        "video_streams": media.get("video_paths", []),
        "mosaic_video_path": media.get("mosaic_video_path"),
        "audio_path": media.get("audio_path"),
        "context_window": {
            "start_frame": media.get("context_start_frame"),
            "end_frame": media.get("context_end_frame"),
            "max_video_frames": media.get("max_video_frames"),
        },
    }


def supervision(answer: dict[str, Any]) -> dict[str, Any]:
    return {
        "json_answer": answer,
        "action": answer.get("action", "unknown"),
        "subtask": answer.get("subtask", "unknown"),
        "objects": answer.get("objects", []),
        "contact": answer.get("contact", "unknown"),
        "transition": answer.get("transition", "unknown"),
        "next_action": answer.get("next_action", "unknown"),
        "evidence_window": answer.get("evidence_window", {}),
    }


def adapter_views(row: dict[str, Any], answer: dict[str, Any]) -> dict[str, Any]:
    media = row.get("media", {})
    center = row.get("center_window", {})
    return {
        "qwen3_omni_lora": {
            "sample_contract": "video/audio/text/messages -> strict JSON answer",
            "messages_available": bool(row.get("messages")),
            "direct_modalities": ["mosaic_video", "audio", "language_prompt"],
            "bridged_modalities": ["sensor_features_npz"],
        },
        "cosmos_world_model": {
            "sample_contract": "context window + conditioning -> future state target",
            "context_window": {
                "start_frame": media.get("context_start_frame"),
                "end_frame": media.get("context_end_frame"),
            },
            "available_conditioning": [
                name
                for name, present in {
                    "video": bool(media.get("mosaic_video_path") or media.get("video_paths")),
                    "audio": bool(media.get("audio_path")),
                    "sensor_features": bool(row.get("sensor_feature_path")),
                    "language": bool(row.get("question")),
                    "action_label": answer.get("action") not in (None, "unknown"),
                    "next_action_label": answer.get("next_action") not in (None, "unknown"),
                }.items()
                if present
            ],
            "candidate_targets": [
                "future_window",
                "future_sensor_features",
                "transition",
                "contact",
                "next_action",
            ],
            "requires_new_exporter": True,
        },
        "policy_vla_branch": {
            "sample_contract": "observation + instruction/context -> action or motion target",
            "observation_window": center,
            "candidate_targets": [
                "action",
                "next_action",
                "contact",
                "objects",
                "hand_trajectory_chunk",
                "retargeted_action_token",
            ],
            "requires_action_space_design": True,
        },
    }


def to_neutral_record(row: dict[str, Any]) -> dict[str, Any]:
    answer = row.get("answer_json") or {}
    return {
        "id": row.get("id"),
        "source_record_id": row.get("id"),
        "episode_id": row.get("episode_id"),
        "split": row.get("split", "unspecified"),
        "target": row.get("target"),
        "prompt_type": row.get("prompt_type"),
        "window": row.get("center_window", {}),
        "modalities": {
            "media": compact_media(row.get("media", {})),
            "sensor_features": {
                "path": row.get("sensor_feature_path"),
                "index": row.get("sensor_feature_index"),
                "dim": row.get("sensor_feature_dim"),
            },
            "language": {
                "question": row.get("question"),
                "action_options": row.get("action_options", []),
                "subtask_options": row.get("subtask_options", []),
            },
        },
        "supervision": supervision(answer),
        "adapter_views": adapter_views(row, answer),
        "provenance": {
            "dataset_record_fields": sorted(row.keys()),
            "parallel_export_shard": row.get("parallel_export_shard"),
        },
    }


def main() -> int:
    args = parse_args()
    dataset_path = args.dataset_jsonl.expanduser().resolve()
    dataset_manifest_path = args.dataset_manifest
    if dataset_manifest_path is None:
        candidate = dataset_path.parent / "dataset_manifest.json"
        dataset_manifest_path = candidate if candidate.exists() else None
    if args.output_jsonl is None:
        args.output_jsonl = dataset_path.parent / "window_index.jsonl"
    if args.output_manifest is None:
        args.output_manifest = dataset_path.parent / "window_index_manifest.json"

    rows = load_jsonl(dataset_path)
    neutral_rows = [to_neutral_record(row) for row in rows]
    write_jsonl(args.output_jsonl, neutral_rows)

    split_counts = Counter(str(row.get("split", "unspecified")) for row in neutral_rows)
    episodes_by_split: dict[str, set[str]] = defaultdict(set)
    modality_counts = Counter()
    for row in neutral_rows:
        split = str(row.get("split", "unspecified"))
        episodes_by_split[split].add(str(row.get("episode_id")))
        media = row["modalities"]["media"]
        sensor = row["modalities"]["sensor_features"]
        if media.get("mosaic_video_path"):
            modality_counts["mosaic_video"] += 1
        if media.get("audio_path"):
            modality_counts["audio"] += 1
        if sensor.get("path") is not None:
            modality_counts["sensor_features"] += 1
        if row["modalities"]["language"].get("question"):
            modality_counts["language"] += 1

    manifest = {
        "run_id": args.run_id,
        "source_dataset_jsonl": str(dataset_path),
        "source_dataset_manifest": str(dataset_manifest_path) if dataset_manifest_path else None,
        "output_jsonl": str(args.output_jsonl),
        "num_records": len(neutral_rows),
        "num_episodes": len({row.get("episode_id") for row in neutral_rows}),
        "sample_split_counts": dict(split_counts),
        "episode_split_counts": {split: len(values) for split, values in episodes_by_split.items()},
        "modality_record_counts": dict(modality_counts),
        "adapter_contracts": {
            "qwen3_omni_lora": "implemented by current JSON-QA train/eval scripts",
            "cosmos_world_model": "requires future-window target exporter and world-model evaluator",
            "policy_vla_branch": "requires action-space conversion and policy evaluator",
        },
        "source_dataset_summary": load_json(dataset_manifest_path),
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

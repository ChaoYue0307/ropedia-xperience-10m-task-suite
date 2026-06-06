#!/usr/bin/env python3
"""Export same-episode future-window pairs for the Cosmos3 world-model branch."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    workspace_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=workspace_default)
    parser.add_argument("--source-dataset-jsonl", type=Path, required=True)
    parser.add_argument("--source-dataset-manifest", type=Path)
    parser.add_argument("--run-id", default="xperience10m_cosmos3_future_window_dataset")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--horizon-windows", type=int, default=5)
    parser.add_argument("--max-pairs-per-episode", type=int, default=0)
    parser.add_argument("--cosmos-model-dir", type=Path)
    return parser.parse_args()


def load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def window_start(row: dict[str, Any]) -> int:
    return int((row.get("center_window") or {}).get("start_frame", 0) or 0)


def compact_media(row: dict[str, Any]) -> dict[str, Any]:
    media = row.get("media") or {}
    return {
        "mosaic_video_path": media.get("mosaic_video_path"),
        "audio_path": media.get("audio_path"),
        "video_paths": media.get("video_paths", []),
        "context_start_frame": media.get("context_start_frame"),
        "context_end_frame": media.get("context_end_frame"),
        "max_video_frames": media.get("max_video_frames"),
    }


def answer(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("answer_json") or {
        "action": row.get("label", "unknown"),
        "subtask": "unknown",
        "objects": [],
        "contact": "unknown",
        "transition": "unknown",
        "next_action": "unknown",
        "evidence_window": row.get("center_window", {}),
    }


def make_pair(current: dict[str, Any], future: dict[str, Any], horizon: int, ordinal: int) -> dict[str, Any]:
    current_answer = answer(current)
    future_answer = answer(future)
    pair_id = f"{current.get('episode_id')}:future:{ordinal:05d}:h{horizon}"
    return {
        "id": pair_id,
        "episode_id": current.get("episode_id"),
        "split": current.get("split", "unspecified"),
        "target": "future_window_world_model",
        "prompt_type": "cosmos3_future_window",
        "window": current.get("center_window", {}),
        "future_window": future.get("center_window", {}),
        "horizon_windows": int(horizon),
        "context_record_id": current.get("id"),
        "future_record_id": future.get("id"),
        "media": compact_media(current),
        "future_media": compact_media(future),
        "sensor_feature_path": current.get("sensor_feature_path"),
        "sensor_feature_index": current.get("sensor_feature_index"),
        "future_sensor_feature_path": future.get("sensor_feature_path"),
        "future_sensor_feature_index": future.get("sensor_feature_index"),
        "sensor_feature_dim": current.get("sensor_feature_dim"),
        "conditioning": {
            "question": current.get("question"),
            "action": current_answer.get("action", "unknown"),
            "subtask": current_answer.get("subtask", "unknown"),
            "objects": current_answer.get("objects", []),
            "contact": current_answer.get("contact", "unknown"),
            "transition": current_answer.get("transition", "unknown"),
            "next_action": current_answer.get("next_action", "unknown"),
        },
        "future_target": {
            "action": future_answer.get("action", "unknown"),
            "subtask": future_answer.get("subtask", "unknown"),
            "objects": future_answer.get("objects", []),
            "contact": future_answer.get("contact", "unknown"),
            "transition": future_answer.get("transition", "unknown"),
            "next_action": future_answer.get("next_action", "unknown"),
        },
        # Compatibility fields keep the shared run validator and package scripts
        # usable while this branch moves beyond Qwen-style JSON QA records.
        "answer_json": future_answer,
        "messages": [],
        "label_options": current.get("label_options", []),
        "adapter_views": {
            "cosmos3_nano": {
                "sample_contract": "context video/audio/text/action metadata -> future window target",
                "target_modalities": ["future_sensor_features", "future_action", "future_contact", "future_transition"],
                "model_role": "Cosmos3-Nano compatibility and future-window world-model branch",
            }
        },
    }


def model_metadata(model_dir: Path | None) -> dict[str, Any]:
    if model_dir is None:
        return {"available": False, "reason": "cosmos model dir not provided"}
    model_dir = model_dir.expanduser()
    files = {
        "config.json": model_dir / "config.json",
        "model_index.json": model_dir / "model_index.json",
        "generation_config.json": model_dir / "generation_config.json",
    }
    payload: dict[str, Any] = {
        "available": model_dir.exists(),
        "path": str(model_dir),
        "files": {name: path.exists() for name, path in files.items()},
    }
    for name, path in files.items():
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if name == "model_index.json":
            payload["pipeline_class"] = data.get("_class_name")
            payload["diffusers_version"] = data.get("_diffusers_version")
        if name == "config.json":
            payload["architectures"] = data.get("architectures")
            cfg = ((data.get("model") or {}).get("config") or {})
            payload["lora_enabled_default"] = cfg.get("lora_enabled")
            payload["lora_rank_default"] = cfg.get("lora_rank")
            payload["lora_target_modules_default"] = cfg.get("lora_target_modules")
            payload["resolution"] = cfg.get("resolution")
    return payload


def main() -> int:
    args = parse_args()
    args.workspace = args.workspace.expanduser().resolve()
    root = args.workspace / "results" / "omni_finetune"
    if args.run_dir is None:
        args.run_dir = root / args.run_id
    if args.output_dir is None:
        args.output_dir = root / f"{args.run_id}_dataset"
    if args.source_dataset_manifest is None:
        candidate = args.source_dataset_jsonl.parent / "dataset_manifest.json"
        args.source_dataset_manifest = candidate if candidate.exists() else None

    source_rows = load_jsonl(args.source_dataset_jsonl)
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        by_episode[str(row.get("episode_id"))].append(row)

    pairs: list[dict[str, Any]] = []
    skipped_by_episode: dict[str, int] = {}
    for episode_id, rows in sorted(by_episode.items()):
        rows = sorted(rows, key=window_start)
        episode_pairs = []
        for idx, current in enumerate(rows):
            future_idx = idx + args.horizon_windows
            if future_idx >= len(rows):
                continue
            future = rows[future_idx]
            if future.get("split") != current.get("split"):
                continue
            if future.get("episode_id") != current.get("episode_id"):
                continue
            episode_pairs.append(make_pair(current, future, args.horizon_windows, len(episode_pairs)))
        if args.max_pairs_per_episode > 0:
            episode_pairs = episode_pairs[: args.max_pairs_per_episode]
        if not episode_pairs:
            skipped_by_episode[episode_id] = len(rows)
        pairs.extend(episode_pairs)

    if not pairs:
        raise ValueError("No future-window pairs were exported. Check horizon and source dataset.")

    split_counts = Counter(str(row.get("split", "unspecified")) for row in pairs)
    episodes_by_split: dict[str, set[str]] = defaultdict(set)
    for row in pairs:
        episodes_by_split[str(row.get("split", "unspecified"))].add(str(row.get("episode_id")))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    dataset_jsonl = args.output_dir / "dataset.jsonl"
    write_jsonl(dataset_jsonl, pairs)

    source_manifest = load_json(args.source_dataset_manifest)
    episode_manifest = {
        "run_id": args.run_id,
        "source_dataset_jsonl": str(args.source_dataset_jsonl),
        "episodes": [
            {
                "episode_id": episode_id,
                "split": sorted({str(row.get("split", "unspecified")) for row in rows})[0],
                "source_record_count": len(rows),
            }
            for episode_id, rows in sorted(by_episode.items())
        ],
    }
    write_json(args.run_dir / "episode_manifest.json", episode_manifest)

    dataset_manifest = {
        "run_id": args.run_id,
        "dataset_path": str(dataset_jsonl),
        "source_dataset_jsonl": str(args.source_dataset_jsonl),
        "source_dataset_manifest": str(args.source_dataset_manifest) if args.source_dataset_manifest else None,
        "dataset_contract": "xperience10m_future_window_world_model_v0",
        "num_samples": len(pairs),
        "num_episodes": len({row.get("episode_id") for row in pairs}),
        "split_counts": dict(split_counts),
        "episode_split_counts": {split: len(values) for split, values in episodes_by_split.items()},
        "horizon_windows": int(args.horizon_windows),
        "max_pairs_per_episode": int(args.max_pairs_per_episode),
        "cosmos_model": model_metadata(args.cosmos_model_dir),
        "source_dataset_summary": {
            "num_samples": source_manifest.get("num_samples"),
            "num_episodes": source_manifest.get("num_episodes"),
            "split_counts": source_manifest.get("split_counts"),
        },
        "skipped_episodes": [
            {"episode_id": episode_id, "source_record_count": count, "reason": "no future pair at requested horizon"}
            for episode_id, count in sorted(skipped_by_episode.items())
        ],
        "notes": [
            "Records contain derived paths and labels only; raw media is referenced for local training but not copied into public packages.",
            "The target is a same-episode future window, preserving train/val/test episode boundaries.",
            "This is the Cosmos3-Nano compatibility dataset used before full diffusion fine-tuning.",
        ],
    }
    write_json(args.output_dir / "dataset_manifest.json", dataset_manifest)
    print(json.dumps(dataset_manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

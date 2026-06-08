#!/usr/bin/env python3
"""Merge Qwen3-Omni per-scale exports into one multiscale training JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from qwen3_omni_dataset_utils import build_messages, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", action="append", required=True, help="scale_id=/path/to/dataset_dir")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def parse_components(values: list[str]) -> list[tuple[str, Path]]:
    components = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Bad --component {value!r}; expected scale_id=/path/to/dataset_dir")
        scale_id, path = value.split("=", 1)
        components.append((scale_id, Path(path).expanduser().resolve()))
    return components


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def collect_options(components: list[tuple[str, Path]]) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    actions = set()
    subtasks = set()
    manifests = []
    for scale_id, dataset_dir in components:
        manifest = load_manifest(dataset_dir / "dataset_manifest.json")
        manifests.append({"scale_id": scale_id, "dataset_dir": str(dataset_dir), "manifest": manifest})
        for row in iter_jsonl(dataset_dir / "dataset.jsonl"):
            answer = row.get("answer_json") or {}
            action = str(answer.get("action") or "").strip()
            subtask = str(answer.get("subtask") or "").strip()
            if action and action != "unknown":
                actions.add(action)
            if subtask and subtask != "unknown":
                subtasks.add(subtask)
    return sorted(actions), sorted(subtasks), manifests


def write_merged(components: list[tuple[str, Path]], output_dir: Path, run_id: str, action_options: list[str], subtask_options: list[str]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "dataset.jsonl"
    counts = Counter()
    scale_counts = Counter()
    episode_ids = set()
    records_written = 0
    with dataset_path.open("w", encoding="utf-8") as handle:
        for scale_id, dataset_dir in components:
            for row in iter_jsonl(dataset_dir / "dataset.jsonl"):
                original_id = row.get("id")
                row["id"] = f"{scale_id}:{original_id}"
                row["source_row_id"] = original_id
                row["multiscale_run_id"] = run_id
                row["scale_id"] = scale_id
                row["action_options"] = action_options
                row["subtask_options"] = subtask_options
                row["label_options"] = action_options
                center_window = row.get("center_window") or {}
                question = row.get("question") or "Given the synchronized egocentric video/audio context and sensor window, identify the current embodied episode state."
                row["question"] = (
                    f"{question} This record belongs to multiscale window scale {scale_id}; "
                    f"label window frames {center_window.get('start_frame', 'unknown')}-{center_window.get('end_frame', 'unknown')}."
                )
                row["messages"] = build_messages(row, action_options, include_answer=True)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[row.get("split", "unspecified")] += 1
                scale_counts[scale_id] += 1
                episode_ids.add(row.get("episode_id"))
                records_written += 1
    return {
        "dataset_path": str(dataset_path),
        "num_samples": records_written,
        "num_episodes": len(episode_ids),
        "split_counts": dict(sorted(counts.items())),
        "scale_counts": dict(sorted(scale_counts.items())),
    }


def main() -> int:
    args = parse_args()
    components = parse_components(args.component)
    for scale_id, dataset_dir in components:
        dataset_jsonl = dataset_dir / "dataset.jsonl"
        if not dataset_jsonl.exists():
            raise FileNotFoundError(f"{scale_id}: missing {dataset_jsonl}")
    action_options, subtask_options, component_manifests = collect_options(components)
    merged = write_merged(components, args.output_dir, args.run_id, action_options, subtask_options)
    manifest = {
        "run_id": args.run_id,
        **merged,
        "action_option_count": len(action_options),
        "subtask_option_count": len(subtask_options),
        "components": component_manifests,
        "clip_policy": {
            "multiscale": True,
            "component_scales": [scale_id for scale_id, _path in components],
            "source_policy": "raw-media per-scale exports merged without copying base media; row paths point to component shard media",
        },
    }
    (args.output_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "config.yaml").write_text(
        "\n".join([
            f"run_id: {args.run_id}",
            "objective: qwen3_omni_multiscale_episode_understanding_json_qa",
            "components:",
            *[f"  - {scale_id}: {path}" for scale_id, path in components],
        ]) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build compact dense/multiscale hierarchical labels for the 128-episode suite.

This builder uses the verified 128-episode JSON-QA export as a sparse label
source and creates a public-safe compact window index. It does not copy raw
media, does not rewrite the sealed 3,808-window dataset, and records label
provenance for every generated dense window.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "tmp/omni_128_dataset_fetch/dataset.jsonl"
DEFAULT_FALLBACK_INPUT = (
    ROOT
    / "results/omni_finetune/xperience10m_qwen3_omni_128ep_96train_16val_16test_valmon_20260605_dataset/dataset.jsonl"
)
DEFAULT_RUN_ID = "xperience10m_128ep_dense_multiscale_hierarchical_v1_20260608"
DEFAULT_OUTPUT_ROOT = ROOT / "results/omni_finetune"

ACTION_FAMILY_PATTERNS = [
    ("locomotion", ["walk", "enter", "approach", "move through", "move towards", "arrive", "leave"]),
    ("reach_grasp_release", ["reach", "grasp", "pick up", "hold", "release", "grab", "take"]),
    ("place_arrange_align", ["place", "arrange", "align", "position", "set", "stack", "insert"]),
    ("manipulate_adjust", ["manipulate", "adjust", "bend", "fold", "open", "close", "secure", "press"]),
    ("tool_cut_mark_write", ["cut", "mark", "draw", "write", "scissor", "knife", "pen", "marker", "ruler"]),
    ("sort_count_organize", ["sort", "count", "organize", "bundle", "gather", "separate", "group"]),
    ("inspect_observe_use", ["inspect", "observe", "browse", "use", "operate", "check", "examine", "scan"]),
    ("clean_cook", ["clean", "wipe", "rinse", "wash", "stir", "cook", "pot", "stove"]),
    ("assemble_craft", ["assemble", "attach", "tape", "glue", "thread", "tie", "wrap"]),
]

SUBTASK_FAMILY_PATTERNS = [
    ("inventory_restocking", ["shelf", "box", "retriev", "restock", "item", "container", "canned", "retail"]),
    ("craft_assembly", ["cardboard", "paper", "lantern", "foam", "assemble", "craft", "cutting"]),
    ("tabletop_sorting", ["mahjong", "tile", "puzzle", "button", "bead", "sort", "count"]),
    ("household_clean_cook", ["clean", "cook", "wash", "pot", "stove", "cloth"]),
    ("tool_use_measurement", ["ruler", "mark", "draw", "write", "scissor", "knife"]),
    ("navigation_setup", ["approach", "workstation", "desk", "table", "packing"]),
]


@dataclass(frozen=True)
class Scale:
    scale_id: str
    window_frames: int
    stride_frames: int


DEFAULT_SCALES = [
    Scale("dense_20f_stride10", 20, 10),
    Scale("medium_40f_stride20", 40, 20),
    Scale("long_80f_stride40", 80, 40),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--scale",
        action="append",
        help="Scale spec as id:window_frames:stride_frames. Defaults to the recommended multiscale 20s10/40s20/80s40 set.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output directory.")
    return parser.parse_args()


def resolve_input(path: Path) -> Path:
    if path.exists():
        return path
    if path == DEFAULT_INPUT and DEFAULT_FALLBACK_INPUT.exists():
        return DEFAULT_FALLBACK_INPUT
    raise FileNotFoundError(path)


def parse_scales(values: list[str] | None) -> list[Scale]:
    if not values:
        return DEFAULT_SCALES
    scales: list[Scale] = []
    for value in values:
        parts = value.split(":")
        if len(parts) != 3:
            raise ValueError(f"Bad --scale spec {value!r}; expected id:window_frames:stride_frames")
        scale_id, window, stride = parts
        scales.append(Scale(scale_id, int(window), int(stride)))
    return scales


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def normalize(value: Any) -> str:
    return str(value or "").strip()


def answer_json(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("answer_json")
    if isinstance(value, dict):
        return value
    value = row.get("true_json")
    return value if isinstance(value, dict) else {}


def family_from_patterns(label: str, patterns: list[tuple[str, list[str]]], fallback: str) -> str:
    text = normalize(label).lower()
    if not text or text == "unknown":
        return "unknown"
    for family, needles in patterns:
        if any(needle in text for needle in needles):
            return family
    return fallback


def action_family(label: str) -> str:
    return family_from_patterns(label, ACTION_FAMILY_PATTERNS, "other_fine_action")


def subtask_family(label: str) -> str:
    return family_from_patterns(label, SUBTASK_FAMILY_PATTERNS, "other_fine_subtask")


def compact_source(row: dict[str, Any]) -> dict[str, Any]:
    window = row.get("center_window") or {}
    media = row.get("media") or {}
    video_names = []
    for item in media.get("video_paths") or []:
        if isinstance(item, dict) and item.get("name"):
            video_names.append(str(item["name"]))
    answer = answer_json(row)
    action = normalize(answer.get("action") or row.get("label") or "unknown") or "unknown"
    subtask = normalize(answer.get("subtask") or "unknown") or "unknown"
    next_action = normalize(answer.get("next_action") or "unknown") or "unknown"
    return {
        "id": row.get("id"),
        "episode_id": row.get("episode_id"),
        "split": row.get("split"),
        "start_frame": int(window.get("start_frame", 0) or 0),
        "end_frame": int(window.get("end_frame", 0) or 0),
        "sensor_feature_index": int(row.get("sensor_feature_index", 0) or 0),
        "sensor_feature_dim": int(row.get("sensor_feature_dim", 0) or 0),
        "media_names": sorted(set(video_names)),
        "has_mosaic": bool(media.get("mosaic_video_path")),
        "has_audio": bool(media.get("audio_path")),
        "answer_json": {
            "action": action,
            "subtask": subtask,
            "objects": list(answer.get("objects") or [])[:8],
            "contact": normalize(answer.get("contact") or "unknown").lower() or "unknown",
            "transition": normalize(answer.get("transition") or "unknown").lower() or "unknown",
            "next_action": next_action,
            "evidence_window": {
                "start_frame": int((answer.get("evidence_window") or {}).get("start_frame", window.get("start_frame", 0)) or 0),
                "end_frame": int((answer.get("evidence_window") or {}).get("end_frame", window.get("end_frame", 0)) or 0),
            },
        },
    }


def load_sources(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(compact_source(json.loads(line)))
    rows.sort(key=lambda row: (row["episode_id"], row["start_frame"], row["end_frame"]))
    return rows


def interval_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start) + 1)


def nearest_source(sources: list[dict[str, Any]], starts: list[int], start: int, end: int) -> dict[str, Any]:
    center = (start + end) / 2.0
    pos = bisect.bisect_left(starts, start)
    candidates = []
    for idx in range(max(0, pos - 4), min(len(sources), pos + 5)):
        src = sources[idx]
        src_center = (src["start_frame"] + src["end_frame"]) / 2.0
        overlap = interval_overlap(start, end, src["start_frame"], src["end_frame"])
        candidates.append((abs(src_center - center), -overlap, idx, src))
    if not candidates:
        raise ValueError("No sparse source rows for episode")
    return min(candidates)[3]


def provenance(source: dict[str, Any], start: int, end: int) -> dict[str, Any]:
    overlap = interval_overlap(start, end, source["start_frame"], source["end_frame"])
    dense_len = max(end - start + 1, 1)
    source_len = max(source["end_frame"] - source["start_frame"] + 1, 1)
    center = (start + end) / 2.0
    source_center = (source["start_frame"] + source["end_frame"]) / 2.0
    if source["start_frame"] == start and source["end_frame"] == end:
        kind = "exact_sparse_window"
    elif overlap:
        kind = "overlap_nearest_sparse_window"
    else:
        kind = "gap_nearest_sparse_window"
    return {
        "kind": kind,
        "source_window_id": source["id"],
        "source_start_frame": source["start_frame"],
        "source_end_frame": source["end_frame"],
        "center_distance_frames": abs(source_center - center),
        "overlap_frames": overlap,
        "overlap_fraction_of_dense": round(overlap / dense_len, 6),
        "overlap_fraction_of_source": round(overlap / source_len, 6),
    }


def build_dense_records(sources: list[dict[str, Any]], scales: list[Scale], run_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sources:
        by_episode[row["episode_id"]].append(row)

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for episode_id, episode_rows in sorted(by_episode.items()):
        episode_rows.sort(key=lambda row: row["start_frame"])
        starts = [row["start_frame"] for row in episode_rows]
        split = str(episode_rows[0]["split"])
        episode_max_end = max(row["end_frame"] for row in episode_rows)
        for scale in scales:
            if episode_max_end + 1 < scale.window_frames:
                skipped.append({
                    "episode_id": episode_id,
                    "split": split,
                    "scale_id": scale.scale_id,
                    "reason": "episode shorter than requested window",
                })
                continue
            for start in range(0, episode_max_end - scale.window_frames + 2, scale.stride_frames):
                end = start + scale.window_frames - 1
                source = nearest_source(episode_rows, starts, start, end)
                answer = dict(source["answer_json"])
                action = normalize(answer.get("action") or "unknown") or "unknown"
                subtask = normalize(answer.get("subtask") or "unknown") or "unknown"
                next_action = normalize(answer.get("next_action") or "unknown") or "unknown"
                labels = {
                    "action": action,
                    "action_family": action_family(action),
                    "subtask": subtask,
                    "subtask_family": subtask_family(subtask),
                    "next_action": next_action,
                    "next_action_family": action_family(next_action),
                    "contact": normalize(answer.get("contact") or "unknown").lower() or "unknown",
                    "transition": normalize(answer.get("transition") or "unknown").lower() or "unknown",
                }
                dense_id = f"{episode_id}:ms:{scale.scale_id}:{start:06d}_{end:06d}"
                rows.append({
                    "id": dense_id,
                    "run_id": run_id,
                    "episode_id": episode_id,
                    "split": split,
                    "scale_id": scale.scale_id,
                    "window_frames": scale.window_frames,
                    "stride_frames": scale.stride_frames,
                    "center_window": {
                        "start_frame": start,
                        "end_frame": end,
                        "num_frames": scale.window_frames,
                    },
                    "source_sparse_window": provenance(source, start, end),
                    "labels": labels,
                    "objects": answer.get("objects") or [],
                    "sensor_feature_dim": source["sensor_feature_dim"],
                    "source_media_names": source["media_names"],
                    "source_has_mosaic": source["has_mosaic"],
                    "source_has_audio": source["has_audio"],
                })

    summary = {
        "num_seed_windows": len(sources),
        "num_dense_windows": len(rows),
        "num_episodes": len(by_episode),
        "split_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
        "scale_counts": dict(sorted(Counter(row["scale_id"] for row in rows).items())),
        "label_provenance_counts": dict(sorted(Counter(row["source_sparse_window"]["kind"] for row in rows).items())),
        "episode_split_counts": {
            split: len({row["episode_id"] for row in rows if row["split"] == split})
            for split in sorted({row["split"] for row in rows})
        },
        "skipped": skipped,
    }
    return rows, summary


def label_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    targets = ["action_family", "action", "subtask_family", "subtask", "next_action_family", "next_action", "contact", "transition"]
    payload: dict[str, Any] = {}
    for target in targets:
        by_split = {split: Counter() for split in ("train", "val", "test")}
        for row in rows:
            split = row["split"]
            if split in by_split:
                by_split[split][row["labels"].get(target, "unknown")] += 1
        train_labels = set(by_split["train"])
        payload[target] = {
            "num_labels": len(set().union(*(set(counter) for counter in by_split.values()))),
            "num_train_labels": len(train_labels),
            "split_top_labels": {
                split: [{"label": label, "count": int(count)} for label, count in counter.most_common(20)]
                for split, counter in by_split.items()
            },
            "unseen_val_labels": sorted(set(by_split["val"]) - train_labels),
            "unseen_test_labels": sorted(set(by_split["test"]) - train_labels),
        }
    return payload


def write_report(path: Path, manifest: dict[str, Any], stats: dict[str, Any]) -> None:
    lines = [
        f"# Dense/Multiscale Hierarchical 128-Episode Dataset",
        "",
        f"Run id: `{manifest['run_id']}`",
        "",
        "This compact dataset expands the current 128-episode sparse export without adding raw episodes.",
        "Labels for newly generated dense windows are inherited from the nearest sparse labeled window and every row records that provenance.",
        "",
        "## Counts",
        "",
        f"- Seed sparse windows: `{manifest['source']['num_seed_windows']}`",
        f"- Dense/multiscale windows: `{manifest['num_samples']}`",
        f"- Episodes: `{manifest['num_episodes']}`",
        f"- Split counts: `{manifest['split_counts']}`",
        f"- Scale counts: `{manifest['scale_counts']}`",
        f"- Label provenance: `{manifest['label_provenance_counts']}`",
        "",
        "## Target Families",
        "",
        "| target | labels | train labels | unseen test labels |",
        "| --- | ---: | ---: | ---: |",
    ]
    for target, payload in stats.items():
        lines.append(
            f"| `{target}` | {payload['num_labels']} | {payload['num_train_labels']} | {len(payload['unseen_test_labels'])} |"
        )
    lines.extend([
        "",
        "## Files",
        "",
        "- `dense_multiscale_windows.jsonl`: compact row-level windows and labels.",
        "- `dataset_manifest.json`: counts, scale contract, and provenance policy.",
        "- `hierarchical_label_stats.json`: label cardinality and unseen-label diagnostics.",
        "- `split_scale_counts.csv`: split/scale window counts.",
        "",
        "This package is intended for fast baselines and for planning the heavier raw-media Qwen/Cosmos runs.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_jsonl = resolve_input(args.input_jsonl.expanduser())
    output_dir = args.output_dir or (args.output_root / args.run_id)
    output_dir = output_dir.expanduser()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise FileExistsError(f"Refusing to overwrite non-empty output dir without --force: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    scales = parse_scales(args.scale)

    sources = load_sources(input_jsonl)
    rows, summary = build_dense_records(sources, scales, args.run_id)
    stats = label_stats(rows)
    scale_contract = [{"scale_id": s.scale_id, "window_frames": s.window_frames, "stride_frames": s.stride_frames} for s in scales]
    manifest = {
        "run_id": args.run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": "dense_multiscale_windows.jsonl",
        "source": {
            "dataset_jsonl": str(input_jsonl),
            "num_seed_windows": summary["num_seed_windows"],
            "policy": "nearest sparse-window label inheritance with row-level provenance; not a replacement for raw human frame labels",
        },
        "scale_contract": scale_contract,
        "num_samples": summary["num_dense_windows"],
        "num_episodes": summary["num_episodes"],
        "split_counts": summary["split_counts"],
        "episode_split_counts": summary["episode_split_counts"],
        "scale_counts": summary["scale_counts"],
        "label_provenance_counts": summary["label_provenance_counts"],
        "skipped": summary["skipped"],
        "public_safe": {
            "copies_raw_media": False,
            "copies_local_paths": False,
            "contains_adapter_weights": False,
            "contains_base_model_weights": False,
        },
    }
    write_jsonl(output_dir / "dense_multiscale_windows.jsonl", rows)
    write_json(output_dir / "dataset_manifest.json", manifest)
    write_json(output_dir / "hierarchical_label_stats.json", stats)
    split_scale_rows = []
    for (split, scale_id), count in sorted(Counter((row["split"], row["scale_id"]) for row in rows).items()):
        split_scale_rows.append({"split": split, "scale_id": scale_id, "windows": int(count)})
    write_csv(output_dir / "split_scale_counts.csv", split_scale_rows)
    write_report(output_dir / "RUN_REPORT.md", manifest, stats)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

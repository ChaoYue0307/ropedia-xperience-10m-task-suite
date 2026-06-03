#!/usr/bin/env python3
"""Analyze the gated Xperience-10M HF repo without downloading dataset files."""

from __future__ import annotations

import argparse
import getpass
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

from huggingface_hub import HfApi


REQUIRED_EPISODE_FILES = [
    "annotation.hdf5",
    "fisheye_cam0.mp4",
    "fisheye_cam1.mp4",
    "fisheye_cam2.mp4",
    "fisheye_cam3.mp4",
    "stereo_left.mp4",
    "stereo_right.mp4",
]
TRAINING_EXCLUDE = {"visualization.rrd"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="ropedia-ai/xperience-10m")
    parser.add_argument("--output", type=Path, default=Path("results/omni_finetune/full_dataset_metadata_audit.json"))
    parser.add_argument("--report-output", type=Path, default=Path("results/omni_finetune/FULL_DATASET_METADATA_AUDIT.md"))
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN", "").strip())
    parser.add_argument("--top-n", type=int, default=20)
    return parser.parse_args()


def file_size(sibling: Any) -> int:
    value = getattr(sibling, "size", None)
    if isinstance(value, int):
        return value
    lfs = getattr(sibling, "lfs", None)
    if isinstance(lfs, dict) and isinstance(lfs.get("size"), int):
        return int(lfs["size"])
    return 0


def human_bytes(num: float | int) -> str:
    value = float(num)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]:
        if abs(value) < 1024.0 or unit == "PiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} PiB"


def pct(part: int, whole: int) -> float:
    return round((part / whole * 100.0), 4) if whole else 0.0


def episode_parent(path: str) -> str:
    return str(Path(path).parent).replace("\\", "/")


def summarize_sizes(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    q1 = ordered[len(ordered) // 4]
    q3 = ordered[(len(ordered) * 3) // 4]
    return {
        "count": len(values),
        "min_bytes": ordered[0],
        "p25_bytes": q1,
        "median_bytes": int(median(ordered)),
        "p75_bytes": q3,
        "max_bytes": ordered[-1],
        "mean_bytes": int(sum(values) / len(values)),
        "min_human": human_bytes(ordered[0]),
        "median_human": human_bytes(median(ordered)),
        "mean_human": human_bytes(sum(values) / len(values)),
        "max_human": human_bytes(ordered[-1]),
    }


def near_size_files(files: list[dict[str, Any]], target: int, count: int) -> list[dict[str, Any]]:
    ranked = sorted(files, key=lambda item: abs(int(item["bytes"]) - target))
    return ranked[:count]


def summarize_numbers(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(values),
        "min": ordered[0],
        "p25": ordered[len(ordered) // 4],
        "median": int(median(ordered)),
        "p75": ordered[(len(ordered) * 3) // 4],
        "max": ordered[-1],
        "mean": round(sum(values) / len(values), 2),
    }


def md_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return lines


def main() -> int:
    args = parse_args()
    token = args.token or getpass.getpass("HF token: ").strip()
    if not token:
        raise SystemExit("HF token is required for gated dataset metadata.")

    api = HfApi(token=token)
    info = api.repo_info(
        repo_id=args.repo_id,
        repo_type="dataset",
        files_metadata=True,
        token=token,
    )
    siblings = list(info.siblings or [])

    files = []
    total_bytes = 0
    ext_counter: Counter[str] = Counter()
    basename_counter: Counter[str] = Counter()
    top_level_counter: Counter[str] = Counter()
    by_parent: dict[str, dict[str, Any]] = defaultdict(lambda: {"files": {}, "bytes": 0})
    by_top_level_bytes: Counter[str] = Counter()

    for sibling in siblings:
        path = str(getattr(sibling, "rfilename", ""))
        if not path or path == ".gitattributes":
            continue
        size = file_size(sibling)
        total_bytes += size
        ext = Path(path).suffix.lower() or "<no_ext>"
        name = Path(path).name
        top = path.split("/", 1)[0]
        ext_counter[ext] += 1
        basename_counter[name] += 1
        top_level_counter[top] += 1
        by_top_level_bytes[top] += size
        parent = episode_parent(path)
        bucket = by_parent[parent]
        bucket["files"][name] = {"path": path, "bytes": size}
        bucket["bytes"] += size
        files.append({"path": path, "bytes": size, "extension": ext, "basename": name, "top_level": top})

    episode_records = []
    for parent, bucket in by_parent.items():
        present = set(bucket["files"])
        if not (present & set(REQUIRED_EPISODE_FILES)):
            continue
        has_annotation = "annotation.hdf5" in present
        has_fisheye_cam0 = "fisheye_cam0.mp4" in present
        video_count = sum(1 for name in REQUIRED_EPISODE_FILES[1:] if name in present)
        missing_required = [name for name in REQUIRED_EPISODE_FILES if name not in present]
        training_bytes = sum(
            meta["bytes"]
            for name, meta in bucket["files"].items()
            if name not in TRAINING_EXCLUDE
        )
        episode_records.append(
            {
                "episode_path": parent,
                "episode_id": Path(parent).name,
                "top_level_session": parent.split("/", 1)[0],
                "file_count": len(present),
                "total_bytes": int(bucket["bytes"]),
                "training_bytes_excluding_visualization_rrd": int(training_bytes),
                "has_annotation": has_annotation,
                "has_fisheye_cam0": has_fisheye_cam0,
                "video_count": video_count,
                "has_all_six_videos": video_count == 6,
                "is_degraded_valid": has_annotation and has_fisheye_cam0,
                "is_complete": has_annotation and video_count == 6,
                "has_visualization_rrd": "visualization.rrd" in present,
                "missing_required_files": missing_required,
            }
        )

    complete = [ep for ep in episode_records if ep["is_complete"]]
    degraded = [ep for ep in episode_records if ep["is_degraded_valid"]]
    incomplete = [ep for ep in episode_records if not ep["is_complete"]]
    training_sizes = [ep["training_bytes_excluding_visualization_rrd"] for ep in complete]
    episode_sizes = [ep["total_bytes"] for ep in episode_records]
    complete_by_session: Counter[str] = Counter(ep["top_level_session"] for ep in complete)
    degraded_by_session: Counter[str] = Counter(ep["top_level_session"] for ep in degraded)
    episode_count_by_session: Counter[str] = Counter(ep["top_level_session"] for ep in episode_records)
    video_count_hist = Counter(str(ep["video_count"]) for ep in episode_records)
    rrd_bytes = sum(item["bytes"] for item in files if item["basename"] == "visualization.rrd")
    all_complete_training_bytes = sum(ep["training_bytes_excluding_visualization_rrd"] for ep in complete)
    median_32_bytes = int(median(training_sizes)) * 32 if training_sizes else 0
    mean_32_bytes = int(sum(training_sizes) / len(training_sizes)) * 32 if training_sizes else 0

    largest_files = sorted(files, key=lambda item: item["bytes"], reverse=True)[: args.top_n]
    annotation_files = [item for item in files if item["basename"] == "annotation.hdf5"]
    annotation_sizes = [item["bytes"] for item in annotation_files]
    annotation_size_summary = summarize_sizes(annotation_sizes)
    annotation_median = int(annotation_size_summary.get("median_bytes", 0))
    largest_episodes = sorted(episode_records, key=lambda item: item["total_bytes"], reverse=True)[: args.top_n]
    smallest_complete = sorted(complete, key=lambda item: item["training_bytes_excluding_visualization_rrd"])[: args.top_n]

    selected_32 = []
    for session, _count in sorted(complete_by_session.items()):
        candidates = [ep for ep in complete if ep["top_level_session"] == session]
        candidates.sort(key=lambda ep: ep["training_bytes_excluding_visualization_rrd"])
        selected_32.append(candidates[0])
        if len(selected_32) == 32:
            break

    payload = {
        "status": "pass",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repo_id": args.repo_id,
        "repo_sha": getattr(info, "sha", None),
        "gated": getattr(info, "gated", None),
        "last_modified": getattr(info, "last_modified", None).isoformat() if getattr(info, "last_modified", None) else None,
        "card_data": getattr(info, "card_data", None).to_dict() if getattr(info, "card_data", None) and hasattr(getattr(info, "card_data", None), "to_dict") else None,
        "summary": {
            "sibling_count": len(siblings),
            "file_count_excluding_gitattributes": len(files),
            "total_bytes_from_file_metadata": total_bytes,
            "total_human_from_file_metadata": human_bytes(total_bytes),
            "training_bytes_excluding_visualization_rrd": total_bytes - rrd_bytes,
            "training_human_excluding_visualization_rrd": human_bytes(total_bytes - rrd_bytes),
            "visualization_rrd_bytes": rrd_bytes,
            "visualization_rrd_human": human_bytes(rrd_bytes),
            "top_level_session_count": len(top_level_counter),
            "episode_like_folder_count": len(episode_records),
            "annotation_hdf5_count": basename_counter["annotation.hdf5"],
            "mp4_count": sum(count for name, count in basename_counter.items() if name.endswith(".mp4")),
            "visualization_rrd_count": basename_counter["visualization.rrd"],
            "complete_episode_count": len(complete),
            "degraded_valid_episode_count": len(degraded),
            "complete_episode_pct": pct(len(complete), len(episode_records)),
            "degraded_valid_episode_pct": pct(len(degraded), len(episode_records)),
            "complete_sessions": len(complete_by_session),
            "degraded_valid_sessions": len(degraded_by_session),
            "all_complete_episode_training_bytes_excluding_visualization_rrd": all_complete_training_bytes,
            "all_complete_episode_training_human_excluding_visualization_rrd": human_bytes(all_complete_training_bytes),
        },
        "file_type_counts": dict(sorted(ext_counter.items())),
        "basename_counts": dict(sorted(basename_counter.items())),
        "video_count_histogram": dict(sorted(video_count_hist.items())),
        "episode_count_per_session_summary": summarize_numbers(list(episode_count_by_session.values())),
        "episode_size_summary": summarize_sizes(episode_sizes),
        "annotation_file_size_summary": annotation_size_summary,
        "complete_episode_training_size_summary": summarize_sizes(training_sizes),
        "incomplete_episode_records": incomplete,
        "pilot_scale_estimates": {
            "windows_per_episode": 256,
            "all_complete_episodes_windows_at_256_each": len(complete) * 256,
            "episode_32_windows_at_256_each": 32 * 256,
            "episode_100_windows_at_256_each": 100 * 256,
            "episode_500_windows_at_256_each": 500 * 256,
            "median_based_32_episode_training_bytes": median_32_bytes,
            "median_based_32_episode_training_human": human_bytes(median_32_bytes),
            "mean_based_32_episode_training_bytes": mean_32_bytes,
            "mean_based_32_episode_training_human": human_bytes(mean_32_bytes),
        },
        "selected_32_smallest_one_per_session_estimate": {
            "episode_count": len(selected_32),
            "estimated_training_bytes_excluding_visualization_rrd": sum(
                ep["training_bytes_excluding_visualization_rrd"] for ep in selected_32
            ),
            "estimated_training_human": human_bytes(
                sum(ep["training_bytes_excluding_visualization_rrd"] for ep in selected_32)
            ),
            "episodes": selected_32,
        },
        "top_level_sessions_by_file_count_top_n": top_level_counter.most_common(args.top_n),
        "top_level_sessions_by_bytes_top_n": [
            {"session": session, "bytes": bytes_, "human": human_bytes(bytes_)}
            for session, bytes_ in by_top_level_bytes.most_common(args.top_n)
        ],
        "largest_files_top_n": [
            {**item, "human": human_bytes(item["bytes"])} for item in largest_files
        ],
        "smallest_annotation_files_top_n": [
            {**item, "human": human_bytes(item["bytes"])}
            for item in sorted(annotation_files, key=lambda item: item["bytes"])[: args.top_n]
        ],
        "median_annotation_files_top_n": [
            {**item, "human": human_bytes(item["bytes"])}
            for item in near_size_files(annotation_files, annotation_median, args.top_n)
        ],
        "largest_annotation_files_top_n": [
            {**item, "human": human_bytes(item["bytes"])}
            for item in sorted(annotation_files, key=lambda item: item["bytes"], reverse=True)[: args.top_n]
        ],
        "largest_episode_folders_top_n": [
            {**item, "total_human": human_bytes(item["total_bytes"]), "training_human": human_bytes(item["training_bytes_excluding_visualization_rrd"])}
            for item in largest_episodes
        ],
        "smallest_complete_episode_training_folders_top_n": [
            {**item, "total_human": human_bytes(item["total_bytes"]), "training_human": human_bytes(item["training_bytes_excluding_visualization_rrd"])}
            for item in smallest_complete
        ],
        "download_recommendation": {
            "metadata_only_audit_requires_training_host": False,
            "recommended_download_host": "Any HF-reachable relay host with enough scratch storage; transfer staged episodes to the training host if that host cannot access Hugging Face.",
            "training_host_role": "training and local manifest validation after data is staged",
            "exclude_files": sorted(TRAINING_EXCLUDE),
            "minimum_pilot": "32 complete episodes from different top-level sessions if storage permits; degraded-valid episodes only for loader smoke tests.",
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    summary = payload["summary"]
    complete_sizes = payload["complete_episode_training_size_summary"]
    annotation_sizes_report = payload["annotation_file_size_summary"]
    pilot = payload["pilot_scale_estimates"]
    selected_32_estimate = payload["selected_32_smallest_one_per_session_estimate"]
    card_data = payload["card_data"] or {}
    report = [
        "# Xperience-10M HF Metadata Audit",
        "",
        "Metadata-only analysis of the gated Hugging Face dataset. No MP4, HDF5, RRD, or model files were downloaded.",
        "",
        "## Access and Source",
        "",
        f"- Repo: `{args.repo_id}`",
        f"- Repo SHA: `{payload['repo_sha']}`",
        f"- Last modified: `{payload['last_modified']}`",
        f"- Gated mode: `{payload['gated']}`",
        f"- Pretty name: `{card_data.get('pretty_name', 'Xperience-10M')}`",
        f"- License field: `{card_data.get('license', 'unknown')}`",
        f"- HF size category: `{', '.join(card_data.get('size_categories', [])) or 'unknown'}`",
        f"- Tags: `{', '.join(card_data.get('tags', []))}`",
        "",
        "## Current Hub File Metadata",
        "",
        *md_table(
            ["Measure", "Value"],
            [
                ["Files listed by API", f"{summary['file_count_excluding_gitattributes']:,}"],
                ["Total bytes from file metadata", f"{summary['total_human_from_file_metadata']} ({summary['total_bytes_from_file_metadata']:,} bytes)"],
                ["Bytes excluding visualization.rrd", f"{summary['training_human_excluding_visualization_rrd']} ({summary['training_bytes_excluding_visualization_rrd']:,} bytes)"],
                ["visualization.rrd bytes", f"{summary['visualization_rrd_human']} ({summary['visualization_rrd_bytes']:,} bytes)"],
                ["Top-level session folders", f"{summary['top_level_session_count']:,}"],
                ["Episode-like folders", f"{summary['episode_like_folder_count']:,}"],
            ],
        ),
        "",
        "## File Composition",
        "",
        *md_table(
            ["File type", "Count"],
            [[key, f"{value:,}"] for key, value in payload["file_type_counts"].items()],
        ),
        "",
        "## Episode Completeness",
        "",
        *md_table(
            ["Measure", "Value"],
            [
                ["annotation.hdf5 files", f"{summary['annotation_hdf5_count']:,}"],
                ["MP4 files", f"{summary['mp4_count']:,}"],
                ["visualization.rrd files", f"{summary['visualization_rrd_count']:,}"],
                ["Complete episodes: annotation + all six MP4 views", f"{summary['complete_episode_count']:,} ({summary['complete_episode_pct']}%)"],
                ["Degraded-valid episodes: annotation + fisheye_cam0", f"{summary['degraded_valid_episode_count']:,} ({summary['degraded_valid_episode_pct']}%)"],
                ["Sessions with complete episodes", f"{summary['complete_sessions']:,}"],
                ["Video-count histogram per episode", json.dumps(payload["video_count_histogram"], sort_keys=True)],
            ],
        ),
        "",
        "## Episode Size Distribution",
        "",
        *md_table(
            ["Statistic", "Training bytes per complete episode, excluding visualization.rrd"],
            [
                ["Min", complete_sizes.get("min_human")],
                ["P25", human_bytes(complete_sizes.get("p25_bytes", 0))],
                ["Median", complete_sizes.get("median_human")],
                ["P75", human_bytes(complete_sizes.get("p75_bytes", 0))],
                ["Mean", complete_sizes.get("mean_human")],
                ["Max", complete_sizes.get("max_human")],
            ],
        ),
        "",
        "## Annotation File Size Distribution",
        "",
        *md_table(
            ["Statistic", "annotation.hdf5 size"],
            [
                ["Min", annotation_sizes_report.get("min_human")],
                ["P25", human_bytes(annotation_sizes_report.get("p25_bytes", 0))],
                ["Median", annotation_sizes_report.get("median_human")],
                ["P75", human_bytes(annotation_sizes_report.get("p75_bytes", 0))],
                ["Mean", annotation_sizes_report.get("mean_human")],
                ["Max", annotation_sizes_report.get("max_human")],
            ],
        ),
        "",
        "## Pilot Scale Estimates",
        "",
        *md_table(
            ["Pilot", "Episodes", "Max windows at 256/episode", "Storage estimate"],
            [
                ["32-episode smallest one-per-session", selected_32_estimate["episode_count"], pilot["episode_32_windows_at_256_each"], selected_32_estimate["estimated_training_human"]],
                ["32-episode median-sized estimate", 32, pilot["episode_32_windows_at_256_each"], pilot["median_based_32_episode_training_human"]],
                ["32-episode mean-sized estimate", 32, pilot["episode_32_windows_at_256_each"], pilot["mean_based_32_episode_training_human"]],
                ["100-episode pilot", 100, pilot["episode_100_windows_at_256_each"], f"roughly {human_bytes(complete_sizes.get('median_bytes', 0) * 100)} at median episode size"],
                ["500-episode pilot", 500, pilot["episode_500_windows_at_256_each"], f"roughly {human_bytes(complete_sizes.get('median_bytes', 0) * 500)} at median episode size"],
                ["All complete visible HF episodes", summary["complete_episode_count"], pilot["all_complete_episodes_windows_at_256_each"], summary["all_complete_episode_training_human_excluding_visualization_rrd"]],
            ],
        ),
        "",
        "## Incomplete Episode Records",
        "",
        json.dumps(incomplete, indent=2) if incomplete else "None found.",
        "",
        "## Download and Compute Recommendation",
        "",
        "- This metadata audit can run on any machine with Hugging Face access.",
        "- If the training host cannot reach Hugging Face, download on an HF-reachable relay host, then transfer staged episode folders to the training host.",
        "- For training downloads, include `annotation.hdf5` plus the six MP4 streams; exclude `visualization.rrd` unless Rerun visualization is specifically needed.",
        "- For the first real training pilot, prefer 32 complete episodes from different top-level sessions and avoid selecting only the tiny outlier episodes.",
        "- The training host is used after staged data exists: manifest validation, preprocessing, LoRA training, and held-out evaluation.",
    ]
    args.report_output.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"PASS: wrote {args.output}")
    print(f"PASS: wrote {args.report_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

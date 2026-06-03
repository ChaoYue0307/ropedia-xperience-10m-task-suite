#!/usr/bin/env python3
"""Select a metadata-balanced Xperience-10M pilot subset.

The selector uses Hugging Face file metadata only. It does not download episode
data. Content-category balancing is deferred until annotations are staged,
because category text lives inside annotation.hdf5 files.
"""

from __future__ import annotations

import argparse
import csv
import getpass
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

from huggingface_hub import HfApi


REQUIRED_FILES = [
    "annotation.hdf5",
    "fisheye_cam0.mp4",
    "fisheye_cam1.mp4",
    "fisheye_cam2.mp4",
    "fisheye_cam3.mp4",
    "stereo_left.mp4",
    "stereo_right.mp4",
]
EXCLUDED_TRAINING_FILES = {"visualization.rrd"}
SIZE_BANDS = ["short", "lower_mid", "upper_mid", "long"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="ropedia-ai/xperience-10m")
    parser.add_argument("--target-episodes", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--val-fraction", type=float, default=0.125)
    parser.add_argument("--test-fraction", type=float, default=0.125)
    parser.add_argument("--drop-bottom-annotation-percentile", type=float, default=0.05)
    parser.add_argument("--drop-bottom-training-percentile", type=float, default=0.05)
    parser.add_argument("--min-annotation-gib", type=float, default=0.5)
    parser.add_argument("--windows-per-episode", type=int, default=256)
    parser.add_argument("--output-json", type=Path, default=Path("results/omni_finetune/xperience10m_128_episode_selection.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("results/omni_finetune/xperience10m_128_episode_selection.csv"))
    parser.add_argument("--download-list-output", type=Path, default=Path("results/omni_finetune/xperience10m_128_episode_download_files.txt"))
    parser.add_argument("--report-output", type=Path, default=Path("results/omni_finetune/XPERIENCE10M_128_EPISODE_SELECTION.md"))
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN", "").strip())
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
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if abs(value) < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TiB"


def quantile(values: list[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = min(max(q, 0.0), 1.0) * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return int(round(ordered[lo] * (1.0 - frac) + ordered[hi] * frac))


def summarize_sizes(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min_bytes": ordered[0],
        "p05_bytes": quantile(ordered, 0.05),
        "p25_bytes": quantile(ordered, 0.25),
        "median_bytes": int(median(ordered)),
        "p75_bytes": quantile(ordered, 0.75),
        "p95_bytes": quantile(ordered, 0.95),
        "max_bytes": ordered[-1],
        "mean_bytes": int(sum(ordered) / len(ordered)),
        "min_human": human_bytes(ordered[0]),
        "p05_human": human_bytes(quantile(ordered, 0.05)),
        "p25_human": human_bytes(quantile(ordered, 0.25)),
        "median_human": human_bytes(median(ordered)),
        "p75_human": human_bytes(quantile(ordered, 0.75)),
        "p95_human": human_bytes(quantile(ordered, 0.95)),
        "max_human": human_bytes(ordered[-1]),
        "mean_human": human_bytes(sum(ordered) / len(ordered)),
    }


def stable_hash(seed: int, text: str) -> str:
    return hashlib.sha256(f"{seed}:{text}".encode("utf-8")).hexdigest()


def stable_float(seed: int, text: str) -> float:
    return int(stable_hash(seed, text)[:12], 16) / float(16**12)


def episode_number(episode_id: str) -> int | None:
    match = re.fullmatch(r"ep(\d+)", episode_id)
    return int(match.group(1)) if match else None


def size_band(annotation_bytes: int, q25: int, q50: int, q75: int) -> str:
    if annotation_bytes <= q25:
        return "short"
    if annotation_bytes <= q50:
        return "lower_mid"
    if annotation_bytes <= q75:
        return "upper_mid"
    return "long"


def build_episode_records(siblings: list[Any]) -> list[dict[str, Any]]:
    by_parent: dict[str, dict[str, Any]] = defaultdict(lambda: {"files": {}, "bytes": 0})
    for sibling in siblings:
        path = str(getattr(sibling, "rfilename", ""))
        if not path or path == ".gitattributes":
            continue
        name = Path(path).name
        parent = Path(path).parent.as_posix()
        if not parent:
            continue
        size = file_size(sibling)
        bucket = by_parent[parent]
        bucket["files"][name] = {"path": path, "bytes": size}
        bucket["bytes"] += size

    records = []
    for parent, bucket in by_parent.items():
        files = bucket["files"]
        present = set(files)
        if "annotation.hdf5" not in present:
            continue
        has_all_six_videos = all(name in present for name in REQUIRED_FILES[1:])
        training_bytes = sum(
            meta["bytes"]
            for name, meta in files.items()
            if name not in EXCLUDED_TRAINING_FILES
        )
        records.append(
            {
                "episode_path": parent,
                "episode_id": Path(parent).name,
                "episode_number": episode_number(Path(parent).name),
                "top_level_session": parent.split("/", 1)[0],
                "file_count": len(present),
                "total_bytes": int(bucket["bytes"]),
                "training_bytes_excluding_visualization_rrd": int(training_bytes),
                "annotation_bytes": int(files["annotation.hdf5"]["bytes"]),
                "video_bytes": int(sum(files[name]["bytes"] for name in REQUIRED_FILES[1:] if name in files)),
                "has_annotation": True,
                "has_all_six_videos": has_all_six_videos,
                "has_visualization_rrd": "visualization.rrd" in present,
                "missing_required_files": [name for name in REQUIRED_FILES if name not in present],
                "download_files": [files[name]["path"] for name in REQUIRED_FILES if name in files],
            }
        )
    return records


def choose_target_counts(target: int) -> dict[str, int]:
    base = target // len(SIZE_BANDS)
    remainder = target % len(SIZE_BANDS)
    return {
        band: base + (1 if idx < remainder else 0)
        for idx, band in enumerate(SIZE_BANDS)
    }


def select_balanced(records: list[dict[str, Any]], target: int, seed: int) -> list[dict[str, Any]]:
    counts = choose_target_counts(target)
    by_band: dict[str, list[dict[str, Any]]] = {band: [] for band in SIZE_BANDS}
    band_medians = {
        band: median([record["annotation_bytes"] for record in records if record["size_band"] == band])
        for band in SIZE_BANDS
        if any(record["size_band"] == band for record in records)
    }

    # Keep the best representative episode per session per band. This prevents
    # one long session from dominating the sample.
    session_band_best: dict[tuple[str, str], dict[str, Any]] = {}
    global_training_median = median([record["training_bytes_excluding_visualization_rrd"] for record in records])
    for record in records:
        band = record["size_band"]
        band_median = float(band_medians.get(band, record["annotation_bytes"]) or 1.0)
        size_score = abs(record["annotation_bytes"] - band_median) / band_median
        training_score = abs(record["training_bytes_excluding_visualization_rrd"] - global_training_median) / float(global_training_median or 1.0)
        ep_num = record["episode_number"]
        index_score = 0.0 if ep_num is None else min(ep_num / 64.0, 1.0) * 0.02
        tie = stable_float(seed, record["episode_path"]) * 0.001
        record["selection_score"] = round(float(size_score + 0.25 * training_score + index_score + tie), 8)
        key = (record["top_level_session"], band)
        current = session_band_best.get(key)
        if current is None or record["selection_score"] < current["selection_score"]:
            session_band_best[key] = record

    for record in session_band_best.values():
        by_band[record["size_band"]].append(record)
    for band in SIZE_BANDS:
        by_band[band].sort(key=lambda item: (item["selection_score"], stable_hash(seed, item["episode_path"])))

    selected: list[dict[str, Any]] = []
    used_sessions: set[str] = set()
    selected_by_band = Counter()
    for band in SIZE_BANDS:
        for record in by_band[band]:
            if selected_by_band[band] >= counts[band]:
                break
            if record["top_level_session"] in used_sessions:
                continue
            selected.append(record)
            used_sessions.add(record["top_level_session"])
            selected_by_band[band] += 1

    if len(selected) < target:
        remaining = [
            record
            for band in SIZE_BANDS
            for record in by_band[band]
            if record["top_level_session"] not in used_sessions
        ]
        remaining.sort(key=lambda item: (item["selection_score"], stable_hash(seed, item["episode_path"])))
        for record in remaining:
            selected.append(record)
            used_sessions.add(record["top_level_session"])
            selected_by_band[record["size_band"]] += 1
            if len(selected) >= target:
                break

    if len(selected) < target:
        raise RuntimeError(f"Only selected {len(selected)} unique-session episodes; target is {target}.")
    return selected[:target]


def assign_splits(selected: list[dict[str, Any]], seed: int, train_fraction: float, val_fraction: float, test_fraction: float) -> None:
    total_fraction = train_fraction + val_fraction + test_fraction
    if abs(total_fraction - 1.0) > 1e-6:
        raise ValueError(f"Split fractions must sum to 1.0, got {total_fraction}")

    for band in SIZE_BANDS:
        band_records = [record for record in selected if record["size_band"] == band]
        band_records.sort(key=lambda item: stable_hash(seed + 101, item["episode_path"]))
        n = len(band_records)
        val_n = int(round(n * val_fraction))
        test_n = int(round(n * test_fraction))
        train_n = n - val_n - test_n
        for idx, record in enumerate(band_records):
            if idx < train_n:
                split = "train"
            elif idx < train_n + val_n:
                split = "val"
            else:
                split = "test"
            record["split"] = split


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
        "selection_rank",
        "split",
        "size_band",
        "episode_path",
        "top_level_session",
        "episode_id",
        "annotation_human",
        "training_human",
        "annotation_bytes",
        "training_bytes_excluding_visualization_rrd",
        "has_visualization_rrd",
        "selection_score",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def main() -> int:
    args = parse_args()
    token = args.token or getpass.getpass("HF token: ").strip()
    if not token:
        raise SystemExit("HF token is required for gated dataset metadata.")

    api = HfApi(token=token)
    info = api.repo_info(args.repo_id, repo_type="dataset", files_metadata=True, token=token)
    records = build_episode_records(list(info.siblings or []))
    complete = [record for record in records if record["has_all_six_videos"]]

    annotation_sizes = [record["annotation_bytes"] for record in complete]
    training_sizes = [record["training_bytes_excluding_visualization_rrd"] for record in complete]
    q25 = quantile(annotation_sizes, 0.25)
    q50 = quantile(annotation_sizes, 0.50)
    q75 = quantile(annotation_sizes, 0.75)
    min_annotation = max(
        int(args.min_annotation_gib * (1024**3)),
        quantile(annotation_sizes, args.drop_bottom_annotation_percentile),
    )
    min_training = quantile(training_sizes, args.drop_bottom_training_percentile)

    candidates = []
    rejected = Counter()
    for record in complete:
        if record["annotation_bytes"] < min_annotation:
            rejected["annotation_too_small"] += 1
            continue
        if record["training_bytes_excluding_visualization_rrd"] < min_training:
            rejected["training_too_small"] += 1
            continue
        record = dict(record)
        record["size_band"] = size_band(record["annotation_bytes"], q25, q50, q75)
        record["annotation_human"] = human_bytes(record["annotation_bytes"])
        record["training_human"] = human_bytes(record["training_bytes_excluding_visualization_rrd"])
        candidates.append(record)

    selected = select_balanced(candidates, args.target_episodes, args.seed)
    selected.sort(key=lambda item: (SIZE_BANDS.index(item["size_band"]), item["selection_score"], item["episode_path"]))
    assign_splits(selected, args.seed, args.train_fraction, args.val_fraction, args.test_fraction)
    for idx, record in enumerate(selected, start=1):
        record["selection_rank"] = idx

    selected_download_files = [
        filename
        for record in selected
        for filename in record["download_files"]
    ]
    split_counts = Counter(record["split"] for record in selected)
    band_counts = Counter(record["size_band"] for record in selected)
    split_band_counts = Counter((record["split"], record["size_band"]) for record in selected)
    selected_sessions = {record["top_level_session"] for record in selected}
    train_sessions = {record["top_level_session"] for record in selected if record["split"] == "train"}
    val_sessions = {record["top_level_session"] for record in selected if record["split"] == "val"}
    test_sessions = {record["top_level_session"] for record in selected if record["split"] == "test"}

    payload = {
        "status": "pass",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repo_id": args.repo_id,
        "repo_sha": getattr(info, "sha", None),
        "selection_type": "metadata_balanced_first_pass",
        "target_episodes": args.target_episodes,
        "seed": args.seed,
        "rules": {
            "complete_episode_required_files": REQUIRED_FILES,
            "excluded_training_files": sorted(EXCLUDED_TRAINING_FILES),
            "one_episode_per_top_level_session": True,
            "drop_bottom_annotation_percentile": args.drop_bottom_annotation_percentile,
            "drop_bottom_training_percentile": args.drop_bottom_training_percentile,
            "min_annotation_bytes": min_annotation,
            "min_annotation_human": human_bytes(min_annotation),
            "min_training_bytes": min_training,
            "min_training_human": human_bytes(min_training),
            "content_category_status": "not directly visible in HF metadata; refine after annotations are downloaded and captions are parsed",
        },
        "available_complete_episodes": len(complete),
        "candidate_episodes_after_filters": len(candidates),
        "rejected_counts": dict(rejected),
        "annotation_size_summary_complete": summarize_sizes(annotation_sizes),
        "training_size_summary_complete": summarize_sizes(training_sizes),
        "selected_summary": {
            "episode_count": len(selected),
            "unique_session_count": len(selected_sessions),
            "split_counts": dict(split_counts),
            "size_band_counts": dict(band_counts),
            "split_band_counts": {f"{split}/{band}": count for (split, band), count in split_band_counts.items()},
            "estimated_download_bytes_excluding_visualization_rrd": sum(record["training_bytes_excluding_visualization_rrd"] for record in selected),
            "estimated_download_human_excluding_visualization_rrd": human_bytes(sum(record["training_bytes_excluding_visualization_rrd"] for record in selected)),
            "estimated_annotation_bytes": sum(record["annotation_bytes"] for record in selected),
            "estimated_annotation_human": human_bytes(sum(record["annotation_bytes"] for record in selected)),
            "estimated_windows_at_configured_limit": len(selected) * args.windows_per_episode,
            "windows_per_episode": args.windows_per_episode,
            "train_sessions_overlap_val": sorted(train_sessions & val_sessions),
            "train_sessions_overlap_test": sorted(train_sessions & test_sessions),
            "val_sessions_overlap_test": sorted(val_sessions & test_sessions),
        },
        "selected_episodes": selected,
        "download_files": selected_download_files,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_csv(args.output_csv, selected)
    args.download_list_output.write_text("\n".join(selected_download_files) + "\n", encoding="utf-8")

    summary = payload["selected_summary"]
    report = [
        "# Xperience-10M 128-Episode Metadata-Balanced Selection",
        "",
        "This is a download plan, not a trained model result. It uses Hugging Face file metadata only and downloads no raw episode data.",
        "",
        "## Why This Selection",
        "",
        "- Use only complete episodes: `annotation.hdf5` plus six MP4 streams.",
        "- Exclude `visualization.rrd` from the training download plan.",
        "- Avoid tiny annotation outliers that are likely one-segment examples.",
        "- Use one episode per top-level session to reduce leakage and overfitting to one capture session.",
        "- Balance across four annotation-size bands as a proxy for duration/content richness before category labels are available.",
        "- Split by session into train/val/test.",
        "",
        "## Selection Summary",
        "",
        *md_table(
            ["Measure", "Value"],
            [
                ["Selected episodes", summary["episode_count"]],
                ["Unique sessions", summary["unique_session_count"]],
                ["Split counts", json.dumps(summary["split_counts"], sort_keys=True)],
                ["Size-band counts", json.dumps(summary["size_band_counts"], sort_keys=True)],
                ["Estimated training download, no RRD", summary["estimated_download_human_excluding_visualization_rrd"]],
                ["Estimated annotation bytes", summary["estimated_annotation_human"]],
                ["Estimated windows at 256/episode", summary["estimated_windows_at_configured_limit"]],
                ["Session leakage train/val", len(summary["train_sessions_overlap_val"])],
                ["Session leakage train/test", len(summary["train_sessions_overlap_test"])],
                ["Session leakage val/test", len(summary["val_sessions_overlap_test"])],
            ],
        ),
        "",
        "## Filters",
        "",
        *md_table(
            ["Rule", "Value"],
            [
                ["Available complete episodes", len(complete)],
                ["Candidates after filters", len(candidates)],
                ["Minimum annotation size", payload["rules"]["min_annotation_human"]],
                ["Minimum training size", payload["rules"]["min_training_human"]],
                ["Rejected counts", json.dumps(payload["rejected_counts"], sort_keys=True)],
            ],
        ),
        "",
        "## Split x Size Band",
        "",
        *md_table(
            ["Split", *SIZE_BANDS],
            [
                [split, *[split_band_counts.get((split, band), 0) for band in SIZE_BANDS]]
                for split in ["train", "val", "test"]
            ],
        ),
        "",
        "## Important Limitation",
        "",
        "HF metadata does not expose semantic content categories. This selection is the best first-pass balance before downloading. After the selected annotations are staged, parse `Main Task`, `Sub Task`, `Current Action`, objects, and interaction text; then swap episodes if one content cluster dominates.",
        "",
        "## Output Files",
        "",
        f"- JSON: `{args.output_json}`",
        f"- CSV: `{args.output_csv}`",
        f"- Download file list: `{args.download_list_output}`",
    ]
    args.report_output.write_text("\n".join(report) + "\n", encoding="utf-8")

    print(json.dumps(payload["selected_summary"], indent=2))
    print(f"PASS: wrote {args.output_json}")
    print(f"PASS: wrote {args.output_csv}")
    print(f"PASS: wrote {args.download_list_output}")
    print(f"PASS: wrote {args.report_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Download and inspect minimal Xperience-10M annotation.hdf5 files.

This probe intentionally downloads only annotation files, not videos or RRD
viewer files. Raw annotations are cached outside the repo by default.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from huggingface_hub import hf_hub_download


DEFAULT_ANNOTATIONS = [
    "9cecac72-8874-4b97-9541-18d4858f8e43/ep10/annotation.hdf5",
]
TEXT_RELATED_TERMS = ("caption", "action", "interaction", "object", "subtask", "task")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="ropedia-ai/xperience-10m")
    parser.add_argument("--filenames", nargs="+", default=DEFAULT_ANNOTATIONS)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(os.environ.get("XPERIENCE10M_ANNOTATION_PROBE_CACHE", "xperience10m_annotation_probe_cache")),
    )
    parser.add_argument("--output", type=Path, default=Path("results/omni_finetune/annotation_record_probe.json"))
    parser.add_argument("--report-output", type=Path, default=Path("results/omni_finetune/ANNOTATION_RECORD_PROBE.md"))
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN", "").strip())
    parser.add_argument("--sample-values", type=int, default=3)
    parser.add_argument("--local-files-only", action="store_true", help="Use the HF cache and do not contact the Hub.")
    return parser.parse_args()


def human_bytes(num: float | int) -> str:
    value = float(num)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if abs(value) < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TiB"


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def sample_dataset(ds: h5py.Dataset, limit: int) -> list[Any]:
    if ds.shape == ():
        raw = ds[()]
        return [json_safe(raw)]
    if not ds.shape or ds.shape[0] == 0:
        return []
    count = min(limit, int(ds.shape[0]))
    samples: list[Any] = []
    for idx in range(count):
        try:
            if ds.dtype.kind in {"S", "O", "U"}:
                raw = ds.asstr()[idx]
            else:
                raw = ds[idx]
            if isinstance(raw, np.ndarray) and raw.size > 12:
                raw = raw.reshape(-1)[:12]
            samples.append(json_safe(raw))
        except Exception as exc:  # pragma: no cover - defensive HDF5 read path
            samples.append(f"<sample failed: {exc}>")
            break
    return samples


def inspect_annotation(path: Path, sample_limit: int) -> dict[str, Any]:
    datasets: list[dict[str, Any]] = []
    related_datasets: list[dict[str, Any]] = []
    top_group_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"dataset_count": 0, "max_first_dim": 0, "first_dim_values": Counter()}
    )

    caption_json_summary = None
    with h5py.File(path, "r") as h5:
        top_level_keys = sorted(h5.keys())

        def visitor(name: str, obj: Any) -> None:
            if not isinstance(obj, h5py.Dataset):
                return
            shape = tuple(int(dim) for dim in obj.shape)
            first_dim = int(shape[0]) if shape else None
            top = name.split("/", 1)[0]
            stats = top_group_stats[top]
            stats["dataset_count"] += 1
            if first_dim is not None:
                stats["max_first_dim"] = max(int(stats["max_first_dim"]), first_dim)
                stats["first_dim_values"][str(first_dim)] += 1

            record = {
                "path": name,
                "shape": list(shape),
                "dtype": str(obj.dtype),
                "first_dim": first_dim,
                "storage_bytes": int(obj.id.get_storage_size()),
                "storage_human": human_bytes(obj.id.get_storage_size()),
            }
            datasets.append(record)

            lowered = name.lower()
            if any(term in lowered for term in TEXT_RELATED_TERMS):
                related = dict(record)
                related["sample_values"] = sample_dataset(obj, sample_limit)
                related_datasets.append(related)

        h5.visititems(visitor)
        if "caption" in h5 and isinstance(h5["caption"], h5py.Dataset):
            caption_json_summary = summarize_caption_json(h5["caption"])

    top_stats_out = {
        key: {
            "dataset_count": int(value["dataset_count"]),
            "max_first_dim": int(value["max_first_dim"]),
            "first_dim_values": dict(value["first_dim_values"].most_common(10)),
        }
        for key, value in sorted(top_group_stats.items())
    }
    dataset_first_dims = Counter(
        str(item["first_dim"]) for item in datasets if item["first_dim"] is not None
    )
    max_first_dim_dataset = max(datasets, key=lambda item: item["first_dim"] or -1) if datasets else None
    return {
        "cache_note": "annotation file cached outside the published repo",
        "local_bytes": path.stat().st_size,
        "local_human": human_bytes(path.stat().st_size),
        "top_level_keys": top_level_keys,
        "dataset_count": len(datasets),
        "dataset_first_dim_histogram_top20": dict(dataset_first_dims.most_common(20)),
        "top_group_stats": top_stats_out,
        "max_first_dim_dataset": max_first_dim_dataset,
        "text_action_interaction_related_datasets": related_datasets,
        "caption_json_summary": caption_json_summary,
    }


def summarize_caption_json(ds: h5py.Dataset) -> dict[str, Any] | None:
    try:
        raw = ds[()]
        text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
        data = json.loads(text)
    except Exception as exc:  # pragma: no cover - defensive parse path
        return {"parse_status": "failed", "error": str(exc)}

    segments = data.get("segments", [])
    if not isinstance(segments, list):
        segments = []

    sub_tasks = []
    action_labels = []
    object_names = []
    object_frame_count = 0
    interaction_frame_count = 0
    sampled_frame_count = 0

    for segment in segments:
        if not isinstance(segment, dict):
            continue
        if segment.get("Sub Task"):
            sub_tasks.append(str(segment["Sub Task"]))
        actions = segment.get("Current Action", [])
        if isinstance(actions, list):
            for action in actions:
                if isinstance(action, dict) and action.get("label"):
                    action_labels.append(str(action["label"]))
        objects = segment.get("objects", {})
        if isinstance(objects, dict):
            object_frame_count += len(objects)
            for names in objects.values():
                if isinstance(names, list):
                    object_names.extend(str(name) for name in names)
        interaction = segment.get("interaction", {})
        if isinstance(interaction, dict):
            interaction_frame_count += len(interaction)
        sampled_frames = segment.get("sampled_frames", {})
        if isinstance(sampled_frames, dict):
            sampled_frame_count += len(sampled_frames)

    config = data.get("config", {}) if isinstance(data, dict) else {}
    return {
        "parse_status": "ok",
        "json_bytes": len(text.encode("utf-8")),
        "top_keys": list(data.keys()) if isinstance(data, dict) else [],
        "config": config,
        "segment_count": len(segments),
        "current_action_count": len(action_labels),
        "unique_sub_task_count": len(set(sub_tasks)),
        "unique_action_label_count": len(set(action_labels)),
        "object_frame_count": object_frame_count,
        "interaction_frame_count": interaction_frame_count,
        "sampled_frame_count": sampled_frame_count,
        "unique_object_count": len(set(object_names)),
        "sub_tasks": sorted(set(sub_tasks))[:20],
        "action_labels": sorted(set(action_labels))[:20],
        "objects": sorted(set(object_names))[:30],
        "global_summary_preview": str(data.get("global_summary", ""))[:240] if isinstance(data, dict) else "",
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
    token = args.token
    if not args.local_files_only and not token:
        token = getpass.getpass("HF token: ").strip()
    if not args.local_files_only and not token:
        raise SystemExit("HF token is required for gated dataset annotation probing.")

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    probes = []
    for filename in args.filenames:
        local_path = hf_hub_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            filename=filename,
            cache_dir=args.cache_dir,
            token=token or None,
            local_files_only=args.local_files_only,
        )
        local = Path(local_path)
        probes.append(
            {
                "repo_filename": filename,
                "inspection": inspect_annotation(local, args.sample_values),
            }
        )

    payload = {
        "status": "pass",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repo_id": args.repo_id,
        "download_policy": "annotation.hdf5 only; no videos or visualization.rrd downloaded",
        "cache_note": "raw annotation files were cached outside the published repo",
        "probes": probes,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    report = [
        "# Xperience-10M Annotation Record Probe",
        "",
        "Minimal-cost probe. Downloaded only `annotation.hdf5`; no MP4 or `visualization.rrd` files were downloaded.",
        "",
        f"- Repo: `{args.repo_id}`",
        f"- Probe count: {len(probes)}",
        "- Raw annotation cache: outside the published repo",
        f"- Local files only: `{args.local_files_only}`",
        "",
    ]
    for probe in probes:
        inspection = probe["inspection"]
        max_ds = inspection.get("max_first_dim_dataset") or {}
        report.extend(
            [
                f"## {probe['repo_filename']}",
                "",
                f"- Downloaded annotation size: {inspection['local_human']} ({inspection['local_bytes']:,} bytes)",
                f"- HDF5 top-level keys: `{', '.join(inspection['top_level_keys'])}`",
                f"- HDF5 dataset count: {inspection['dataset_count']:,}",
                f"- Largest first-dimension dataset: `{max_ds.get('path')}` with first dimension `{max_ds.get('first_dim')}`",
                "",
                "### Caption JSON Summary",
                "",
            ]
        )
        caption_summary = inspection.get("caption_json_summary") or {}
        report.extend(
            md_table(
                ["Measure", "Value"],
                [
                    ["Parse status", caption_summary.get("parse_status")],
                    ["JSON bytes", f"{caption_summary.get('json_bytes', 0):,}"],
                    ["Segment count", caption_summary.get("segment_count")],
                    ["Current-action count", caption_summary.get("current_action_count")],
                    ["Object-frame count", caption_summary.get("object_frame_count")],
                    ["Interaction-frame count", caption_summary.get("interaction_frame_count")],
                    ["Sampled-frame count", caption_summary.get("sampled_frame_count")],
                    ["Unique subtasks", caption_summary.get("unique_sub_task_count")],
                    ["Unique action labels", caption_summary.get("unique_action_label_count")],
                    ["Unique objects", caption_summary.get("unique_object_count")],
                    ["Action labels", json.dumps(caption_summary.get("action_labels", []), ensure_ascii=False)],
                    ["Objects", json.dumps(caption_summary.get("objects", []), ensure_ascii=False)],
                ],
            )
        )
        report.extend(
            [
                "",
                "### Top Groups",
                "",
                *md_table(
                    ["Group", "Dataset count", "Max first dimension", "First-dim histogram top values"],
                    [
                        [
                            group,
                            stats["dataset_count"],
                            stats["max_first_dim"],
                            json.dumps(stats["first_dim_values"], sort_keys=False),
                        ]
                        for group, stats in inspection["top_group_stats"].items()
                    ],
                ),
                "",
                "### Caption / Action / Interaction Related Datasets",
                "",
            ]
        )
        related_rows = []
        for item in inspection["text_action_interaction_related_datasets"]:
            sample = json.dumps(item.get("sample_values", []), ensure_ascii=False)
            if len(sample) > 160:
                sample = sample[:157] + "..."
            related_rows.append([item["path"], item["shape"], item["dtype"], item["first_dim"], sample])
        report.extend(md_table(["Dataset", "Shape", "Dtype", "First dim", "Sample values"], related_rows or [["None", "", "", "", ""]]))
        report.append("")

    args.report_output.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"PASS: wrote {args.output}")
    print(f"PASS: wrote {args.report_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

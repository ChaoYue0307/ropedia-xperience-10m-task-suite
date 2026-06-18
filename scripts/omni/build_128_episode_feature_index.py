#!/usr/bin/env python3
"""Build the public 128-episode source and processed-feature index.

The index links every selected Xperience-10M episode back to the official
gated dataset path, then lists the public-safe derived feature artifacts that
are mirrored in this repository and the Hugging Face bundles.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import numpy as np
except Exception:  # pragma: no cover - the index can still be built without np
    np = None


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_JSON = ROOT / "docs/data/xperience10m_128_episode_feature_index.json"
OUTPUT_MD = ROOT / "XPERIENCE10M_128_EPISODE_FEATURE_INDEX.md"

OFFICIAL_REPO_ID = "ropedia-ai/xperience-10m"
OFFICIAL_TREE_BASE = f"https://huggingface.co/datasets/{OFFICIAL_REPO_ID}/tree/main"
OFFICIAL_RESOLVE_BASE = f"https://huggingface.co/datasets/{OFFICIAL_REPO_ID}/resolve/main"
PROJECT_ARTIFACT_REPO = "cy0307/ropedia-xperience-10m-task-suite-artifacts"
PROJECT_MODEL_REPO = "cy0307/ropedia-xperience-10m-task-baselines"

SELECTION_JSON = ROOT / "results/omni_finetune/xperience10m_128_episode_selection.json"
SELECTION_CSV = ROOT / "results/omni_finetune/xperience10m_128_episode_selection.csv"
DOWNLOAD_LIST = ROOT / "results/omni_finetune/xperience10m_128_episode_download_files.txt"
EPISODE_MANIFEST = ROOT / "results/omni_finetune/episode_manifest.json"
SPARSE_DATASET_MANIFEST = ROOT / "results/omni_finetune/dataset_manifest.json"
QWEN_V6_DATASET_MANIFEST = (
    ROOT
    / "results/omni_finetune/verified_public/"
    / "xperience10m_qwen3_omni_128ep_multiscale_cap96_v6_rank64_lr5e5_full8gpu_lora_eval_test_full/"
    / "dataset/dataset_manifest.json"
)
DENSE_DIR = ROOT / "results/omni_finetune/xperience10m_128ep_dense_multiscale_hierarchical_v1_20260608"
DENSE_DATASET = DENSE_DIR / "dense_multiscale_windows.jsonl"
DENSE_MANIFEST = DENSE_DIR / "dataset_manifest.json"
DENSE_LABEL_STATS = DENSE_DIR / "hierarchical_label_stats.json"
DENSE_SPLIT_SCALE_COUNTS = DENSE_DIR / "split_scale_counts.csv"
METADATA_MATRIX_V2 = (
    ROOT / "results/omni_finetune/a100_128_metadata_task_baselines_20260616_v2/metadata_feature_matrix.npz"
)
METADATA_MATRIX_SPARSE = (
    ROOT / "results/omni_finetune/multi_episode_128_task_baselines/metadata_feature_matrix.npz"
)
RAW20_DIR = ROOT / "results/omni_finetune/a100_128_raw20_task_baselines_complete20_proxy_20260616T091500Z"
RAW20_SUMMARY = RAW20_DIR / "run_summary_all.json"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def line_count(path: Path) -> int | None:
    if not path.exists():
        return None
    count = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            count += chunk.count(b"\n")
    return count


def npz_summary(path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {"available": path.exists()}
    if not path.exists():
        return summary
    summary.update({"bytes": path.stat().st_size, "sha256": sha256(path)})
    if np is None:
        return summary
    with np.load(path, allow_pickle=True) as data:
        arrays = {}
        for key in data.files:
            arr = data[key]
            arrays[key] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
        summary["arrays"] = arrays
        if "X" in data:
            summary["row_count"] = int(data["X"].shape[0])
            summary["feature_dim"] = int(data["X"].shape[1]) if data["X"].ndim > 1 else None
        if "split" in data:
            summary["split_counts"] = dict(Counter(str(x) for x in data["split"].tolist()))
    return summary


def artifact_record(path: Path, title: str, description: str, *, kind: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "title": title,
        "kind": kind,
        "description": description,
        "repo_path": rel(path),
        "github_url": f"https://github.com/ChaoYue0307/ropedia-xperience-10m-task-suite/blob/main/{rel(path)}",
        "hf_artifact_url": f"https://huggingface.co/datasets/{PROJECT_ARTIFACT_REPO}/resolve/main/{rel(path)}",
        "hf_model_url": f"https://huggingface.co/{PROJECT_MODEL_REPO}/resolve/main/{rel(path)}",
        "available": path.exists(),
    }
    if path.exists() and path.is_file():
        record["bytes"] = path.stat().st_size
        record["sha256"] = sha256(path)
        if path.suffix == ".jsonl":
            record["line_count"] = line_count(path)
        if path.suffix == ".npz":
            record["npz"] = npz_summary(path)
    return record


def split_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(row.get("split", "unknown")) for row in rows))


def selected_episode_records(selection: dict[str, Any], episode_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    manifest_by_key = {
        episode.get("episode_path"): episode for episode in episode_manifest.get("episodes", [])
    }
    rows = []
    for item in selection.get("selected_episodes", []):
        episode_path = item["episode_path"]
        manifest = manifest_by_key.get(episode_path, {})
        file_records = []
        for file_path in item.get("download_files", []):
            file_name = Path(file_path).name
            manifest_file = next(
                (entry for entry in manifest.get("files", []) if entry.get("name") == file_name),
                {},
            )
            file_records.append(
                {
                    "name": file_name,
                    "official_repo_path": file_path,
                    "gated_download_url": f"{OFFICIAL_RESOLVE_BASE}/{file_path}",
                    "bytes": manifest_file.get("bytes"),
                    "exists_in_selected_manifest": manifest_file.get("exists"),
                }
            )
        row_id_prefix = f"{item['top_level_session']}__{item['episode_id']}"
        rows.append(
            {
                "selection_rank": item.get("selection_rank"),
                "split": item.get("split"),
                "size_band": item.get("size_band"),
                "official_episode_path": episode_path,
                "top_level_session": item.get("top_level_session"),
                "source_episode_id": item.get("episode_id"),
                "canonical_episode_id": manifest.get("episode_id", row_id_prefix),
                "row_id_prefix": row_id_prefix,
                "official_tree_url": f"{OFFICIAL_TREE_BASE}/{episode_path}",
                "official_files": file_records,
                "main_task": manifest.get("main_task"),
                "frame_count": manifest.get("frame_count"),
                "window_counts": manifest.get("label_stats", {}).get("num_labeled_windows", {}),
                "hdf5_modalities": manifest.get("hdf5_modalities", {}),
                "annotation_bytes": item.get("annotation_bytes"),
                "training_bytes_excluding_visualization_rrd": item.get(
                    "training_bytes_excluding_visualization_rrd"
                ),
                "has_all_six_videos": item.get("has_all_six_videos"),
                "has_visualization_rrd": item.get("has_visualization_rrd"),
            }
        )
    return rows


def csv_split_scale_counts(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_index() -> dict[str, Any]:
    selection = load_json(SELECTION_JSON)
    episode_manifest = load_json(EPISODE_MANIFEST)
    sparse_manifest = load_json(SPARSE_DATASET_MANIFEST)
    qwen_v6_manifest = load_json(QWEN_V6_DATASET_MANIFEST)
    dense_manifest = load_json(DENSE_MANIFEST)
    raw20_summary = load_json(RAW20_SUMMARY)

    episodes = selected_episode_records(selection, episode_manifest)
    artifacts = [
        artifact_record(
            SELECTION_JSON,
            "128-episode selected source manifest",
            "Public-safe selection record with official episode paths, split labels, size bands, and file lists.",
            kind="source_index",
        ),
        artifact_record(
            SELECTION_CSV,
            "128-episode selected source table",
            "CSV table for quick scanning of rank, split, official episode path, source session, and size band.",
            kind="source_index",
        ),
        artifact_record(
            DOWNLOAD_LIST,
            "Official raw-file download list",
            "Seven official gated raw files per selected episode: annotation HDF5 plus six synchronized MP4 streams.",
            kind="source_index",
        ),
        artifact_record(
            EPISODE_MANIFEST,
            "Inspected 128-episode manifest",
            "Per-episode file sizes, frame counts, task labels, HDF5 modality availability, and selected split metadata.",
            kind="episode_manifest",
        ),
        artifact_record(
            SPARSE_DATASET_MANIFEST,
            "Sparse 20-frame JSONL export manifest",
            "Manifest for the sparse Qwen-style 20-frame window export after export-time filtering.",
            kind="processed_manifest",
        ),
        artifact_record(
            QWEN_V6_DATASET_MANIFEST,
            "Qwen3-Omni v6 multiscale dataset manifest",
            "Verified package manifest for the current 34,269-window Qwen3-Omni v6 dataset branch.",
            kind="processed_manifest",
        ),
        artifact_record(
            DENSE_DATASET,
            "Dense multiscale public-safe windows JSONL",
            "Compact public-safe rows for dense/medium/long windows, labels, object sets, and sparse-window provenance.",
            kind="processed_feature_table",
        ),
        artifact_record(
            DENSE_MANIFEST,
            "Dense multiscale manifest",
            "Counts and provenance for the dense multiscale public-safe feature table.",
            kind="processed_manifest",
        ),
        artifact_record(
            DENSE_LABEL_STATS,
            "Dense hierarchical label stats",
            "Action/subtask family and object-label statistics for the dense multiscale feature table.",
            kind="processed_summary",
        ),
        artifact_record(
            DENSE_SPLIT_SCALE_COUNTS,
            "Dense split-by-scale counts",
            "CSV counts by split and scale id for the dense multiscale feature table.",
            kind="processed_summary",
        ),
        artifact_record(
            METADATA_MATRIX_V2,
            "128-episode metadata feature matrix v2",
            "Public-safe 34,269 x 394 metadata/text feature matrix used by the aligned metadata baselines.",
            kind="processed_feature_matrix",
        ),
        artifact_record(
            METADATA_MATRIX_SPARSE,
            "128-episode sparse metadata feature matrix",
            "Earlier 3,808 x 394 metadata/text feature matrix for the sparse 20-frame export.",
            kind="processed_feature_matrix",
        ),
        artifact_record(
            RAW20_SUMMARY,
            "128-episode raw20 baseline summary",
            "All 40 simple/raw and neural/raw result records for the 20-task raw-feature baseline run.",
            kind="result_summary",
        ),
    ]

    return {
        "status": "pass",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "official_dataset": {
            "repo_id": selection.get("repo_id", OFFICIAL_REPO_ID),
            "repo_sha": selection.get("repo_sha"),
            "tree_base_url": OFFICIAL_TREE_BASE,
            "gated_resolve_base_url": OFFICIAL_RESOLVE_BASE,
            "access_note": (
                "Episode directory pages are browsable on Hugging Face. Raw annotation/video file "
                "downloads require access to the gated official dataset and are not redistributed here."
            ),
        },
        "selection_summary": {
            "selected_episode_count": len(selection.get("selected_episodes", [])),
            "selected_split_counts": split_counts(selection.get("selected_episodes", [])),
            "size_band_counts": dict(Counter(e.get("size_band") for e in selection.get("selected_episodes", []))),
            "one_episode_per_top_level_session": len(
                {e.get("top_level_session") for e in selection.get("selected_episodes", [])}
            )
            == len(selection.get("selected_episodes", [])),
            "selected_download_size_excluding_visualization_rrd_bytes": sum(
                int(e.get("training_bytes_excluding_visualization_rrd") or 0)
                for e in selection.get("selected_episodes", [])
            ),
        },
        "processed_summary": {
            "inspected_episode_manifest_count": len(episode_manifest.get("episodes", [])),
            "sparse_export": {
                "num_episodes": sparse_manifest.get("num_episodes"),
                "num_samples": sparse_manifest.get("num_samples"),
                "split_counts": sparse_manifest.get("split_counts"),
            },
            "qwen_v6_multiscale_export": {
                "num_episodes": qwen_v6_manifest.get("num_episodes"),
                "num_samples": qwen_v6_manifest.get("num_samples"),
                "split_counts": qwen_v6_manifest.get("split_counts"),
            },
            "dense_multiscale_compact_export": {
                "num_episodes": dense_manifest.get("num_episodes"),
                "num_samples": dense_manifest.get("num_samples"),
                "split_counts": dense_manifest.get("split_counts"),
                "scale_counts": dense_manifest.get("scale_counts"),
                "split_scale_counts": csv_split_scale_counts(DENSE_SPLIT_SCALE_COUNTS),
            },
            "metadata_matrix_v2": npz_summary(METADATA_MATRIX_V2),
            "metadata_matrix_sparse": npz_summary(METADATA_MATRIX_SPARSE),
            "raw20_result_records": raw20_summary.get("num_result_records"),
            "raw20_status_counts": raw20_summary.get("status_counts"),
            "raw20_proxy_tasks": raw20_summary.get("proxy_tasks"),
        },
        "processed_feature_artifacts": artifacts,
        "episodes": episodes,
        "non_redistribution_policy": {
            "raw_not_included": ["annotation.hdf5", "fisheye_cam*.mp4", "stereo_*.mp4", "visualization.rrd"],
            "reason": "Raw Xperience-10M files remain in the official gated dataset.",
            "included_public_safe": [
                "episode/source ids",
                "file sizes and modality availability",
                "compact dense window rows",
                "metadata feature matrices",
                "baseline metrics and predictions",
                "verified public model-package summaries",
            ],
        },
    }


def write_markdown(index: dict[str, Any]) -> None:
    lines = [
        "# Xperience-10M 128-Episode Feature Index",
        "",
        "This file links the selected 128-episode split to the official Xperience-10M episode paths and to the public-safe processed feature artifacts mirrored by this project.",
        "",
        "Raw Xperience-10M annotation/video files are not redistributed. Each episode row includes the official Hugging Face tree URL and gated raw-file URLs so a reader with access can resolve the source data.",
        "",
        "## Summary",
        "",
        f"- official_dataset: `{index['official_dataset']['repo_id']}`",
        f"- official_repo_sha: `{index['official_dataset'].get('repo_sha')}`",
        f"- selected_episode_count: `{index['selection_summary']['selected_episode_count']}`",
        f"- selected_split_counts: `{json.dumps(index['selection_summary']['selected_split_counts'], sort_keys=True)}`",
        f"- qwen_v6_multiscale_windows: `{index['processed_summary']['qwen_v6_multiscale_export']['num_samples']}`",
        f"- dense_multiscale_compact_windows: `{index['processed_summary']['dense_multiscale_compact_export']['num_samples']}`",
        f"- metadata_matrix_v2_shape: `{index['processed_summary']['metadata_matrix_v2'].get('npz', index['processed_summary']['metadata_matrix_v2']).get('arrays', {}).get('X', {}).get('shape')}`",
        "",
        "## Processed Feature Artifacts",
        "",
        "| Artifact | What it represents | Repo path |",
        "| --- | --- | --- |",
    ]
    for artifact in index["processed_feature_artifacts"]:
        lines.append(
            f"| {artifact['title']} | {artifact['description']} | `{artifact['repo_path']}` |"
        )

    lines += [
        "",
        "## Selected Episodes",
        "",
        "| Rank | Split | Official episode path | Canonical id | Frames | Action windows | Official tree |",
        "| ---: | --- | --- | --- | ---: | ---: | --- |",
    ]
    for episode in index["episodes"]:
        action_windows = episode.get("window_counts", {}).get("action", "")
        lines.append(
            "| {rank} | {split} | `{path}` | `{canonical}` | {frames} | {windows} | [HF tree]({url}) |".format(
                rank=episode.get("selection_rank"),
                split=episode.get("split"),
                path=episode.get("official_episode_path"),
                canonical=episode.get("canonical_episode_id"),
                frames=episode.get("frame_count") or "",
                windows=action_windows,
                url=episode.get("official_tree_url"),
            )
        )
    lines.append("")
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    index = build_index()
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    write_markdown(index)
    print(f"WROTE {OUTPUT_JSON}")
    print(f"WROTE {OUTPUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

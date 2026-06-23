#!/usr/bin/env python3
"""Build reader-facing data exploration assets for Xperience-10M.

The script intentionally separates three scopes:

1. The official public sample episode mirrored in this repository.
2. The selected 128-episode public-safe feature/export surface.
3. The gated upstream Hugging Face dataset, inspected through Hub file metadata.

It does not download or redistribute gated raw files.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ropedia-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/ropedia-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BG = "#020502"
PANEL = "#071307"
GRID = "#23341f"
TEXT = "#f4f7ee"
MUTED = "#b8c4b4"
GREEN = "#c6ff92"
GREEN_DARK = "#6fb03f"
CYAN = "#67e8d1"
BLUE = "#9bb8ff"
GOLD = "#ffd166"
PINK = "#f472b6"
PURPLE = "#b084ff"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def human_bytes(value: int | float | None) -> str:
    if value is None:
        return "n/a"
    value = float(value)
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    index = 0
    while value >= 1024 and index < len(units) - 1:
        value /= 1024
        index += 1
    if index == 0:
        return f"{int(value):,} {units[index]}"
    return f"{value:,.2f} {units[index]}"


def pct(numer: float, denom: float) -> float:
    return 0.0 if denom == 0 else 100.0 * numer / denom


def top_items(counter: Counter[str], limit: int = 10) -> list[dict[str, Any]]:
    return [
        {"name": name, "count": int(count)}
        for name, count in counter.most_common(limit)
    ]


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(True, axis="x", color=GRID, alpha=0.55, linewidth=0.8)
    ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    path.write_text(
        "\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n",
        encoding="utf-8",
    )


def add_value_labels(ax: plt.Axes, bars: Iterable[Any], formatter=str, pad: float = 0.02) -> None:
    xmax = ax.get_xlim()[1]
    for bar in bars:
        width = bar.get_width()
        label = formatter(width)
        ax.text(
            width + xmax * pad,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            ha="left",
            color=TEXT,
            fontsize=9,
            fontweight="bold",
        )


def build_public_sample(root: Path) -> dict[str, Any]:
    raw = read_json(root / "docs/data/raw_sample_files.json")
    explorer = read_json(root / "docs/data/single_episode_explorer.json")
    windows = read_csv_rows(root / "results/episode_task_suite/windows.csv")

    files = raw.get("files", [])
    file_bytes_by_kind: Counter[str] = Counter()
    file_count_by_kind: Counter[str] = Counter()
    for item in files:
        kind = str(item.get("kind", "other"))
        file_count_by_kind[kind] += 1
        file_bytes_by_kind[kind] += int(item.get("bytes", 0) or 0)

    feature_by_modality: Counter[str] = Counter()
    feature_blocks = []
    for block in explorer.get("feature_blocks", []):
        modality = str(block.get("modality", "other"))
        dim = int(block.get("dim", 0) or 0)
        feature_by_modality[modality] += dim
        feature_blocks.append(
            {
                "name": block.get("name"),
                "display": block.get("display"),
                "modality": modality,
                "dim": dim,
            }
        )

    action_counts = Counter(row.get("action_label", "unknown") for row in windows)
    subtask_counts = Counter(row.get("subtask_label", "unknown") for row in windows)
    object_counts: Counter[str] = Counter()
    for row in explorer.get("windows", []):
        for obj in row.get("objects", []) or []:
            if obj:
                object_counts[str(obj)] += 1

    windowization = raw.get("windowization", {})
    frames = int(windowization.get("num_frames", 0) or 0)
    fps = float(windowization.get("fps_observed", 0.0) or 0.0)
    duration_sec = frames / fps if fps > 0 else None

    return {
        "dataset": raw.get("dataset", {}),
        "windowization": {
            **windowization,
            "duration_sec": duration_sec,
            "duration_human": f"{duration_sec / 60.0:.2f} min" if duration_sec else "n/a",
        },
        "file_count": len(files),
        "total_bytes": sum(int(item.get("bytes", 0) or 0) for item in files),
        "total_human": human_bytes(sum(int(item.get("bytes", 0) or 0) for item in files)),
        "file_bytes_by_kind": {
            key: {"bytes": int(value), "human": human_bytes(value), "count": int(file_count_by_kind[key])}
            for key, value in sorted(file_bytes_by_kind.items())
        },
        "hdf5_groups": raw.get("hdf5_organization", []),
        "feature_dim_by_modality": dict(sorted(feature_by_modality.items(), key=lambda item: (-item[1], item[0]))),
        "feature_blocks": sorted(feature_blocks, key=lambda item: (-item["dim"], item["display"] or item["name"] or "")),
        "top_actions": top_items(action_counts, 12),
        "top_subtasks": top_items(subtask_counts, 12),
        "top_objects": top_items(object_counts, 12),
        "segment_count": len(explorer.get("segments", [])),
        "object_vocab_count": int(explorer.get("meta", {}).get("object_vocab_count", 0) or 0),
        "source_policy": explorer.get("meta", {}).get("source_policy"),
    }


def build_selected_128(root: Path) -> dict[str, Any]:
    feature_index = read_json(root / "docs/data/xperience10m_128_episode_feature_index.json")
    selection_rows = read_csv_rows(root / "results/omni_finetune/xperience10m_128_episode_selection.csv")
    sparse_windows = read_csv_rows(root / "results/omni_finetune/multi_episode_128_task_baselines/windows.csv")

    split_episode_counts = Counter(row.get("split", "unknown") for row in selection_rows)
    band_counts = Counter(row.get("size_band", "unknown") for row in selection_rows)
    bytes_by_split: Counter[str] = Counter()
    bytes_by_band: Counter[str] = Counter()
    for row in selection_rows:
        value = int(float(row.get("training_bytes_excluding_visualization_rrd", 0) or 0))
        bytes_by_split[row.get("split", "unknown")] += value
        bytes_by_band[row.get("size_band", "unknown")] += value

    sparse_windows_by_split = Counter(row.get("split", "unknown") for row in sparse_windows)
    main_task_counts = Counter(row.get("main_task", "unknown") for row in sparse_windows)

    processed = feature_index.get("processed_summary", {})
    export_rows = []
    for key, label in [
        ("sparse_export", "Sparse selected-128 export"),
        ("qwen_v6_multiscale_export", "Qwen3-Omni v6 multiscale JSONL"),
        ("dense_multiscale_compact_export", "Dense multiscale compact export"),
    ]:
        item = processed.get(key, {})
        export_rows.append(
            {
                "key": key,
                "label": label,
                "episodes": int(item.get("num_episodes", 0) or 0),
                "samples": int(item.get("num_samples", 0) or 0),
                "split_counts": {k: int(v) for k, v in (item.get("split_counts", {}) or {}).items()},
                "scale_counts": {k: int(v) for k, v in (item.get("scale_counts", {}) or {}).items()},
            }
        )

    matrix = processed.get("metadata_matrix_v2", {})
    sparse_matrix = processed.get("metadata_matrix_sparse", {})
    return {
        "official_dataset": feature_index.get("official_dataset", {}),
        "selection_summary": feature_index.get("selection_summary", {}),
        "split_episode_counts": dict(split_episode_counts),
        "size_band_counts": dict(band_counts),
        "bytes_by_split": {
            key: {"bytes": int(value), "human": human_bytes(value)}
            for key, value in sorted(bytes_by_split.items())
        },
        "bytes_by_size_band": {
            key: {"bytes": int(value), "human": human_bytes(value)}
            for key, value in sorted(bytes_by_band.items())
        },
        "sparse_windows_by_split": dict(sparse_windows_by_split),
        "sparse_main_task_counts": top_items(main_task_counts, 12),
        "exports": export_rows,
        "metadata_matrix_v2": {
            "row_count": int(matrix.get("row_count", 0) or 0),
            "feature_dim": int(matrix.get("feature_dim", 0) or 0),
            "bytes": int(matrix.get("bytes", 0) or 0),
            "human": human_bytes(int(matrix.get("bytes", 0) or 0)),
            "split_counts": {k: int(v) for k, v in (matrix.get("split_counts", {}) or {}).items()},
            "sha256": matrix.get("sha256"),
        },
        "metadata_matrix_sparse": {
            "row_count": int(sparse_matrix.get("row_count", 0) or 0),
            "feature_dim": int(sparse_matrix.get("feature_dim", 0) or 0),
            "bytes": int(sparse_matrix.get("bytes", 0) or 0),
            "human": human_bytes(int(sparse_matrix.get("bytes", 0) or 0)),
            "split_counts": {k: int(v) for k, v in (sparse_matrix.get("split_counts", {}) or {}).items()},
            "sha256": sparse_matrix.get("sha256"),
        },
        "raw20_result_records": int(processed.get("raw20_result_records", 0) or 0),
        "raw20_proxy_tasks": processed.get("raw20_proxy_tasks", []),
    }


def build_full_hf_dataset(root: Path) -> dict[str, Any]:
    audit = read_json(root / "results/omni_finetune/full_dataset_metadata_audit.json")
    summary = audit.get("summary", {})
    return {
        "repo_id": audit.get("repo_id"),
        "repo_sha": audit.get("repo_sha"),
        "gated": audit.get("gated"),
        "last_modified": audit.get("last_modified"),
        "card_data": audit.get("card_data", {}),
        "summary": summary,
        "file_type_counts": audit.get("file_type_counts", {}),
        "basename_counts": audit.get("basename_counts", {}),
        "video_count_histogram": audit.get("video_count_histogram", {}),
        "episode_count_per_session_summary": audit.get("episode_count_per_session_summary", {}),
        "episode_size_summary": audit.get("episode_size_summary", {}),
        "annotation_file_size_summary": audit.get("annotation_file_size_summary", {}),
        "complete_episode_training_size_summary": audit.get("complete_episode_training_size_summary", {}),
        "incomplete_episode_records": audit.get("incomplete_episode_records", []),
        "pilot_scale_estimates": audit.get("pilot_scale_estimates", {}),
        "metadata_note": (
            "The official dataset is gated on Hugging Face. These full-corpus figures "
            "use authenticated Hugging Face Hub file metadata for the HF-hosted dataset "
            "version only; they do not inspect private row content or redistribute raw "
            "MP4/HDF5/RRD files."
        ),
    }


def plot_scope_ladder(payload: dict[str, Any], out: Path) -> None:
    sample = payload["public_sample"]
    selected = payload["selected_128"]
    full = payload["full_hf_dataset"]

    labels = ["Sample", "Selected-128", "HF full dataset"]
    episodes = [
        1,
        selected["selection_summary"].get("selected_episode_count", 0),
        full["summary"].get("episode_like_folder_count", 0),
    ]
    windows = [
        sample["windowization"].get("num_windows", 0),
        selected["metadata_matrix_v2"].get("row_count", 0),
        full["pilot_scale_estimates"].get("all_complete_episodes_windows_at_256_each", 0),
    ]
    storage = [
        sample["total_bytes"],
        selected["selection_summary"].get("selected_download_size_excluding_visualization_rrd_bytes", 0),
        full["summary"].get("training_bytes_excluding_visualization_rrd", 0),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), facecolor=BG)
    specs = [
        ("Episodes", episodes, lambda v: f"{int(v):,}"),
        ("Window rows", windows, lambda v: f"{int(v):,}"),
        ("Training bytes", storage, lambda v: human_bytes(v)),
    ]
    colors = [GREEN, CYAN, BLUE]
    for ax, (title, values, formatter) in zip(axes, specs):
        style_axis(ax)
        safe_values = [max(float(v), 1.0) for v in values]
        bars = ax.barh(labels, safe_values, color=colors, edgecolor=TEXT, linewidth=0.4)
        ax.set_xscale("log")
        ax.set_title(title, color=TEXT, fontsize=15, fontweight="bold", loc="left", pad=10)
        ax.tick_params(axis="y", colors=TEXT, labelsize=10)
        xmax = max(safe_values) * 3.8
        ax.set_xlim(0.8, xmax)
        for bar, raw_value in zip(bars, values):
            raw_value = max(float(raw_value), 1.0)
            if raw_value > 20:
                label_x = raw_value / 1.16
                ha = "right"
                color = BG
            else:
                label_x = raw_value * 1.18
                ha = "left"
                color = TEXT
            ax.text(
                label_x,
                bar.get_y() + bar.get_height() / 2,
                formatter(raw_value),
                va="center",
                ha=ha,
                color=color,
                fontsize=9,
                fontweight="bold",
            )
    fig.suptitle(
        "Xperience-10M scope ladder",
        color=TEXT,
        fontsize=18,
        fontweight="bold",
        x=0.02,
        y=0.99,
        ha="left",
    )
    fig.subplots_adjust(left=0.08, right=0.985, top=0.79, bottom=0.18, wspace=0.34)
    fig.text(
        0.02,
        0.02,
        "Log-scale bars compare the one public sample, the selected-128 surface, and authenticated full-corpus file metadata.",
        color=MUTED,
        fontsize=10,
    )
    save_figure(fig, out)


def plot_feature_breakdown(payload: dict[str, Any], out: Path) -> None:
    values = payload["public_sample"]["feature_dim_by_modality"]
    labels = list(values.keys())[::-1]
    dims = [values[label] for label in labels]
    colors = [GREEN, CYAN, BLUE, GOLD, PINK, PURPLE, GREEN_DARK, MUTED][: len(labels)]

    fig, ax = plt.subplots(figsize=(11, 6.2), facecolor=BG)
    style_axis(ax)
    bars = ax.barh(labels, dims, color=colors[::-1], edgecolor=TEXT, linewidth=0.35)
    ax.set_title("Public sample feature dimensions by modality", color=TEXT, fontsize=18, fontweight="bold", loc="left", pad=14)
    ax.set_xlabel("Feature dimensions in the 8,546-D task input", color=MUTED)
    ax.tick_params(axis="y", colors=TEXT, labelsize=10)
    add_value_labels(ax, bars, lambda v: f"{int(v):,}")
    save_figure(fig, out)


def plot_action_distribution(payload: dict[str, Any], out: Path) -> None:
    items = payload["public_sample"]["top_actions"][:10]
    labels = [item["name"] for item in items][::-1]
    counts = [item["count"] for item in items][::-1]

    fig, ax = plt.subplots(figsize=(11.5, 6.4), facecolor=BG)
    style_axis(ax)
    bars = ax.barh(labels, counts, color=GREEN, edgecolor=TEXT, linewidth=0.35)
    ax.set_title("Public sample action-window distribution", color=TEXT, fontsize=18, fontweight="bold", loc="left", pad=14)
    ax.set_xlabel("20-frame windows carrying each action label", color=MUTED)
    ax.tick_params(axis="y", colors=TEXT, labelsize=9)
    add_value_labels(ax, bars, lambda v: f"{int(v):,}")
    save_figure(fig, out)


def plot_selected_split_windows(payload: dict[str, Any], out: Path) -> None:
    exports = payload["selected_128"]["exports"]
    split_order = ["train", "val", "test"]
    split_colors = {"train": GREEN, "val": CYAN, "test": BLUE}
    labels = [item["label"] for item in exports]
    y_positions = range(len(exports))

    fig, ax = plt.subplots(figsize=(12.5, 5.8), facecolor=BG)
    style_axis(ax)
    left = [0] * len(exports)
    for split in split_order:
        values = [int(item.get("split_counts", {}).get(split, 0) or 0) for item in exports]
        bars = ax.barh(
            list(y_positions),
            values,
            left=left,
            color=split_colors[split],
            label=split,
            edgecolor=BG,
            linewidth=0.4,
        )
        for index, (bar, value) in enumerate(zip(bars, values)):
            if value:
                ax.text(
                    left[index] + value / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:,}",
                    va="center",
                    ha="center",
                    color=BG,
                    fontsize=8,
                    fontweight="bold",
                )
        left = [left_value + value for left_value, value in zip(left, values)]
    for y, total in zip(y_positions, left):
        ax.text(total * 1.01, y, f"{total:,}", va="center", ha="left", color=TEXT, fontsize=9, fontweight="bold")
    ax.set_yticks(list(y_positions), labels)
    ax.tick_params(axis="y", colors=TEXT, labelsize=9)
    ax.set_xlabel("Rows / samples", color=MUTED)
    ax.set_title("Selected-128 processed rows by split", color=TEXT, fontsize=18, fontweight="bold", loc="left", pad=14)
    leg = ax.legend(loc="lower right", frameon=True, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
    for text in leg.get_texts():
        text.set_color(TEXT)
    save_figure(fig, out)


def plot_full_file_composition(payload: dict[str, Any], out: Path) -> None:
    counts = payload["full_hf_dataset"]["basename_counts"]
    ordered = [
        ("annotation.hdf5", counts.get("annotation.hdf5", 0)),
        ("all MP4 streams", payload["full_hf_dataset"]["summary"].get("mp4_count", 0)),
        ("visualization.rrd", counts.get("visualization.rrd", 0)),
        ("README.md", counts.get("README.md", 0)),
    ]
    labels = [name for name, _ in ordered][::-1]
    values = [value for _, value in ordered][::-1]

    fig, ax = plt.subplots(figsize=(10.5, 5.1), facecolor=BG)
    style_axis(ax)
    bars = ax.barh(labels, values, color=[MUTED, BLUE, CYAN, GREEN], edgecolor=TEXT, linewidth=0.35)
    ax.set_xscale("log")
    ax.set_xlabel("File count, log scale", color=MUTED)
    ax.set_title("Full gated dataset file composition", color=TEXT, fontsize=18, fontweight="bold", loc="left", pad=14)
    ax.tick_params(axis="y", colors=TEXT, labelsize=10)
    ax.set_xlim(0.8, max(values) * 5)
    for bar, value in zip(bars, values):
        ax.text(max(value, 1) * 1.08, bar.get_y() + bar.get_height() / 2, f"{int(value):,}", va="center", ha="left", color=TEXT, fontsize=9, fontweight="bold")
    save_figure(fig, out)


def render_markdown(payload: dict[str, Any]) -> str:
    sample = payload["public_sample"]
    selected = payload["selected_128"]
    full = payload["full_hf_dataset"]
    lines = [
        "# Ropedia Xperience-10M Data Explorer Analysis",
        "",
        f"Generated: {payload['generated_at_utc']}",
        "",
        "This report summarizes three data scopes without mixing them: the official public sample episode, the selected 128-episode public-safe feature surface, and authenticated metadata for the Hugging Face-hosted gated full dataset.",
        "",
        "## Scope Summary",
        "",
        "| Scope | Episodes | Rows / windows | Storage view | Notes |",
        "|---|---:|---:|---:|---|",
        f"| Public sample | 1 | {sample['windowization'].get('num_windows', 0):,} | {sample['total_human']} | Raw sample files are playable or source-linked. |",
        f"| Selected 128 | {selected['selection_summary'].get('selected_episode_count', 0):,} | {selected['metadata_matrix_v2'].get('row_count', 0):,} | {human_bytes(selected['selection_summary'].get('selected_download_size_excluding_visualization_rrd_bytes', 0))} | Public-safe matrices and window manifests, not raw redistribution. |",
        f"| Full HF dataset | {full['summary'].get('episode_like_folder_count', 0):,} episode-like folders | {full['pilot_scale_estimates'].get('all_complete_episodes_windows_at_256_each', 0):,} projected rows at 256/episode | {full['summary'].get('training_human_excluding_visualization_rrd', 'n/a')} | Gated upstream file metadata only. |",
        "",
        "## Public Sample",
        "",
        f"- {sample['windowization'].get('num_frames', 0):,} frames at about {sample['windowization'].get('fps_observed', 0):.2f} fps.",
        f"- {sample['windowization'].get('num_windows', 0):,} aligned 20-frame windows with {sample['windowization'].get('stride_frames', 0)}-frame stride.",
        f"- {sample['windowization'].get('feature_dim', 0):,} model-input dimensions across {len(sample['feature_dim_by_modality'])} modality groups.",
        f"- {sample['segment_count']:,} action segments and {sample['object_vocab_count']:,} object labels in the derived explorer.",
        "",
        "## Selected 128 Episodes",
        "",
        f"- Split: train {selected['split_episode_counts'].get('train', 0)}, val {selected['split_episode_counts'].get('val', 0)}, test {selected['split_episode_counts'].get('test', 0)} episodes.",
        f"- Size bands: {', '.join(f'{k} {v}' for k, v in selected['size_band_counts'].items())}.",
        f"- Qwen3-Omni v6 multiscale export: {next((item['samples'] for item in selected['exports'] if item['key'] == 'qwen_v6_multiscale_export'), 0):,} rows.",
        f"- Dense multiscale compact export: {next((item['samples'] for item in selected['exports'] if item['key'] == 'dense_multiscale_compact_export'), 0):,} rows.",
        "",
        "## Hugging Face Full Dataset Metadata",
        "",
        f"- Repo: Hugging Face gated dataset `{full['repo_id']}` at `{full['repo_sha']}`.",
        "- Scope note: this is the HF-hosted full dataset version and file-listing metadata, not a local raw-data mirror.",
        f"- {full['summary'].get('file_count_excluding_gitattributes', 0):,} files excluding `.gitattributes`.",
        f"- {full['summary'].get('complete_episode_count', 0):,} complete episode folders ({full['summary'].get('complete_episode_pct', 0):.4f}%).",
        f"- {full['summary'].get('mp4_count', 0):,} MP4 files and {full['summary'].get('annotation_hdf5_count', 0):,} `annotation.hdf5` files.",
        "",
        "## Generated Charts",
        "",
    ]
    for chart in payload["charts"]:
        lines.append(f"- {chart['title']}: `{chart['path']}`")
    return "\n".join(lines) + "\n"


def build_payload(root: Path) -> dict[str, Any]:
    payload = {
        "status": "pass",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "sources": [
            {"scope": "public_sample", "path": "docs/data/raw_sample_files.json"},
            {"scope": "public_sample", "path": "docs/data/single_episode_explorer.json"},
            {"scope": "public_sample", "path": "results/episode_task_suite/windows.csv"},
            {"scope": "selected_128", "path": "docs/data/xperience10m_128_episode_feature_index.json"},
            {"scope": "selected_128", "path": "results/omni_finetune/xperience10m_128_episode_selection.csv"},
            {"scope": "selected_128", "path": "results/omni_finetune/multi_episode_128_task_baselines/windows.csv"},
            {"scope": "full_hf_dataset", "path": "results/omni_finetune/full_dataset_metadata_audit.json"},
        ],
        "public_sample": build_public_sample(root),
        "selected_128": build_selected_128(root),
        "full_hf_dataset": build_full_hf_dataset(root),
    }
    payload["charts"] = [
        {
            "title": "Scope ladder",
            "path": "assets/charts/data_explorer_scope_ladder.svg",
            "question": "How do the public sample, selected 128 episodes, and Hugging Face gated full dataset version differ in scale?",
        },
        {
            "title": "Public sample feature dimensions",
            "path": "assets/charts/data_explorer_sample_feature_modalities.svg",
            "question": "Which modality groups dominate the one-sample task input?",
        },
        {
            "title": "Public sample action distribution",
            "path": "assets/charts/data_explorer_sample_action_distribution.svg",
            "question": "Which action labels occupy the most 20-frame windows in the sample?",
        },
        {
            "title": "Selected-128 split rows",
            "path": "assets/charts/data_explorer_selected128_split_rows.svg",
            "question": "How many rows are available per selected-128 export and split?",
        },
        {
            "title": "Hugging Face full dataset file composition",
            "path": "assets/charts/data_explorer_full_file_composition.svg",
            "question": "What file types dominate the Hugging Face gated full-dataset metadata?",
        },
    ]
    return payload


def build_charts(root: Path, payload: dict[str, Any]) -> None:
    chart_dir = root / "docs/assets/charts"
    plot_scope_ladder(payload, chart_dir / "data_explorer_scope_ladder.svg")
    plot_feature_breakdown(payload, chart_dir / "data_explorer_sample_feature_modalities.svg")
    plot_action_distribution(payload, chart_dir / "data_explorer_sample_action_distribution.svg")
    plot_selected_split_windows(payload, chart_dir / "data_explorer_selected128_split_rows.svg")
    plot_full_file_composition(payload, chart_dir / "data_explorer_full_file_composition.svg")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=repo_root(), help="Repository root")
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--report-output", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    payload = build_payload(root)
    build_charts(root, payload)

    json_output = args.json_output or (root / "docs/data/data_explorer_analysis.json")
    report_output = args.report_output or (root / "DATA_EXPLORER_ANALYSIS.md")
    write_json(json_output, payload)
    report_output.write_text(render_markdown(payload))
    print(f"PASS: wrote {json_output.relative_to(root)}")
    print(f"PASS: wrote {report_output.relative_to(root)}")
    for chart in payload["charts"]:
        print(f"PASS: wrote {chart['path']}")


if __name__ == "__main__":
    main()

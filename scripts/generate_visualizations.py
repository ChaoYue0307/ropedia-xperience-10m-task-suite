#!/usr/bin/env python3
"""
Generate static SVG visualizations and website data for the Ropedia task suite.

No plotting dependencies are required; this uses only the Python standard
library so the repo stays easy to run.

The polished GitHub Pages homepage in docs/index.html is hand-curated and is
not overwritten by this script. This script refreshes docs/assets/*.svg,
docs/assets/charts/*.svg, and docs/data/summary_metrics.json.
"""

from __future__ import annotations

import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DOCS = ROOT / "docs"
ASSETS = DOCS / "assets"
CHARTS = ASSETS / "charts"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def svg_bar_chart(path: Path, title: str, rows: list[tuple[str, float]], x_label: str = "score", max_value: float | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 1100
    row_h = 34
    top = 78
    left = 310
    right = 70
    height = top + row_h * len(rows) + 70
    max_value = max_value if max_value is not None else max([v for _, v in rows] + [1.0])
    max_value = max(max_value, 1e-9)
    plot_w = width - left - right
    colors = ["#2563eb", "#059669", "#ea580c", "#7b5d12", "#0891b2", "#dc2626"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="32" y="42" font-family="Arial, sans-serif" font-size="26" font-weight="700" fill="#111827">{html.escape(title)}</text>',
        f'<text x="{left}" y="{height - 24}" font-family="Arial, sans-serif" font-size="13" fill="#6b7280">{html.escape(x_label)}</text>',
    ]
    for tick in range(6):
        x = left + plot_w * tick / 5
        val = max_value * tick / 5
        parts.append(f'<line x1="{x:.1f}" y1="{top - 18}" x2="{x:.1f}" y2="{height - 50}" stroke="#e5e7eb" stroke-width="1"/>')
        parts.append(f'<text x="{x:.1f}" y="{height - 30}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#6b7280">{val:.2f}</text>')
    for i, (label, value) in enumerate(rows):
        y = top + i * row_h
        bar_w = max(0.0, min(value / max_value, 1.0)) * plot_w
        color = colors[i % len(colors)]
        parts.append(f'<text x="{left - 14}" y="{y + 21}" text-anchor="end" font-family="Arial, sans-serif" font-size="14" fill="#111827">{html.escape(label)}</text>')
        parts.append(f'<rect x="{left}" y="{y + 5}" width="{bar_w:.1f}" height="20" rx="4" fill="{color}"/>')
        parts.append(f'<text x="{left + bar_w + 8:.1f}" y="{y + 21}" font-family="Arial, sans-serif" font-size="13" fill="#374151">{value:.4f}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def svg_feature_blocks(path: Path, feature_manifest: list[dict]) -> None:
    rows = [(block["name"], float(block["dim"])) for block in feature_manifest]
    svg_bar_chart(path, "All-Modality Feature Blocks", rows, x_label="feature dimensions", max_value=max(v for _, v in rows) * 1.08)


def svg_pipeline_diagram(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suite = summary["suite"]
    task_count = len(suite["tasks"])
    width, height = 1400, 760
    boxes = [
        (60, 110, 250, 132, "1. Raw public sample", [
            "annotation.hdf5",
            "6 video files",
            f"{suite['num_frames']:,} aligned frames",
        ], "#1f63e9"),
        (365, 110, 250, 132, "2. HOMIE loader", [
            "mocap, IMU, depth",
            "caption map",
            "SLAM and calibration",
        ], "#008b9a"),
        (670, 110, 250, 132, "3. Window builder", [
            f"{suite['window_frames']}-frame windows",
            f"{suite['stride_frames']}-frame stride",
            f"{suite['num_windows']:,} windows",
        ], "#0a7f55"),
        (975, 110, 300, 132, "4. Feature vector", [
            f"{suite['feature_dim']:,} dimensions",
            "17 named feature blocks",
            "stored manifest",
        ], "#b65b04"),
        (60, 380, 360, 168, "5. Baseline models", [
            "motion-only action/subtask",
            "all-modality action/subtask",
            "numpy softmax classifier",
            "metrics and predictions",
        ], "#1f63e9"),
        (520, 380, 360, 168, "6. Episode task suite", [
            f"{task_count} supervised/self-supervised tasks",
            "chronological split",
            "retrieval, forecast, alignment",
            "per-task artifacts",
        ], "#008b9a"),
        (980, 380, 300, 168, "7. Published artifacts", [
            "results/**/*.json/csv/npz",
            "docs/data/summary_metrics.json",
            "GitHub Pages dashboard",
            "reproducibility audit",
        ], "#0a7f55"),
    ]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<rect x="0" y="0" width="1400" height="760" fill="#ffffff"/>',
        '<text x="60" y="58" font-family="Arial, sans-serif" font-size="32" font-weight="700" fill="#10141f">Verified Ropedia Episode Pipeline</text>',
        '<text x="60" y="88" font-family="Arial, sans-serif" font-size="16" fill="#5b6475">Generated from committed scripts and metrics; no conceptual placeholder stages.</text>',
    ]
    arrows = [
        (310, 176, 365, 176),
        (615, 176, 670, 176),
        (920, 176, 975, 176),
        (215, 242, 240, 380),
        (1095, 242, 700, 380),
        (420, 464, 520, 464),
        (880, 464, 980, 464),
    ]
    for x1, y1, x2, y2 in arrows:
        parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#cbd5e1" stroke-width="3" marker-end="url(#arrow)"/>')
    parts.insert(1, '<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#cbd5e1"/></marker></defs>')
    for x, y, w, h, title, lines, color in boxes:
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="#ffffff" stroke="#dce2ec" stroke-width="2"/>')
        parts.append(f'<rect x="{x}" y="{y}" width="8" height="{h}" rx="4" fill="{color}"/>')
        parts.append(f'<text x="{x + 24}" y="{y + 34}" font-family="Arial, sans-serif" font-size="18" font-weight="700" fill="#10141f">{html.escape(title)}</text>')
        for i, line in enumerate(lines):
            parts.append(f'<text x="{x + 24}" y="{y + 66 + i * 22}" font-family="Arial, sans-serif" font-size="14" fill="#394255">{html.escape(line)}</text>')
    checks = [
        "Audit check: rerunning scripts to /private/tmp reproduced committed metrics exactly.",
        "Video/depth check: fresh cache read depth plus fisheye_cam0/1/2/3 and stereo_left/right from raw files.",
        "Scope check: this validates one public sample episode, not cross-episode generalization.",
    ]
    parts.append('<rect x="60" y="620" width="1220" height="96" rx="8" fill="#f8fafc" stroke="#dce2ec"/>')
    for i, line in enumerate(checks):
        parts.append(f'<text x="84" y="{650 + i * 24}" font-family="Arial, sans-serif" font-size="15" fill="#273143">{html.escape(line)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def collect_summary() -> dict:
    all_action = read_json(RESULTS / "min_all_modalities_action_model/metrics.json")
    all_subtask = read_json(RESULTS / "min_all_modalities_subtask_model/metrics.json")
    min_action = read_json(RESULTS / "min_action_model/metrics.json")
    min_subtask = read_json(RESULTS / "min_subtask_model/metrics.json")
    suite = read_json(RESULTS / "episode_task_suite/summary_report.json")
    manifest = read_json(RESULTS / "episode_task_suite/feature_manifest.json")
    return {
        "models": {
            "motion_action": min_action,
            "motion_subtask": min_subtask,
            "all_modalities_action": all_action,
            "all_modalities_subtask": all_subtask,
        },
        "suite": suite,
        "feature_manifest": manifest,
    }


def generate_charts(summary: dict) -> None:
    CHARTS.mkdir(parents=True, exist_ok=True)
    svg_pipeline_diagram(ASSETS / "pipeline_diagram.svg", summary)
    model_rows = [
        ("Motion-only action macro-F1", summary["models"]["motion_action"]["macro_f1"]),
        ("All-modality action macro-F1", summary["models"]["all_modalities_action"]["macro_f1"]),
        ("Motion-only subtask macro-F1", summary["models"]["motion_subtask"]["macro_f1"]),
        ("All-modality subtask macro-F1", summary["models"]["all_modalities_subtask"]["macro_f1"]),
    ]
    svg_bar_chart(CHARTS / "model_macro_f1.svg", "Minimal Model Macro-F1 Comparison", model_rows, max_value=1.0)

    suite = summary["suite"]["tasks"]
    task_rows = []
    for task_name, metrics in suite.items():
        score = metrics.get("macro_f1", metrics.get("f1", metrics.get("micro_f1", metrics.get("top5_accuracy", metrics.get("r2", 0.0)))))
        if score is None:
            score = 0.0
        score = max(float(score), 0.0)
        task_rows.append((task_name, score))
    svg_bar_chart(CHARTS / "episode_task_scores.svg", "Episode Task Suite: Main Scores", task_rows, max_value=1.0)
    svg_feature_blocks(CHARTS / "feature_blocks.svg", summary["feature_manifest"])

    retrieval = suite["cross_modal_retrieval"]
    retrieval_rows = [
        ("top1", retrieval["top1_accuracy"]),
        ("top5", retrieval["top5_accuracy"]),
        ("top10", retrieval["top10_accuracy"]),
        ("MRR", retrieval["mrr"]),
    ]
    svg_bar_chart(CHARTS / "cross_modal_retrieval.svg", "Cross-Modal Retrieval", retrieval_rows, max_value=1.0)


def write_summary_data(summary: dict) -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "data").mkdir(parents=True, exist_ok=True)
    (DOCS / "data/summary_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> int:
    summary = collect_summary()
    generate_charts(summary)
    write_summary_data(summary)
    print(f"Wrote pipeline diagram: {ASSETS / 'pipeline_diagram.svg'}")
    print(f"Wrote charts: {CHARTS}")
    print(f"Wrote data: {DOCS / 'data/summary_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

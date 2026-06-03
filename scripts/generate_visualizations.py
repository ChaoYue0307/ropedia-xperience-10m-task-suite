#!/usr/bin/env python3
"""
Generate static SVG visualizations and website data for the Xperience-10M task suite.

No plotting dependencies are required; this uses only the Python standard
library so the repo stays easy to run.

The polished GitHub Pages homepage in docs/index.html is hand-curated and is
not overwritten by this script. This script refreshes docs/assets/*.svg,
docs/assets/charts/*.svg, and docs/data/summary_metrics.json.
"""

from __future__ import annotations

import html
import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DOCS = ROOT / "docs"
ASSETS = DOCS / "assets"
CHARTS = ASSETS / "charts"

OMNI_RELAY = {
    "status": "pending_huggingface_gated_access",
    "dataset": "ropedia-ai/xperience-10m",
    "staging": "prepared_generic_host_to_host_transfer",
    "training_target": "external_multi_gpu_training_host",
    "selection_strategy": "stratified_round_robin_by_top_level_session",
    "target_episodes": 32,
    "selected_sessions": 32,
    "candidate_scan_top_level_sessions": 64,
    "valid_candidates": 680,
    "estimated_bytes": 72031620552,
    "exclude": ["visualization.rrd"],
    "access_status": "Hugging Face returns 403 pending review for the full Xperience-10M gated dataset.",
    "current_scope": "The 32-episode Qwen3-Omni fine-tune requires gated data staging and held-out evaluation.",
}


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
    colors = ["#ccffa0", "#ffffff", "#7ae5c3", "#d8f4a5", "#9bdfff", "#ff8f7a"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#020502"/>',
        '<rect x="18" y="18" width="1064" height="' + str(height - 36) + '" rx="18" fill="#050905" stroke="#ccffa0" stroke-opacity="0.25"/>',
        f'<text x="32" y="42" font-family="Inter Tight, Arial, sans-serif" font-size="26" font-weight="800" fill="#f4f8ef">{html.escape(title)}</text>',
        f'<text x="{left}" y="{height - 24}" font-family="Space Grotesk, Arial, sans-serif" font-size="13" fill="#a5afa2">{html.escape(x_label)}</text>',
    ]
    for tick in range(6):
        x = left + plot_w * tick / 5
        val = max_value * tick / 5
        parts.append(f'<line x1="{x:.1f}" y1="{top - 18}" x2="{x:.1f}" y2="{height - 50}" stroke="#ccffa0" stroke-opacity="0.13" stroke-width="1"/>')
        parts.append(f'<text x="{x:.1f}" y="{height - 30}" text-anchor="middle" font-family="Space Grotesk, Arial, sans-serif" font-size="12" fill="#a5afa2">{val:.2f}</text>')
    for i, (label, value) in enumerate(rows):
        y = top + i * row_h
        bar_w = max(0.0, min(value / max_value, 1.0)) * plot_w
        color = colors[i % len(colors)]
        parts.append(f'<text x="{left - 14}" y="{y + 21}" text-anchor="end" font-family="Space Grotesk, Arial, sans-serif" font-size="14" fill="#dce8d7">{html.escape(label)}</text>')
        parts.append(f'<rect x="{left}" y="{y + 5}" width="{bar_w:.1f}" height="20" rx="4" fill="{color}"/>')
        parts.append(f'<text x="{left + bar_w + 8:.1f}" y="{y + 21}" font-family="Space Grotesk, Arial, sans-serif" font-size="13" fill="#f4f8ef">{value:.4f}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def svg_feature_blocks(path: Path, feature_manifest: list[dict]) -> None:
    rows = [(block["name"], float(block["dim"])) for block in feature_manifest]
    svg_bar_chart(path, "Current Extracted Feature Blocks", rows, x_label="feature dimensions", max_value=max(v for _, v in rows) * 1.08)


def svg_pipeline_diagram(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suite = summary["suite"]
    task_count = len(suite["tasks"])
    width, height = 1400, 760
    boxes = [
        (60, 110, 250, 132, "1. Raw public sample", [
            "annotation.hdf5",
            "6 MP4 videos with audio",
            f"{suite['num_frames']:,} aligned frames",
        ], "#9bdfff"),
        (365, 110, 250, 132, "2. HOMIE loader", [
            "video, depth, pose",
            "mocap, IMU, language",
            "AAC audio features",
        ], "#7ae5c3"),
        (670, 110, 250, 132, "3. Window builder", [
            f"{suite['window_frames']}-frame windows",
            f"{suite['stride_frames']}-frame stride",
            f"{suite['num_windows']:,} windows",
        ], "#ccffa0"),
        (975, 110, 300, 132, "4. Feature vector", [
            f"{suite['feature_dim']:,} dimensions",
            f"{len(summary['feature_manifest'])} named blocks",
            "audio block included",
            "stored manifest",
        ], "#d8f4a5"),
        (60, 380, 360, 168, "5. Baseline models", [
            "motion-only action/subtask",
            "current all-feature action/subtask",
            "numpy softmax classifier",
            "metrics and predictions",
        ], "#9bdfff"),
        (520, 380, 360, 168, "6. Ropedia Xperience-10M suite", [
            f"{task_count} supervised/self-supervised tasks",
            "chronological split",
            "retrieval, forecast, alignment",
            "per-task artifacts",
        ], "#7ae5c3"),
        (980, 380, 300, 168, "7. Published artifacts", [
            "results/**/*.json/csv/npz",
            "docs/data/summary_metrics.json",
            "GitHub Pages dashboard",
            "reproducibility check",
        ], "#ccffa0"),
    ]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#020502"/>',
        '<rect x="0" y="0" width="1400" height="760" fill="#020502"/>',
        '<rect x="0" y="0" width="1400" height="760" fill="url(#dotgrid)" opacity="0.55"/>',
        '<circle cx="1120" cy="132" r="170" fill="#ccffa0" opacity="0.10"/>',
        '<text x="60" y="58" font-family="Inter Tight, Arial, sans-serif" font-size="32" font-weight="800" fill="#f4f8ef">Verified Ropedia Xperience-10M Pipeline</text>',
        '<text x="60" y="88" font-family="Space Grotesk, Arial, sans-serif" font-size="16" fill="#a5afa2">Generated from committed scripts and metrics with traceable stage labels.</text>',
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
        parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#ccffa0" stroke-opacity="0.54" stroke-width="3" marker-end="url(#arrow)"/>')
    parts.insert(1, '<defs><pattern id="dotgrid" width="18" height="18" patternUnits="userSpaceOnUse"><circle cx="2" cy="2" r="1.2" fill="#ccffa0" opacity="0.20"/></pattern><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#ccffa0" fill-opacity="0.72"/></marker></defs>')
    for x, y, w, h, title, lines, color in boxes:
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="#061006" stroke="#ccffa0" stroke-opacity="0.26" stroke-width="2"/>')
        parts.append(f'<rect x="{x}" y="{y}" width="8" height="{h}" rx="4" fill="{color}"/>')
        parts.append(f'<text x="{x + 24}" y="{y + 34}" font-family="Inter Tight, Arial, sans-serif" font-size="18" font-weight="800" fill="#f4f8ef">{html.escape(title)}</text>')
        for i, line in enumerate(lines):
            parts.append(f'<text x="{x + 24}" y="{y + 66 + i * 22}" font-family="Space Grotesk, Arial, sans-serif" font-size="14" fill="#dce8d7">{html.escape(line)}</text>')
    checks = [
        "Reproduction check: rerunning scripts to an ignored scratch workspace reproduced committed metrics exactly.",
        "Modality check: sample covers video, AAC audio, depth, pose/SLAM, mocap, IMU, and language annotation.",
        "Feature check: current manifest has video/depth/pose/mocap/IMU/language blocks plus a real AAC audio block.",
        "Scope check: this validates one public sample episode, not cross-episode generalization.",
    ]
    parts.append('<rect x="60" y="620" width="1220" height="96" rx="8" fill="#071207" stroke="#ccffa0" stroke-opacity="0.24"/>')
    for i, line in enumerate(checks):
        parts.append(f'<text x="84" y="{650 + i * 24}" font-family="Space Grotesk, Arial, sans-serif" font-size="15" fill="#dce8d7">{html.escape(line)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def feature_dim(feature_manifest: list[dict], include: list[str] | None = None, exclude: list[str] | None = None) -> int:
    include = include or []
    exclude = exclude or []
    total = 0
    for block in feature_manifest:
        name = block["name"]
        if include and not any(name == prefix or name.startswith(prefix) for prefix in include):
            continue
        if exclude and any(name == prefix or name.startswith(prefix) for prefix in exclude):
            continue
        total += int(block["dim"])
    return total


def metric_text(task_name: str, metrics: dict) -> str:
    if task_name == "hand_trajectory_forecast":
        return f"MPJPE {metrics['mpjpe']:.4f}"
    if task_name == "cross_modal_retrieval":
        return f"top-5 {metrics['top5_accuracy']:.4f}"
    if task_name == "caption_grounding":
        return f"MRR {metrics['mrr']:.4f}"
    if task_name == "object_relevance":
        return f"micro-F1 {metrics['micro_f1']:.4f}"
    if task_name == "modality_reconstruction":
        return f"R2 {metrics['r2']:.4f}"
    if task_name in {"temporal_order", "misalignment_detection"}:
        return f"F1 {metrics['f1']:.4f}"
    if "macro_f1" in metrics:
        return f"macro-F1 {metrics['macro_f1']:.4f}"
    if "accuracy" in metrics:
        return f"accuracy {metrics['accuracy']:.4f}"
    return "metric in summary_report.json"


def metric_text_with_neural(task_name: str, metrics: dict, neural_tasks: dict) -> str:
    text = metric_text(task_name, metrics)
    neural_metrics = neural_tasks.get(task_name)
    if not neural_metrics or "error" in neural_metrics:
        return text
    return f"min {text}; NN {metric_text(task_name, neural_metrics)}"


def draw_text_block(parts: list[str], x: int, y: int, lines: list[str], size: int = 13, color: str = "#dce8d7", weight: str = "500", max_chars: int = 42, line_h: int = 18) -> int:
    cursor = y
    for line in lines:
        wrapped = textwrap.wrap(line, width=max_chars) or [""]
        for item in wrapped:
            parts.append(f'<text x="{x}" y="{cursor}" font-family="Space Grotesk, Arial, sans-serif" font-size="{size}" font-weight="{weight}" fill="{color}">{html.escape(item)}</text>')
            cursor += line_h
    return cursor


def task_architecture_rows(summary: dict) -> list[dict]:
    suite = summary["suite"]
    tasks = suite["tasks"]
    neural_tasks = suite.get("neural_tasks", {})
    manifest = summary["feature_manifest"]
    all_dim = int(suite["feature_dim"])
    no_contact_text_dim = feature_dim(manifest, exclude=["body_contacts", "caption_objects_interaction_text"])
    no_text_dim = feature_dim(manifest, exclude=["caption_objects_interaction_text"])
    sensor_dim = no_text_dim
    text_dim = feature_dim(manifest, include=["caption_objects_interaction_text"])
    motion_dim = feature_dim(manifest, include=["hand_", "body_joints", "body_contacts", "camera_", "imu_"])
    motion_audio_dim = feature_dim(manifest, include=["hand_", "body_joints", "body_contacts", "camera_", "imu_", "audio_"])
    visual_dim = feature_dim(manifest, include=["depth_confidence", "video_"])
    visual_audio_dim = feature_dim(manifest, include=["depth_confidence", "video_", "audio_"])
    pair_dim = all_dim * 3
    align_dim = motion_dim + visual_audio_dim

    return [
        {
            "task": "timeline_action",
            "family": "softmax",
            "input": f"X_all window, {all_dim:,}d",
            "head": "minimal linear softmax; optional NN MLP softmax",
            "output": f"current action class, {tasks['timeline_action']['num_classes']} classes",
            "metric": metric_text_with_neural("timeline_action", tasks["timeline_action"], neural_tasks),
        },
        {
            "task": "timeline_subtask",
            "family": "softmax",
            "input": f"X_all window, {all_dim:,}d",
            "head": "minimal linear softmax; optional NN MLP softmax",
            "output": f"current subtask class, {tasks['timeline_subtask']['num_classes']} classes",
            "metric": metric_text_with_neural("timeline_subtask", tasks["timeline_subtask"], neural_tasks),
        },
        {
            "task": "transition_detection",
            "family": "softmax",
            "input": f"X_all window, {all_dim:,}d",
            "head": "minimal linear softmax; optional NN MLP softmax",
            "output": "steady vs transition near action boundary",
            "metric": f"{metric_text_with_neural('transition_detection', tasks['transition_detection'], neural_tasks)}; boundary-F1 {tasks['transition_detection']['boundary_f1']:.4f}",
        },
        {
            "task": "next_action",
            "family": "softmax",
            "input": f"X_all at time t, {all_dim:,}d",
            "head": "minimal linear softmax; optional NN MLP softmax",
            "output": f"action at t+{tasks['next_action'].get('future_frames', 20)} frames",
            "metric": metric_text_with_neural("next_action", tasks["next_action"], neural_tasks),
        },
        {
            "task": "hand_trajectory_forecast",
            "family": "ridge",
            "input": f"X_all at time t, {all_dim:,}d",
            "head": "minimal dual ridge; optional NN MLP regression",
            "output": f"future hand joints, {tasks['hand_trajectory_forecast']['target_dim']}d",
            "metric": metric_text_with_neural("hand_trajectory_forecast", tasks["hand_trajectory_forecast"], neural_tasks),
        },
        {
            "task": "contact_prediction",
            "family": "softmax",
            "input": f"X without contact/text leakage, {no_contact_text_dim:,}d",
            "head": "minimal linear softmax; optional NN MLP softmax",
            "output": "any body contact in window; degenerate one-class sample",
            "metric": metric_text_with_neural("contact_prediction", tasks["contact_prediction"], neural_tasks),
        },
        {
            "task": "object_relevance",
            "family": "multilabel",
            "input": f"X without caption text, {no_text_dim:,}d",
            "head": "minimal sigmoid logistic; optional NN MLP multilabel",
            "output": f"multi-hot object set, {tasks['object_relevance']['num_objects']} objects",
            "metric": metric_text_with_neural("object_relevance", tasks["object_relevance"], neural_tasks),
        },
        {
            "task": "caption_grounding",
            "family": "ridge+rank",
            "input": f"sensor {sensor_dim:,}d -> text space {text_dim:,}d",
            "head": "minimal ridge or NN MLP projection, then cosine rank",
            "output": "text query retrieves matching time window",
            "metric": metric_text_with_neural("caption_grounding", tasks["caption_grounding"], neural_tasks),
        },
        {
            "task": "cross_modal_retrieval",
            "family": "ridge+rank",
            "input": f"motion/IMU/camera/audio {motion_audio_dim:,}d -> visual {visual_dim:,}d",
            "head": "minimal ridge or NN MLP projection, then cosine rank",
            "output": "retrieve matching depth/video window",
            "metric": metric_text_with_neural("cross_modal_retrieval", tasks["cross_modal_retrieval"], neural_tasks),
        },
        {
            "task": "modality_reconstruction",
            "family": "ridge",
            "input": f"motion/IMU/camera/audio {motion_audio_dim:,}d",
            "head": "minimal dual ridge; optional NN MLP regression",
            "output": f"depth/video feature vector, {visual_dim:,}d",
            "metric": metric_text_with_neural("modality_reconstruction", tasks["modality_reconstruction"], neural_tasks),
        },
        {
            "task": "temporal_order",
            "family": "softmax",
            "input": f"concat[x_t, x_t+1, diff], {pair_dim:,}d",
            "head": "minimal binary softmax; optional NN MLP softmax",
            "output": "correct vs reversed adjacent windows",
            "metric": metric_text_with_neural("temporal_order", tasks["temporal_order"], neural_tasks),
        },
        {
            "task": "misalignment_detection",
            "family": "softmax",
            "input": f"concat[motion_t, visual+audio_t/shifted], {align_dim:,}d",
            "head": "minimal binary softmax; optional NN MLP softmax",
            "output": "aligned vs shifted by 8 windows",
            "metric": metric_text_with_neural("misalignment_detection", tasks["misalignment_detection"], neural_tasks),
        },
    ]


def svg_task_architectures(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suite = summary["suite"]
    rows = task_architecture_rows(summary)
    family_colors = {
        "softmax": "#9bdfff",
        "ridge": "#ccffa0",
        "ridge+rank": "#7ae5c3",
        "multilabel": "#d8f4a5",
    }
    width, height = 1500, 1840
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<defs><pattern id="dotgrid2" width="18" height="18" patternUnits="userSpaceOnUse"><circle cx="2" cy="2" r="1.2" fill="#ccffa0" opacity="0.18"/></pattern><marker id="arrow2" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#ccffa0" fill-opacity="0.72"/></marker></defs>',
        '<rect width="100%" height="100%" fill="#020502"/>',
        '<rect width="100%" height="100%" fill="url(#dotgrid2)" opacity="0.58"/>',
        '<circle cx="1190" cy="150" r="210" fill="#ccffa0" opacity="0.08"/>',
        '<text x="60" y="56" font-family="Inter Tight, Arial, sans-serif" font-size="34" font-weight="800" fill="#f4f8ef">Minimal Architectures for 12 Ropedia Xperience-10M Tasks</text>',
        '<text x="60" y="88" font-family="Space Grotesk, Arial, sans-serif" font-size="16" fill="#a5afa2">Generated from scripts/episode_task_suite.py semantics and committed summary metrics. These are minimal baselines, not deep foundation models.</text>',
    ]

    setup = [
        (60, 122, 310, 110, "Shared episode windows", [
            f"{suite['num_frames']:,} frames -> {suite['num_windows']:,} windows",
            f"{suite['window_frames']}-frame window, {suite['stride_frames']}-frame stride",
            "chronological 70/30 split",
        ], "#9bdfff"),
        (410, 122, 310, 110, "Feature vector", [
            f"X_all = {suite['feature_dim']:,} dimensions",
            f"{len(summary['feature_manifest'])} named blocks incl. audio",
            "mean/std fit on train only",
        ], "#7ae5c3"),
        (760, 122, 320, 110, "Reusable heads", [
            "linear softmax classifier",
            "dual ridge regression/projection",
            "multi-label logistic + cosine rank",
        ], "#ccffa0"),
        (1120, 122, 320, 110, "Artifacts", [
            "metrics.json, predictions.csv/npz",
            "model.npz with scaler and weights",
            "summary_report.json source of numbers",
        ], "#d8f4a5"),
    ]
    for i in range(len(setup) - 1):
        x1 = setup[i][0] + setup[i][2]
        x2 = setup[i + 1][0]
        y = setup[i][1] + 55
        parts.append(f'<line x1="{x1 + 12}" y1="{y}" x2="{x2 - 14}" y2="{y}" stroke="#ccffa0" stroke-opacity="0.54" stroke-width="3" marker-end="url(#arrow2)"/>')
    for x, y, w, h, title, lines, color in setup:
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="#061006" stroke="#ccffa0" stroke-opacity="0.26" stroke-width="2"/>')
        parts.append(f'<rect x="{x}" y="{y}" width="8" height="{h}" rx="4" fill="{color}"/>')
        parts.append(f'<text x="{x + 24}" y="{y + 31}" font-family="Inter Tight, Arial, sans-serif" font-size="18" font-weight="800" fill="#f4f8ef">{html.escape(title)}</text>')
        draw_text_block(parts, x + 24, y + 58, lines, size=13, color="#dce8d7", max_chars=34, line_h=18)

    families = [
        ("Softmax classifier", "logits = z(X)W + b; CE + L2; class weights for classifiers", "#9bdfff", 60, 270),
        ("Ridge regression/projection", "closed-form dual ridge on z(X), z(Y); used for forecast and reconstruction", "#ccffa0", 780, 270),
        ("Ridge + cosine ranking", "project one modality into another feature space, then rank candidates by cosine", "#7ae5c3", 60, 394),
        ("Multi-label logistic", "sigmoid heads for object vocabulary; threshold 0.5 with top-1 fallback", "#d8f4a5", 780, 394),
    ]
    for title, desc, color, x, y in families:
        parts.append(f'<rect x="{x}" y="{y}" width="660" height="100" rx="8" fill="#071207" stroke="#ccffa0" stroke-opacity="0.22"/>')
        parts.append(f'<text x="{x + 18}" y="{y + 33}" font-family="Inter Tight, Arial, sans-serif" font-size="18" font-weight="800" fill="{color}">{html.escape(title)}</text>')
        draw_text_block(parts, x + 18, y + 60, [desc], size=13, color="#dce8d7", max_chars=76, line_h=18)

    card_w, card_h = 440, 248
    gap_x, gap_y = 30, 30
    start_x, start_y = 60, 540
    for idx, row in enumerate(rows):
        col, card_row = idx % 3, idx // 3
        x = start_x + col * (card_w + gap_x)
        y = start_y + card_row * (card_h + gap_y)
        color = family_colors[row["family"]]
        parts.append(f'<rect x="{x}" y="{y}" width="{card_w}" height="{card_h}" rx="8" fill="#061006" stroke="#ccffa0" stroke-opacity="0.24" stroke-width="2"/>')
        parts.append(f'<rect x="{x}" y="{y}" width="8" height="{card_h}" rx="4" fill="{color}"/>')
        parts.append(f'<rect x="{x + 20}" y="{y + 18}" width="96" height="24" rx="6" fill="#071207" stroke="{color}" stroke-opacity="0.72"/>')
        parts.append(f'<text x="{x + 68}" y="{y + 35}" text-anchor="middle" font-family="Space Grotesk, Arial, sans-serif" font-size="11" font-weight="800" fill="{color}">{html.escape(row["family"])}</text>')
        parts.append(f'<text x="{x + 20}" y="{y + 72}" font-family="Inter Tight, Arial, sans-serif" font-size="20" font-weight="800" fill="#f4f8ef">{html.escape(row["task"])}</text>')
        cursor = y + 104
        for label in ("input", "head", "output", "metric"):
            parts.append(f'<text x="{x + 20}" y="{cursor}" font-family="Space Grotesk, Arial, sans-serif" font-size="12" font-weight="800" fill="{color}">{label.upper()}</text>')
            cursor = draw_text_block(parts, x + 92, cursor, [row[label]], size=13, color="#dce8d7", max_chars=41, line_h=17)
            cursor += 8

    notes = [
        "Interpretation: this suite tests whether each input/output contract is wired correctly before scaling to many episodes.",
        "Research-grade conclusions need held-out episode splits and stronger sequence/vision-language/robot-policy models.",
    ]
    parts.append('<rect x="60" y="1688" width="1380" height="72" rx="8" fill="#071207" stroke="#ccffa0" stroke-opacity="0.22"/>')
    for i, line in enumerate(notes):
        parts.append(f'<text x="84" y="{1718 + i * 24}" font-family="Space Grotesk, Arial, sans-serif" font-size="15" fill="#dce8d7">{html.escape(line)}</text>')
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
        "omni_relay": OMNI_RELAY,
        "models": {
            "motion_action": min_action,
            "motion_subtask": min_subtask,
            "all_modalities_action": all_action,
            "all_modalities_subtask": all_subtask,
        },
        "suite": suite,
        "feature_manifest": manifest,
    }


def task_score(metrics: dict) -> float:
    score = metrics.get("macro_f1", metrics.get("f1", metrics.get("micro_f1", metrics.get("top5_accuracy", metrics.get("r2", 0.0)))))
    if score is None:
        score = 0.0
    return max(float(score), 0.0)


def generate_charts(summary: dict) -> None:
    CHARTS.mkdir(parents=True, exist_ok=True)
    svg_pipeline_diagram(ASSETS / "pipeline_diagram.svg", summary)
    svg_task_architectures(ASSETS / "task_architectures.svg", summary)
    model_rows = [
        ("Motion-only action macro-F1", summary["models"]["motion_action"]["macro_f1"]),
        ("Current all-feature action macro-F1", summary["models"]["all_modalities_action"]["macro_f1"]),
        ("Motion-only subtask macro-F1", summary["models"]["motion_subtask"]["macro_f1"]),
        ("Current all-feature subtask macro-F1", summary["models"]["all_modalities_subtask"]["macro_f1"]),
    ]
    svg_bar_chart(CHARTS / "model_macro_f1.svg", "Minimal Model Macro-F1 Comparison", model_rows, max_value=1.0)

    suite = summary["suite"]["tasks"]
    task_rows = []
    for task_name, metrics in suite.items():
        task_rows.append((task_name, task_score(metrics)))
    svg_bar_chart(CHARTS / "episode_task_scores.svg", "Ropedia Xperience-10M Suite: Main Scores", task_rows, max_value=1.0)

    neural = summary["suite"].get("neural_tasks", {})
    if neural:
        neural_rows = [(task_name, task_score(metrics)) for task_name, metrics in neural.items() if "error" not in metrics]
        if neural_rows:
            svg_bar_chart(CHARTS / "episode_task_scores_neural_mlp.svg", "Ropedia Xperience-10M Suite: Neural MLP Main Scores", neural_rows, max_value=1.0)

        comparison_rows = []
        for task_name, metrics in suite.items():
            comparison_rows.append((f"{task_name} minimal", task_score(metrics)))
            neural_metrics = neural.get(task_name)
            if neural_metrics and "error" not in neural_metrics:
                comparison_rows.append((f"{task_name} neural", task_score(neural_metrics)))
        if comparison_rows:
            svg_bar_chart(CHARTS / "episode_task_scores_minimal_vs_neural.svg", "Episode Task Scores: Minimal vs Neural MLP", comparison_rows, max_value=1.0)
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
    (DOCS / "data/summary_metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    summary = collect_summary()
    generate_charts(summary)
    write_summary_data(summary)
    print(f"Wrote pipeline diagram: {ASSETS / 'pipeline_diagram.svg'}")
    print(f"Wrote task architectures diagram: {ASSETS / 'task_architectures.svg'}")
    print(f"Wrote charts: {CHARTS}")
    print(f"Wrote data: {DOCS / 'data/summary_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Render a ChatGPT-image-backed 12-task infographic.

The background bitmap is AI-generated. The task names, inputs, and metrics are
read from results/episode_task_suite/summary_report.json so the published image
does not rely on image-model text generation.
"""

from __future__ import annotations

import argparse
import html
import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "results/episode_task_suite/summary_report.json"
DEFAULT_BASE = ROOT / "docs/assets/task_suite_infographic_base.png"
DEFAULT_OUTPUT = ROOT / "docs/assets/task_suite_infographic.png"


GROUPS = [
    {
        "name": "Label + State",
        "color": "#008b9a",
        "left": 94,
        "top": 374,
        "width": 246,
        "tasks": [
            ("timeline_action", "supervised"),
            ("timeline_subtask", "supervised"),
            ("next_action", "supervised"),
        ],
    },
    {
        "name": "Prediction + Reconstruction",
        "color": "#1f63e9",
        "left": 472,
        "top": 374,
        "width": 248,
        "tasks": [
            ("hand_trajectory_forecast", "forecast"),
            ("modality_reconstruction", "forecast"),
            ("contact_prediction", "supervised"),
        ],
    },
    {
        "name": "Grounding + Retrieval",
        "color": "#b65b04",
        "left": 848,
        "top": 374,
        "width": 220,
        "tasks": [
            ("caption_grounding", "retrieval"),
            ("cross_modal_retrieval", "retrieval"),
            ("object_relevance", "supervised"),
        ],
    },
    {
        "name": "Temporal Diagnostics",
        "color": "#b42318",
        "left": 1202,
        "top": 374,
        "width": 244,
        "tasks": [
            ("transition_detection", "diagnostic"),
            ("temporal_order", "diagnostic"),
            ("misalignment_detection", "diagnostic"),
        ],
    },
]


def load_summary() -> dict:
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


def fmt(value: float) -> str:
    return f"{float(value):.4f}"


def metric_for(task_name: str, metrics: dict) -> tuple[str, str]:
    if task_name == "hand_trajectory_forecast":
        return "MPJPE", fmt(metrics["mpjpe"])
    if task_name == "cross_modal_retrieval":
        return "top-5", fmt(metrics["top5_accuracy"])
    if task_name == "caption_grounding":
        return "MRR", fmt(metrics["mrr"])
    if task_name == "object_relevance":
        return "micro-F1", fmt(metrics["micro_f1"])
    if task_name == "modality_reconstruction":
        return "R2", fmt(metrics["r2"])
    if task_name in {"temporal_order", "misalignment_detection"}:
        return "F1", fmt(metrics["f1"])
    if "macro_f1" in metrics:
        return "macro-F1", fmt(metrics["macro_f1"])
    if "accuracy" in metrics:
        return "accuracy", fmt(metrics["accuracy"])
    raise KeyError(f"No main metric configured for {task_name}")


def short_io(task_name: str, metrics: dict) -> str:
    custom = {
        "timeline_action": "all modalities -> action label",
        "timeline_subtask": "all modalities -> subtask label",
        "transition_detection": "all modalities -> boundary / steady",
        "next_action": "window at t -> action at t+20",
        "hand_trajectory_forecast": "all modalities -> future hand joints",
        "contact_prediction": "non-contact modalities -> contact",
        "object_relevance": "non-caption modalities -> object set",
        "caption_grounding": "text query -> matching window",
        "cross_modal_retrieval": "motion / IMU / camera -> depth / video",
        "modality_reconstruction": "motion / IMU / camera -> depth / video vec",
        "temporal_order": "two windows -> correct order?",
        "misalignment_detection": "motion + visual -> aligned / shifted",
    }
    return custom.get(task_name, metrics.get("input", ""))


def task_html(task_name: str, kind: str, metrics: dict, top: int, group: dict) -> str:
    label, value = metric_for(task_name, metrics)
    io = short_io(task_name, metrics)
    name_size = 17 if len(task_name) > 22 else 18
    return f"""
      <section class="task" style="left:{group['left']}px;top:{top}px;width:{group['width']}px;--accent:{group['color']};">
        <div class="kind">{html.escape(kind)}</div>
        <div class="task-name" style="font-size:{name_size}px;">{html.escape(task_name)}</div>
        <div class="io">{html.escape(io)}</div>
        <div class="metric"><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>
      </section>
    """


def build_html(summary: dict, base_image: Path) -> str:
    suite = summary["tasks"]
    task_count = len(suite)
    group_headers = []
    cards = []
    row_tops = [374, 552, 730]
    header_lefts = [38, 417, 792, 1143]
    for group, header_left in zip(GROUPS, header_lefts):
        group_headers.append(
            f'<div class="group-title" style="left:{header_left}px;top:333px;color:{group["color"]};">{html.escape(group["name"])}</div>'
        )
        for row_idx, (task_name, kind) in enumerate(group["tasks"]):
            cards.append(task_html(task_name, kind, suite[task_name], row_tops[row_idx], group))

    stats = [
        f"{summary['num_frames']:,} frames",
        f"{summary['num_windows']:,} windows",
        f"{summary['feature_dim']:,} features",
        f"{task_count} tasks",
        "chronological split",
    ]
    stat_html = "".join(f"<span>{html.escape(item)}</span>" for item in stats)
    base_uri = base_image.resolve().as_uri()
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=1536, initial-scale=1">
  <title>Ropedia 12-Task Episode Suite Infographic</title>
  <style>
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; width: 1536px; height: 1024px; background: #ffffff; }}
    body {{
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
      color: #10141f;
    }}
    .canvas {{
      position: relative;
      width: 1536px;
      height: 1024px;
      overflow: hidden;
      background-image: url("{base_uri}");
      background-size: 1536px 1024px;
      background-repeat: no-repeat;
    }}
    .title {{
      position: absolute;
      left: 330px;
      top: 42px;
      width: 876px;
      text-align: center;
    }}
    h1 {{
      margin: 0;
      font-size: 38px;
      line-height: 1.05;
      letter-spacing: 0;
      font-weight: 820;
    }}
    .subtitle {{
      margin-top: 8px;
      color: #425067;
      font-size: 15px;
      line-height: 1.35;
      font-weight: 520;
    }}
    .stats {{
      margin-top: 12px;
      display: flex;
      justify-content: center;
      gap: 8px;
    }}
    .stats span {{
      display: inline-flex;
      align-items: center;
      height: 24px;
      padding: 0 10px;
      border: 1px solid #cdd8e8;
      background: rgba(255, 255, 255, 0.82);
      border-radius: 999px;
      color: #253046;
      font-size: 12px;
      font-weight: 720;
    }}
    .modality {{
      position: absolute;
      top: 256px;
      width: 180px;
      text-align: center;
      font-size: 12px;
      color: #536074;
      font-weight: 720;
      text-transform: uppercase;
      letter-spacing: 0;
    }}
    .group-title {{
      position: absolute;
      width: 322px;
      text-align: center;
      font-size: 18px;
      line-height: 1;
      font-weight: 830;
      letter-spacing: 0;
    }}
    .task {{
      position: absolute;
      padding: 0;
    }}
    .kind {{
      display: inline-flex;
      align-items: center;
      height: 22px;
      padding: 0 8px;
      border-radius: 6px;
      border: 1px solid color-mix(in srgb, var(--accent) 35%, #ffffff);
      color: var(--accent);
      background: rgba(255, 255, 255, 0.76);
      text-transform: uppercase;
      font-size: 10px;
      line-height: 1;
      font-weight: 840;
      letter-spacing: 0;
    }}
    .task-name {{
      margin-top: 7px;
      color: #111827;
      line-height: 1.05;
      font-weight: 850;
      letter-spacing: 0;
      white-space: nowrap;
    }}
    .io {{
      margin-top: 8px;
      min-height: 36px;
      color: #475569;
      font-size: 13.5px;
      line-height: 1.28;
      font-weight: 570;
    }}
    .metric {{
      display: inline-flex;
      align-items: center;
      gap: 9px;
      margin-top: 8px;
      height: 30px;
      padding: 0 10px;
      border-radius: 7px;
      border: 1px solid color-mix(in srgb, var(--accent) 36%, #ffffff);
      background: rgba(255, 255, 255, 0.90);
      box-shadow: 0 7px 20px rgba(16, 20, 31, 0.07);
    }}
    .metric span {{
      color: #64748b;
      font-size: 12px;
      font-weight: 760;
    }}
    .metric strong {{
      color: var(--accent);
      font-size: 16px;
      line-height: 1;
      font-weight: 860;
    }}
    .footer {{
      position: absolute;
      left: 360px;
      top: 932px;
      width: 816px;
      text-align: center;
      color: #536074;
      font-size: 14px;
      font-weight: 650;
    }}
  </style>
</head>
<body>
  <main class="canvas" aria-label="Ropedia 12-task episode suite infographic">
    <div class="title">
      <h1>Ropedia 12-Task Episode Suite</h1>
      <div class="subtitle">All labels and metrics are overlaid from the verified single-episode results.</div>
      <div class="stats">{stat_html}</div>
    </div>
    <div class="modality" style="left:50px;">fisheye video</div>
    <div class="modality" style="left:270px;">depth</div>
    <div class="modality" style="left:530px;">3D / SLAM</div>
    <div class="modality" style="left:770px;">IMU</div>
    <div class="modality" style="left:1030px;">hands</div>
    <div class="modality" style="left:1278px;">text / objects</div>
    {''.join(group_headers)}
    {''.join(cards)}
    <div class="footer">Single public sample episode: useful for pipeline validation and task design, not cross-episode generalization.</div>
  </main>
</body>
</html>
"""


def render_html(html_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "npx",
            "--yes",
            "playwright",
            "screenshot",
            "--full-page",
            "--viewport-size=1536,1024",
            html_path.resolve().as_uri(),
            str(output_path),
        ],
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-image", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--html", type=Path)
    parser.add_argument("--no-export", action="store_true", help="Only write the HTML overlay.")
    args = parser.parse_args()

    summary = load_summary()
    html_text = build_html(summary, args.base_image)
    if args.html is None:
        with tempfile.NamedTemporaryFile("w", suffix=".html", encoding="utf-8", delete=False) as handle:
            handle.write(html_text)
            html_path = Path(handle.name)
    else:
        html_path = args.html
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(html_text, encoding="utf-8")

    if not args.no_export:
        render_html(html_path, args.output)
        print(f"Wrote image: {args.output}")
    print(f"Wrote overlay HTML: {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

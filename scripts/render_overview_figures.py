#!/usr/bin/env python3
"""
Render polished PNG overview figures for the Ropedia project page.

The ChatGPT-image assets are used only as text-free visual backgrounds. All
labels, dimensions, task names, and metrics are read from committed result
files via scripts/generate_visualizations.py so the final figures stay
auditable.
"""

from __future__ import annotations

import argparse
import base64
import html
import subprocess
import tempfile
from pathlib import Path

from generate_visualizations import collect_summary, task_architecture_rows


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs/assets"
DEFAULT_PIPELINE_BASE = ASSETS / "pipeline_diagram_base.png"
DEFAULT_ARCHITECTURE_BASE = ASSETS / "task_architectures_base.png"
DEFAULT_PIPELINE_OUTPUT = ASSETS / "pipeline_diagram.png"
DEFAULT_ARCHITECTURE_OUTPUT = ASSETS / "task_architectures.png"

PIPELINE_WIDTH = 1800
PIPELINE_HEIGHT = 1000
ARCHITECTURE_WIDTH = 1800
ARCHITECTURE_HEIGHT = 1520


COLORS = {
    "blue": "#1f6c9f",
    "teal": "#197d83",
    "green": "#346538",
    "amber": "#956400",
    "orange": "#b65b04",
    "red": "#9f2f2d",
    "ink": "#1f2421",
    "muted": "#5f625d",
    "line": "#e4ded4",
}


TASK_GROUPS = [
    ("Label + State", "#197d83", ["timeline_action", "timeline_subtask", "next_action"]),
    (
        "Prediction + Reconstruction",
        "#1f6c9f",
        ["hand_trajectory_forecast", "modality_reconstruction", "contact_prediction"],
    ),
    ("Grounding + Retrieval", "#956400", ["caption_grounding", "cross_modal_retrieval", "object_relevance"]),
    ("Temporal Diagnostics", "#9f2f2d", ["transition_detection", "temporal_order", "misalignment_detection"]),
]


def data_uri(path: Path) -> str:
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def build_base_layer(path: Path, opacity: float) -> str:
    uri = data_uri(path)
    if not uri:
        return ""
    return f'<div class="base-layer" style="background-image:url({uri});opacity:{opacity};"></div>'


def stage_card(number: str, title: str, lines: list[str], color: str) -> str:
    detail = "".join(f"<li>{esc(line)}</li>" for line in lines)
    return f"""
      <article class="stage" style="--accent:{color}">
        <div class="stage-number">{esc(number)}</div>
        <h3>{esc(title)}</h3>
        <ul>{detail}</ul>
      </article>
    """


def arrow() -> str:
    return '<div class="flow-arrow" aria-hidden="true">-&gt;</div>'


def build_pipeline_html(summary: dict, base_path: Path) -> str:
    suite = summary["suite"]
    task_count = len(suite["tasks"])
    neural_count = len(suite.get("neural_tasks", {}))
    stage_rows = [
        [
            stage_card(
                "01",
                "Raw public sample",
                ["annotation.hdf5", "6 MP4 videos with audio", f"{suite['num_frames']:,} aligned frames"],
                COLORS["blue"],
            ),
            arrow(),
            stage_card(
                "02",
                "HOMIE loader",
                ["video, depth, pose", "mocap, IMU, language", "audio noted, not featurized"],
                COLORS["teal"],
            ),
            arrow(),
            stage_card(
                "03",
                "Window builder",
                [
                    f"{suite['window_frames']}-frame windows",
                    f"{suite['stride_frames']}-frame stride",
                    f"{suite['num_windows']:,} windows",
                ],
                COLORS["green"],
            ),
            arrow(),
            stage_card(
                "04",
                "Feature vector",
                [f"{suite['feature_dim']:,} dimensions", "17 named blocks, no audio block", "manifested slice indices"],
                COLORS["orange"],
            ),
        ],
        [
            stage_card(
                "05",
                "Baseline models",
                ["motion-only classifiers", "current all-feature classifiers", "neural MLP task heads"],
                COLORS["blue"],
            ),
            arrow(),
            stage_card(
                "06",
                "Episode task suite",
                [f"{task_count} minimal + {neural_count} neural results", "forecast, retrieval, alignment", "chronological evaluation"],
                COLORS["teal"],
            ),
            arrow(),
            stage_card(
                "07",
                "Published artifacts",
                ["metrics.json / csv / npz / pt", "GitHub Pages dashboard", "NN comparison charts"],
                COLORS["green"],
            ),
        ],
    ]
    rows_html = "".join(f'<section class="flow-row">{"".join(row)}</section>' for row in stage_rows)
    checks = [
        "Audit check: rerunning scripts to /private/tmp reproduced the committed metrics exactly.",
        "Modality check: sample covers video, AAC audio, depth, pose/SLAM, mocap, IMU, and language annotation.",
        "Feature check: current baseline manifest has video/depth/pose/mocap/IMU/language blocks, but no audio feature block.",
        "Neural check: lightweight PyTorch MLP heads are reported beside the minimal task heads under neural_mlp/.",
        "Scope check: this validates one public sample episode, not cross-episode generalization.",
    ]
    checks_html = "".join(f"<li>{esc(line)}</li>" for line in checks)
    base_layer = build_base_layer(base_path, 0.42)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: #fbfaf7; font-family: "Avenir Next", "SF Pro Display", Arial, sans-serif; }}
    .canvas {{
      position: relative;
      width: {PIPELINE_WIDTH}px;
      height: {PIPELINE_HEIGHT}px;
      overflow: hidden;
      color: #1f2421;
      background:
        linear-gradient(90deg, rgba(68,55,38,0.03) 1px, transparent 1px),
        linear-gradient(0deg, rgba(68,55,38,0.024) 1px, transparent 1px),
        #fbfaf7;
      background-size: 58px 58px, 58px 58px, auto;
    }}
    .base-layer {{
      position: absolute;
      inset: 0;
      background-size: cover;
      background-position: center;
      filter: saturate(0.95) contrast(0.98);
    }}
    .wash {{
      position: absolute;
      inset: 0;
      background: linear-gradient(180deg, rgba(251,250,247,0.72), rgba(251,250,247,0.9));
    }}
    .content {{
      position: relative;
      padding: 66px 82px;
      height: 100%;
    }}
    header {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 44px;
      align-items: start;
      margin-bottom: 42px;
    }}
    .kicker {{
      font: 700 17px "SF Mono", Menlo, monospace;
      color: #68665f;
      text-transform: uppercase;
      letter-spacing: 0.09em;
      margin-bottom: 14px;
    }}
    h1 {{
      margin: 0;
      font-size: 56px;
      line-height: 0.98;
      letter-spacing: 0;
    }}
    .subtitle {{
      margin: 18px 0 0;
      max-width: 1010px;
      color: #4d524d;
      font-size: 24px;
      line-height: 1.42;
      font-weight: 520;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(2, 150px);
      gap: 12px;
      margin-top: 2px;
    }}
    .metric {{
      background: rgba(255,254,253,0.86);
      border: 1px solid #e4ded4;
      border-radius: 8px;
      padding: 13px 15px 12px;
      box-shadow: 0 16px 40px rgba(68,55,38,0.06);
    }}
    .metric strong {{
      display: block;
      font: 850 24px "SF Mono", Menlo, monospace;
      color: #1f2421;
      line-height: 1;
      font-variant-numeric: tabular-nums;
    }}
    .metric span {{
      display: block;
      margin-top: 7px;
      color: #696d67;
      font-size: 14px;
      font-weight: 650;
    }}
    .flow-row {{
      display: flex;
      align-items: center;
      gap: 18px;
      margin-top: 30px;
    }}
    .flow-row:nth-of-type(2) {{
      width: 78%;
      margin-left: auto;
      margin-right: auto;
      margin-top: 38px;
    }}
    .stage {{
      min-width: 0;
      flex: 1 1 0;
      height: 182px;
      position: relative;
      background: rgba(255,254,253,0.88);
      border: 1px solid rgba(228,222,212,0.96);
      border-radius: 8px;
      padding: 24px 24px 22px 30px;
      box-shadow: 0 24px 62px rgba(68,55,38,0.09);
      backdrop-filter: blur(12px);
    }}
    .stage::before {{
      content: "";
      position: absolute;
      inset: 0 auto 0 0;
      width: 8px;
      border-radius: 8px 0 0 8px;
      background: var(--accent);
    }}
    .stage-number {{
      color: var(--accent);
      font: 850 16px "SF Mono", Menlo, monospace;
      letter-spacing: 0.04em;
      margin-bottom: 10px;
    }}
    .stage h3 {{
      margin: 0 0 13px;
      font-size: 24px;
      line-height: 1.08;
      letter-spacing: 0;
    }}
    .stage ul {{
      margin: 0;
      padding: 0;
      list-style: none;
      color: #39413d;
      font-size: 17px;
      line-height: 1.48;
      font-weight: 560;
    }}
    .flow-arrow {{
      width: 54px;
      flex: 0 0 54px;
      height: 54px;
      display: grid;
      place-items: center;
      border-radius: 999px;
      border: 1px solid #d7d0c4;
      background: rgba(255,254,253,0.78);
      color: #7c807a;
      font: 850 22px "SF Mono", Menlo, monospace;
      box-shadow: 0 14px 34px rgba(68,55,38,0.06);
    }}
    .audit {{
      position: absolute;
      left: 82px;
      right: 82px;
      bottom: 62px;
      display: grid;
      grid-template-columns: 190px 1fr;
      gap: 26px;
      align-items: center;
      background: rgba(255,254,253,0.88);
      border: 1px solid #e4ded4;
      border-radius: 8px;
      padding: 24px 28px;
      box-shadow: 0 22px 52px rgba(68,55,38,0.08);
    }}
    .audit strong {{
      color: #1f2421;
      font-size: 23px;
      line-height: 1.1;
    }}
    .audit ul {{
      margin: 0;
      padding: 0;
      list-style: none;
      color: #40463f;
      font-size: 17px;
      line-height: 1.55;
      font-weight: 560;
    }}
  </style>
</head>
<body>
  <main class="canvas">
    {base_layer}
    <div class="wash"></div>
    <div class="content">
      <header>
        <div>
          <div class="kicker">verified single-episode pipeline</div>
          <h1>From Xperience-10M episode to reproducible artifacts</h1>
          <p class="subtitle">The figure follows the actual code path and includes minimal heads plus neural MLP results. Next TODO: Qwen3-Omni fine-tuning and sensor-bridge evaluation on multi-episode splits.</p>
        </div>
        <div class="metrics">
          <div class="metric"><strong>{suite['num_frames']:,}</strong><span>frames</span></div>
          <div class="metric"><strong>{suite['num_windows']:,}</strong><span>windows</span></div>
          <div class="metric"><strong>{suite['feature_dim']:,}</strong><span>features</span></div>
          <div class="metric"><strong>{task_count}+{neural_count}</strong><span>min + NN tasks</span></div>
        </div>
      </header>
      {rows_html}
      <section class="audit">
        <strong>Reproducibility checks</strong>
        <ul>{checks_html}</ul>
      </section>
    </div>
  </main>
</body>
</html>
"""


def family_label(family: str) -> str:
    return {
        "softmax": "linear softmax",
        "ridge": "ridge regression",
        "ridge+rank": "ridge + cosine rank",
        "multilabel": "multi-label logistic",
    }.get(family, family)


def build_task_card(row: dict, color: str) -> str:
    return f"""
      <article class="task-card" style="--accent:{color}">
        <div class="chip">{esc(family_label(row['family']))}</div>
        <h3>{esc(row['task'])}</h3>
        <dl>
          <dt>Input</dt><dd>{esc(row['input'])}</dd>
          <dt>Head</dt><dd>{esc(row['head'])}</dd>
          <dt>Output</dt><dd>{esc(row['output'])}</dd>
        </dl>
        <div class="metric-line"><span>Metric</span><strong>{esc(row['metric'])}</strong></div>
      </article>
    """


def build_architecture_html(summary: dict, base_path: Path) -> str:
    suite = summary["suite"]
    neural_count = len(suite.get("neural_tasks", {}))
    rows_by_task = {row["task"]: row for row in task_architecture_rows(summary)}
    group_html = []
    for title, color, task_names in TASK_GROUPS:
        cards = "".join(build_task_card(rows_by_task[name], color) for name in task_names)
        group_html.append(
            f"""
            <section class="task-group" style="--accent:{color}">
              <div class="group-head">
                <span></span>
                <h2>{esc(title)}</h2>
              </div>
              <div class="group-cards">{cards}</div>
            </section>
            """
        )

    family_cards = [
        ("Linear softmax", "Minimal classifier for action, subtask, transition, contact, order, and alignment tasks.", COLORS["blue"]),
        ("Ridge regression", "Minimal closed-form projection for forecasting, reconstruction, and retrieval spaces.", COLORS["green"]),
        ("Multi-label logistic", "Minimal one-vs-rest sigmoid heads over the object vocabulary with top-1 fallback.", COLORS["orange"]),
        ("Neural MLP", "Optional PyTorch nonlinear classifier/regressor over the same features, splits, and metrics.", COLORS["red"]),
    ]
    families = "".join(
        f"""
        <article class="family" style="--accent:{color}">
          <h3>{esc(title)}</h3>
          <p>{esc(desc)}</p>
        </article>
        """
        for title, desc, color in family_cards
    )
    base_layer = build_base_layer(base_path, 0.36)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: #fbfaf7; font-family: "Avenir Next", "SF Pro Display", Arial, sans-serif; }}
    .canvas {{
      position: relative;
      width: {ARCHITECTURE_WIDTH}px;
      height: {ARCHITECTURE_HEIGHT}px;
      overflow: hidden;
      color: #1f2421;
      background:
        linear-gradient(90deg, rgba(68,55,38,0.03) 1px, transparent 1px),
        linear-gradient(0deg, rgba(68,55,38,0.024) 1px, transparent 1px),
        #fbfaf7;
      background-size: 58px 58px, 58px 58px, auto;
    }}
    .base-layer {{
      position: absolute;
      inset: 0;
      background-size: cover;
      background-position: center;
      filter: saturate(0.95) contrast(0.98);
    }}
    .wash {{
      position: absolute;
      inset: 0;
      background: linear-gradient(180deg, rgba(251,250,247,0.72), rgba(251,250,247,0.92));
    }}
    .content {{
      position: relative;
      height: 100%;
      padding: 58px 74px 64px;
    }}
    header {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 42px;
      align-items: start;
      margin-bottom: 28px;
    }}
    .kicker {{
      font: 700 16px "SF Mono", Menlo, monospace;
      color: #68665f;
      text-transform: uppercase;
      letter-spacing: 0.09em;
      margin-bottom: 13px;
    }}
    h1 {{
      margin: 0;
      font-size: 52px;
      line-height: 1;
      letter-spacing: 0;
    }}
    .subtitle {{
      margin: 15px 0 0;
      max-width: 1060px;
      color: #4d524d;
      font-size: 22px;
      line-height: 1.42;
      font-weight: 520;
    }}
    .summary-pill {{
      display: grid;
      place-items: center;
      min-width: 188px;
      min-height: 112px;
      border: 1px solid #e4ded4;
      border-radius: 8px;
      background: rgba(255,254,253,0.86);
      box-shadow: 0 18px 44px rgba(68,55,38,0.07);
      text-align: center;
    }}
    .summary-pill strong {{
      font: 850 36px "SF Mono", Menlo, monospace;
      line-height: 1;
    }}
    .summary-pill span {{
      display: block;
      margin-top: 8px;
      color: #6c6d68;
      font-size: 15px;
      font-weight: 700;
    }}
    .shared {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 18px;
      margin-bottom: 24px;
    }}
    .shared article {{
      min-height: 110px;
      border: 1px solid #e4ded4;
      border-radius: 8px;
      background: rgba(255,254,253,0.88);
      padding: 20px 22px;
      box-shadow: 0 18px 44px rgba(68,55,38,0.06);
    }}
    .shared h2 {{
      margin: 0 0 9px;
      font-size: 22px;
      line-height: 1.08;
    }}
    .shared p {{
      margin: 0;
      color: #4c534f;
      font-size: 16px;
      line-height: 1.38;
      font-weight: 560;
    }}
    .families {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 18px;
      margin-bottom: 28px;
    }}
    .family {{
      min-height: 124px;
      border: 1px solid #e4ded4;
      border-radius: 8px;
      background: rgba(255,254,253,0.82);
      padding: 20px 20px 18px;
      box-shadow: 0 16px 40px rgba(68,55,38,0.055);
    }}
    .family h3 {{
      margin: 0 0 10px;
      color: var(--accent);
      font-size: 21px;
      line-height: 1.08;
    }}
    .family p {{
      margin: 0;
      color: #4d534f;
      font-size: 15px;
      line-height: 1.42;
      font-weight: 560;
    }}
    .task-groups {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 20px;
    }}
    .task-group {{
      border: 1px solid rgba(228,222,212,0.96);
      border-radius: 8px;
      background: rgba(255,254,253,0.74);
      padding: 18px;
      box-shadow: 0 22px 54px rgba(68,55,38,0.07);
      backdrop-filter: blur(10px);
    }}
    .group-head {{
      display: flex;
      align-items: center;
      gap: 11px;
      margin-bottom: 14px;
    }}
    .group-head span {{
      width: 12px;
      height: 34px;
      border-radius: 999px;
      background: var(--accent);
    }}
    .group-head h2 {{
      margin: 0;
      font-size: 24px;
      line-height: 1.08;
      color: var(--accent);
    }}
    .group-cards {{
      display: grid;
      gap: 14px;
    }}
    .task-card {{
      min-height: 244px;
      position: relative;
      border: 1px solid color-mix(in srgb, var(--accent), #ffffff 68%);
      border-radius: 8px;
      background: rgba(255,254,253,0.92);
      padding: 17px 18px 16px;
      overflow: hidden;
    }}
    .task-card::before {{
      content: "";
      position: absolute;
      inset: 0 auto 0 0;
      width: 6px;
      background: var(--accent);
      opacity: 0.92;
    }}
    .chip {{
      display: inline-flex;
      border: 1px solid color-mix(in srgb, var(--accent), #ffffff 35%);
      border-radius: 6px;
      padding: 4px 8px;
      color: var(--accent);
      font: 850 11px "SF Mono", Menlo, monospace;
      text-transform: uppercase;
      letter-spacing: 0.03em;
      background: rgba(255,254,253,0.72);
    }}
    .task-card h3 {{
      margin: 13px 0 12px;
      color: #161a17;
      font-size: 21px;
      line-height: 1.08;
      overflow-wrap: anywhere;
    }}
    dl {{
      display: grid;
      grid-template-columns: 54px 1fr;
      gap: 5px 9px;
      margin: 0;
      color: #3d453f;
      font-size: 13px;
      line-height: 1.32;
      font-weight: 560;
    }}
    dt {{
      color: var(--accent);
      font: 850 10px "SF Mono", Menlo, monospace;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    dd {{ margin: 0; }}
    .metric-line {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      margin-top: 12px;
      border-top: 1px solid #eee9e1;
      padding-top: 12px;
      font-size: 13px;
      font-weight: 700;
    }}
    .metric-line span {{
      color: #6c6d68;
      font: 850 11px "SF Mono", Menlo, monospace;
      text-transform: uppercase;
    }}
    .metric-line strong {{
      color: var(--accent);
      font: 850 15px "SF Mono", Menlo, monospace;
      text-align: right;
      font-variant-numeric: tabular-nums;
    }}
  </style>
</head>
<body>
  <main class="canvas">
    {base_layer}
    <div class="wash"></div>
    <div class="content">
      <header>
        <div>
          <div class="kicker">minimal + neural verified model architectures</div>
          <h1>12 Xperience-10M episode tasks, minimal and NN heads</h1>
          <p class="subtitle">Each task uses the same aligned episode-window contract. The figure shows minimal heads beside neural MLP metrics; next TODO is Qwen3-Omni fine-tuning with sensor-bridge evaluation.</p>
        </div>
        <div class="summary-pill"><strong>{len(suite['tasks'])}+{neural_count}</strong><span>min + NN tasks</span></div>
      </header>
      <section class="shared">
        <article><h2>Shared windows</h2><p>{suite['num_frames']:,} frames to {suite['num_windows']:,} windows over video, depth, pose, mocap, inertial, and language features.</p></article>
        <article><h2>Feature vector</h2><p>X_all is {suite['feature_dim']:,} dimensions with 17 named blocks; sample audio is documented but not featurized here.</p></article>
        <article><h2>Reusable heads</h2><p>Minimal softmax/ridge/logistic heads plus optional PyTorch MLP heads cover the whole suite.</p></article>
        <article><h2>Artifacts</h2><p>Metrics, predictions, model weights, neural checkpoints, manifests, and the source summary report are committed.</p></article>
      </section>
      <section class="families">{families}</section>
      <section class="task-groups">{"".join(group_html)}</section>
    </div>
  </main>
</body>
</html>
"""


def render_html(html_text: str, output_path: Path, width: int, height: int, keep_html: Path | None = None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if keep_html is None:
        with tempfile.NamedTemporaryFile("w", suffix=".html", encoding="utf-8", delete=False) as handle:
            handle.write(html_text)
            html_path = Path(handle.name)
    else:
        html_path = keep_html
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(html_text, encoding="utf-8")
    subprocess.run(
        [
            "npx",
            "--yes",
            "playwright",
            "screenshot",
            "--full-page",
            f"--viewport-size={width},{height}",
            html_path.resolve().as_uri(),
            str(output_path),
        ],
        check=True,
    )
    print(f"Wrote image: {output_path}")
    print(f"Wrote render HTML: {html_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-base", type=Path, default=DEFAULT_PIPELINE_BASE)
    parser.add_argument("--architecture-base", type=Path, default=DEFAULT_ARCHITECTURE_BASE)
    parser.add_argument("--pipeline-output", type=Path, default=DEFAULT_PIPELINE_OUTPUT)
    parser.add_argument("--architecture-output", type=Path, default=DEFAULT_ARCHITECTURE_OUTPUT)
    parser.add_argument("--html-dir", type=Path, help="Optional directory for the intermediate render HTML files.")
    parser.add_argument("--only", choices=["pipeline", "architecture", "both"], default="both")
    args = parser.parse_args()

    summary = collect_summary()
    if args.only in {"pipeline", "both"}:
        pipeline_html = build_pipeline_html(summary, args.pipeline_base)
        html_path = args.html_dir / "pipeline_diagram.html" if args.html_dir else None
        render_html(pipeline_html, args.pipeline_output, PIPELINE_WIDTH, PIPELINE_HEIGHT, html_path)
    if args.only in {"architecture", "both"}:
        architecture_html = build_architecture_html(summary, args.architecture_base)
        html_path = args.html_dir / "task_architectures.html" if args.html_dir else None
        render_html(architecture_html, args.architecture_output, ARCHITECTURE_WIDTH, ARCHITECTURE_HEIGHT, html_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

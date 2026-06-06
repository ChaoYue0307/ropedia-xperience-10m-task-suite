#!/usr/bin/env python3
"""Organize the 12 Xperience-10M tasks into the four Ropedia research tracks.

The script is intentionally deterministic: it reads the committed task metrics,
adds a manually curated taxonomy, and writes machine-readable artifacts used by the
README, website, and Hugging Face pages.
"""

from __future__ import annotations

import csv
import html
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

from task_display import task_display_name


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "episode_task_suite"
OUT_DIR = RESULTS / "research_directions"
DOCS_DATA = ROOT / "docs" / "data"
CHARTS = ROOT / "docs" / "assets" / "charts"

SUMMARY_REPORT = RESULTS / "summary_report.json"


DIRECTIONS: OrderedDict[str, dict[str, Any]] = OrderedDict(
    [
        (
            "A",
            {
                "id": "human_motion",
                "name": "Human Modeling & Motion Understanding",
                "focus": "Human/hand/body motion, deformation priors, human-object interaction, affordance modeling.",
                "preferred_background": "Human pose/shape estimation, SMPL-style models, motion capture, or motion generation.",
                "current_status": "partially implemented",
                "current_readout": "The sample supports hand trajectory forecasting and contact/object probes, but it does not yet include a full body/shape model or multi-person priors.",
                "next_steps": [
                    "Add SMPL/SMPL-X or MANO-style body/hand parameter targets where available.",
                    "Train sequence models over multi-episode motion trajectories instead of isolated windows.",
                    "Evaluate affordance prediction on held-out objects and held-out episodes.",
                ],
            },
        ),
        (
            "B",
            {
                "id": "reconstruction_rendering",
                "name": "3D/4D Reconstruction & Neural Rendering",
                "focus": "Multi-view dynamic scene reconstruction, NeRF/Gaussian Splatting, novel-view synthesis.",
                "preferred_background": "3D reconstruction, neural rendering, camera calibration, and bundle adjustment.",
                "current_status": "proxy tasks only",
                "current_readout": "The current suite checks cross-modal alignment and depth/video reconstruction proxies; it does not yet train a renderer or reconstruct geometry.",
                "next_steps": [
                    "Use calibrated multi-view video plus SLAM pose to build per-episode camera trajectories.",
                    "Add depth-supervised point clouds, TSDF, Gaussian Splatting, or NeRF baselines.",
                    "Evaluate novel-view synthesis and temporal consistency across held-out views/time.",
                ],
            },
        ),
        (
            "C",
            {
                "id": "egocentric_interaction",
                "name": "Egocentric Vision & Interaction",
                "focus": "Egocentric action and intention understanding, hand-object interaction, gaze/attention modeling, task structure modeling.",
                "preferred_background": "Video understanding, action recognition, or egocentric vision.",
                "current_status": "strongest implemented track",
                "current_readout": "Most of the 12 tasks directly target egocentric action, task state, interaction, grounding, and alignment.",
                "next_steps": [
                    "Move from single-episode chronological splits to held-out-episode splits.",
                    "Use audio together with stronger multimodal backbones for action, intent, and grounding.",
                    "Evaluate long-horizon task success prediction and action-conditioned generation.",
                ],
            },
        ),
        (
            "D",
            {
                "id": "world_modeling",
                "name": "Scene Reconstruction & World Modeling",
                "focus": "Long-term consistent 3D/4D scene mapping, scene graphs, object- and space-centric representations, spatial reasoning.",
                "preferred_background": "Large-scale mapping, semantic reconstruction, or agent world models.",
                "current_status": "early proxy tasks",
                "current_readout": "The current tasks probe temporal structure, object relevance, cross-modal retrieval, and modality prediction, but they do not yet build persistent maps or scene graphs.",
                "next_steps": [
                    "Convert windows into persistent object/scene-state nodes with timestamps and camera poses.",
                    "Add map consistency, object permanence, and spatial relation prediction tasks.",
                    "Train held-out-episode world models that predict future observations and task state.",
                ],
            },
        ),
    ]
)


TASK_TAXONOMY: OrderedDict[str, dict[str, Any]] = OrderedDict(
    [
        (
            "timeline_action",
            {
                "name": "Timeline action recognition",
                "family": "supervised",
                "input": "all featurized modalities",
                "output": "current action label",
                "primary_direction": "C",
                "direction_roles": {"C": "direct", "A": "proxy"},
                "why": "Reads egocentric sensor state as the current human action; also provides a weak human-motion readout.",
                "current_limit": "Chronological single-episode split creates unseen future action classes.",
            },
        ),
        (
            "timeline_subtask",
            {
                "name": "Timeline subtask recognition",
                "family": "supervised",
                "input": "all featurized modalities",
                "output": "current subtask label",
                "primary_direction": "C",
                "direction_roles": {"C": "direct", "D": "proxy"},
                "why": "Segments egocentric task state and provides a first proxy for symbolic world/task state.",
                "current_limit": "Single-episode ordering makes future subtasks hard to generalize.",
            },
        ),
        (
            "transition_detection",
            {
                "name": "Action transition detection",
                "family": "diagnostic",
                "input": "all featurized modalities",
                "output": "boundary vs steady state",
                "primary_direction": "C",
                "direction_roles": {"C": "direct", "D": "diagnostic"},
                "why": "Localizes egocentric task boundaries and diagnoses temporal state changes.",
                "current_limit": "Boundary class is sparse, so accuracy alone is misleading.",
            },
        ),
        (
            "next_action",
            {
                "name": "Short-horizon next action",
                "family": "supervised",
                "input": "current multimodal window",
                "output": "action 20 frames later",
                "primary_direction": "C",
                "direction_roles": {"C": "direct", "D": "proxy"},
                "why": "Tests action intention/task-flow prediction from egocentric context.",
                "current_limit": "Unseen future labels dominate the single-episode chronological test.",
            },
        ),
        (
            "hand_trajectory_forecast",
            {
                "name": "Hand trajectory forecasting",
                "family": "forecast",
                "input": "current multimodal window",
                "output": "future left/right hand 3D joints",
                "primary_direction": "A",
                "direction_roles": {"A": "direct", "C": "proxy"},
                "why": "Directly predicts human hand motion and supports hand-object interaction modeling.",
                "current_limit": "Forecasting is window-level and not yet a full sequence or policy model.",
            },
        ),
        (
            "contact_prediction",
            {
                "name": "Body/object contact prediction",
                "family": "supervised",
                "input": "non-contact/non-caption features",
                "output": "binary contact label",
                "primary_direction": "A",
                "direction_roles": {"A": "direct", "C": "proxy"},
                "why": "Targets physical interaction state, a core affordance and manipulation signal.",
                "current_limit": "The public sample is degenerate for this target because one class dominates.",
            },
        ),
        (
            "object_relevance",
            {
                "name": "Relevant object set prediction",
                "family": "supervised",
                "input": "non-caption feature blocks",
                "output": "multi-label object set",
                "primary_direction": "C",
                "direction_roles": {"C": "direct", "A": "proxy", "D": "proxy"},
                "why": "Connects egocentric activity to manipulated objects and early object-centric state.",
                "current_limit": "Object labels are language-derived and sparse in one episode.",
            },
        ),
        (
            "caption_grounding",
            {
                "name": "Caption-to-window grounding",
                "family": "retrieval",
                "input": "caption objects/interaction query and candidate sensor windows",
                "output": "matching time window",
                "primary_direction": "C",
                "direction_roles": {"C": "direct", "D": "proxy"},
                "why": "Grounds language annotation into egocentric sensor time and task state.",
                "current_limit": "Bag-of-objects language features are too weak for rich grounding.",
            },
        ),
        (
            "cross_modal_retrieval",
            {
                "name": "Cross-modal retrieval",
                "family": "retrieval",
                "input": "motion/IMU/camera query",
                "output": "matching depth/video window",
                "primary_direction": "C",
                "direction_roles": {"C": "diagnostic", "B": "proxy", "D": "proxy"},
                "why": "Tests whether synchronized modalities identify the same 4D moment, a prerequisite for reconstruction and world modeling.",
                "current_limit": "Retrieval shows an alignment signal, not geometric reconstruction.",
            },
        ),
        (
            "modality_reconstruction",
            {
                "name": "Modality reconstruction",
                "family": "forecast",
                "input": "motion/IMU/camera",
                "output": "depth/video feature vector",
                "primary_direction": "B",
                "direction_roles": {"B": "proxy", "D": "proxy"},
                "why": "Predicts visual/depth state from non-target sensors as a weak reconstruction/world-model objective.",
                "current_limit": "Feature-vector reconstruction is not pixel, depth-map, mesh, NeRF, or Gaussian reconstruction.",
            },
        ),
        (
            "temporal_order",
            {
                "name": "Temporal order verification",
                "family": "diagnostic",
                "input": "two adjacent windows",
                "output": "correct vs reversed order",
                "primary_direction": "C",
                "direction_roles": {"C": "diagnostic", "D": "diagnostic"},
                "why": "Checks whether features encode local time direction and task progression.",
                "current_limit": "Only local adjacent ordering, not long-horizon causal modeling.",
            },
        ),
        (
            "misalignment_detection",
            {
                "name": "Cross-modal misalignment detection",
                "family": "diagnostic",
                "input": "motion plus visual/depth pair",
                "output": "aligned vs shifted",
                "primary_direction": "C",
                "direction_roles": {"C": "diagnostic", "B": "diagnostic", "D": "diagnostic"},
                "why": "Detects temporal desynchronization, a key data-quality gate for multimodal reconstruction and world models.",
                "current_limit": "Synthetic shifts diagnose alignment but do not solve calibration or mapping.",
            },
        ),
    ]
)


METRIC_SPECS = {
    "timeline_action": ("macro_f1", "macro-F1", "higher"),
    "timeline_subtask": ("macro_f1", "macro-F1", "higher"),
    "transition_detection": ("macro_f1", "macro-F1", "higher"),
    "next_action": ("macro_f1", "macro-F1", "higher"),
    "hand_trajectory_forecast": ("mpjpe", "MPJPE", "lower"),
    "contact_prediction": ("macro_f1", "macro-F1", "higher"),
    "object_relevance": ("micro_f1", "micro-F1", "higher"),
    "caption_grounding": ("mrr", "MRR", "higher"),
    "cross_modal_retrieval": ("mrr", "MRR", "higher"),
    "modality_reconstruction": ("r2", "R2", "higher"),
    "temporal_order": ("f1", "F1", "higher"),
    "misalignment_detection": ("f1", "F1", "higher"),
}


def load_summary() -> dict[str, Any]:
    return json.loads(SUMMARY_REPORT.read_text(encoding="utf-8"))


def metric_value(metrics: dict[str, Any] | None, task: str) -> float | None:
    if not metrics:
        return None
    key = METRIC_SPECS[task][0]
    value = metrics.get(key)
    return float(value) if value is not None else None


def choose_better(task: str, minimal: float | None, neural: float | None) -> str:
    if minimal is None or neural is None:
        return "unavailable"
    _, _, direction = METRIC_SPECS[task]
    delta = neural - minimal
    if abs(delta) < 1e-9:
        return "tie"
    if direction == "lower":
        return "neural_mlp" if delta < 0 else "minimal"
    return "neural_mlp" if delta > 0 else "minimal"


def fmt_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    if abs(value) >= 10:
        return f"{value:.3f}"
    return f"{value:.4f}"


def baseline_readout(label: str) -> str:
    if label == "tie":
        return "Both baselines are tied"
    if label == "minimal":
        return "Minimal baseline is stronger"
    if label == "neural_mlp":
        return "Neural MLP is stronger"
    return "Baseline comparison is unavailable"


def build_taxonomy(summary: dict[str, Any]) -> dict[str, Any]:
    minimal_tasks = summary["tasks"]
    neural_tasks = summary.get("neural_tasks", {})

    task_records: OrderedDict[str, dict[str, Any]] = OrderedDict()
    direction_counts = {
        code: {"direct": 0, "proxy": 0, "diagnostic": 0, "total_links": 0}
        for code in DIRECTIONS
    }

    for task, spec in TASK_TAXONOMY.items():
        metric_key, metric_name, metric_direction = METRIC_SPECS[task]
        minimal_metric = metric_value(minimal_tasks.get(task), task)
        neural_metric = metric_value(neural_tasks.get(task), task)
        better = choose_better(task, minimal_metric, neural_metric)

        roles = spec["direction_roles"]
        for direction_code, role in roles.items():
            direction_counts[direction_code][role] += 1
            direction_counts[direction_code]["total_links"] += 1

        task_records[task] = {
            **spec,
            "display_name": task_display_name(task),
            "artifact_id": task,
            "metric": {
                "key": metric_key,
                "name": metric_name,
                "direction": metric_direction,
                "minimal": minimal_metric,
                "neural_mlp": neural_metric,
                "better_baseline": better,
            },
        }

    direction_records = OrderedDict()
    for code, info in DIRECTIONS.items():
        linked_tasks = [
            task
            for task, spec in task_records.items()
            if code in spec["direction_roles"]
        ]
        direction_records[code] = {
            **info,
            "tasks": linked_tasks,
            "task_display_names": [task_display_name(task) for task in linked_tasks],
            "counts": direction_counts[code],
        }

    return {
        "source": "results/episode_task_suite/summary_report.json",
        "dataset_scope": {
            "sample_episode_count": 1,
            "num_frames": summary.get("num_frames"),
            "num_windows": summary.get("num_windows"),
            "feature_dim": summary.get("feature_dim"),
            "warning": "Single public sample episode; this supports pipeline/task evidence, while cross-episode generalization requires held-out episodes.",
        },
        "baselines": {
            "minimal": f"Interpretable softmax, logistic, ridge, and retrieval heads over the {summary.get('feature_dim'):,}-d window feature vector.",
            "neural_mlp": "Small PyTorch MLP classifiers/regressors using the same features, splits, and task contracts.",
        },
        "directions": direction_records,
        "tasks": task_records,
    }


def write_csv(taxonomy: dict[str, Any]) -> None:
    path = OUT_DIR / "research_direction_task_map.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "direction",
                "direction_name",
                "task",
                "task_display_name",
                "task_name",
                "family",
                "relationship",
                "primary_direction",
                "metric_name",
                "minimal_metric",
                "neural_mlp_metric",
                "better_baseline",
                "why",
                "current_limit",
            ]
        )
        for task, spec in taxonomy["tasks"].items():
            metric = spec["metric"]
            for direction_code, relationship in spec["direction_roles"].items():
                writer.writerow(
                    [
                        direction_code,
                        taxonomy["directions"][direction_code]["name"],
                        task,
                        spec["display_name"],
                        spec["name"],
                        spec["family"],
                        relationship,
                        spec["primary_direction"],
                        metric["name"],
                        "" if metric["minimal"] is None else f"{metric['minimal']:.12g}",
                        "" if metric["neural_mlp"] is None else f"{metric['neural_mlp']:.12g}",
                        metric["better_baseline"],
                        spec["why"],
                        spec["current_limit"],
                    ]
                )


def write_markdown(taxonomy: dict[str, Any]) -> None:
    lines = [
        "# Four-Direction Task Taxonomy",
        "",
        "This file is generated by `scripts/research_direction_taxonomy.py` from the committed 12-task metrics.",
        "It maps the current Xperience-10M sample tasks to the four Ropedia research directions and marks which parts require multi-episode evidence.",
        "",
        "## Baseline Families",
        "",
        "| Baseline | Meaning |",
        "| --- | --- |",
        f"| Minimal | {taxonomy['baselines']['minimal']} |",
        f"| Neural MLP | {taxonomy['baselines']['neural_mlp']} |",
        "",
        "## Direction Coverage",
        "",
        "| Direction | Current status | Direct | Proxy | Diagnostic | Current readout |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for code, info in taxonomy["directions"].items():
        counts = info["counts"]
        lines.append(
            f"| {code}. {info['name']} | {info['current_status']} | {counts['direct']} | {counts['proxy']} | {counts['diagnostic']} | {info['current_readout']} |"
        )

    lines.extend(
        [
            "",
            "## Task Mapping With Two Baselines",
            "",
            "| Task | Artifact id | Primary direction | Related directions | Minimal | Neural MLP | Readout |",
            "| --- | --- | --- | --- | ---: | ---: | --- |",
        ]
    )
    for task, spec in taxonomy["tasks"].items():
        metric = spec["metric"]
        related = ", ".join(
            f"{code}:{role}" for code, role in spec["direction_roles"].items()
        )
        minimal = f"{fmt_metric(metric['minimal'])} {metric['name']}"
        neural = f"{fmt_metric(metric['neural_mlp'])} {metric['name']}"
        readout = f"{baseline_readout(metric['better_baseline'])}. {spec['current_limit']}"
        lines.append(
            f"| {spec['display_name']} | `{task}` | {spec['primary_direction']} | {related} | {minimal} | {neural} | {readout} |"
        )

    lines.extend(["", "## Next-Step Interpretation", ""])
    for code, info in taxonomy["directions"].items():
        lines.append(f"### {code}. {info['name']}")
        lines.append("")
        lines.append(info["current_readout"])
        lines.append("")
        for step in info["next_steps"]:
            lines.append(f"- {step}")
        lines.append("")

    (OUT_DIR / "research_direction_summary.md").write_text(
        "\n".join(lines).rstrip() + "\n", encoding="utf-8"
    )


def svg_text(x: int, y: int, text: str, size: int = 16, weight: int = 500, color: str = "#f4f8ef") -> str:
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
        f'fill="{color}">{html.escape(text)}</text>'
    )


def write_svg(taxonomy: dict[str, Any]) -> None:
    width = 1180
    height = 700
    margin = 58
    card_w = 515
    card_h = 220
    colors = {"direct": "#ccffa0", "proxy": "#7ae5c3", "diagnostic": "#d8f4a5"}
    cards = []

    for idx, (code, info) in enumerate(taxonomy["directions"].items()):
        row = idx // 2
        col = idx % 2
        x = margin + col * (card_w + 34)
        y = 130 + row * (card_h + 34)
        counts = info["counts"]
        total = max(1, counts["direct"] + counts["proxy"] + counts["diagnostic"])
        bar_x = x + 24
        bar_y = y + 132
        bar_w = card_w - 48
        cursor = bar_x
        segments = []
        for key in ("direct", "proxy", "diagnostic"):
            seg_w = round(bar_w * counts[key] / total)
            if counts[key] > 0:
                segments.append(
                    f'<rect x="{cursor}" y="{bar_y}" width="{seg_w}" height="16" rx="8" fill="{colors[key]}"/>'
                )
            cursor += seg_w

        task_labels = ", ".join(info["task_display_names"][:5])
        if len(info["task_display_names"]) > 5:
            task_labels += f", +{len(info['task_display_names']) - 5}"

        cards.append(
            "\n".join(
                [
                    f'<rect x="{x}" y="{y}" width="{card_w}" height="{card_h}" rx="8" fill="#050905" stroke="#ccffa0" stroke-opacity="0.24"/>',
                    svg_text(x + 24, y + 42, f"{code}. {info['name']}", 21, 700),
                    svg_text(x + 24, y + 75, info["current_status"], 15, 700, "#ccffa0"),
                    svg_text(x + 24, y + 108, f"Tasks: {task_labels}", 14, 500, "#dce8d7"),
                    *segments,
                    svg_text(x + 24, y + 174, f"Direct {counts['direct']}", 14, 700, colors["direct"]),
                    svg_text(x + 150, y + 174, f"Proxy {counts['proxy']}", 14, 700, colors["proxy"]),
                    svg_text(x + 270, y + 174, f"Diagnostic {counts['diagnostic']}", 14, 700, colors["diagnostic"]),
                ]
            )
        )

    legend = []
    lx = margin
    for key, label in (
        ("direct", "Direct task"),
        ("proxy", "Proxy / prerequisite"),
        ("diagnostic", "Diagnostic probe"),
    ):
        legend.extend(
            [
                f'<rect x="{lx}" y="622" width="16" height="16" rx="4" fill="{colors[key]}"/>',
                svg_text(lx + 24, 636, label, 14, 600, "#dce8d7"),
            ]
        )
        lx += 200

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Xperience-10M task coverage across four research directions">
  <rect width="100%" height="100%" fill="#020502"/>
  <rect x="24" y="24" width="1132" height="652" rx="20" fill="#050905" stroke="#ccffa0" stroke-opacity="0.24"/>
  {svg_text(margin, 64, "Xperience-10M 12-Task Suite: Four Research Directions", 30, 800)}
  {svg_text(margin, 96, "One public sample episode, two baseline families, explicit direct/proxy/diagnostic coverage.", 16, 500, "#a5afa2")}
  {"".join(cards)}
  {"".join(legend)}
  {svg_text(margin, 670, "Generated from results/episode_task_suite/summary_report.json and scripts/research_direction_taxonomy.py", 13, 500, "#a5afa2")}
</svg>
"""
    (CHARTS / "research_direction_coverage.svg").write_text(svg, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DATA.mkdir(parents=True, exist_ok=True)
    CHARTS.mkdir(parents=True, exist_ok=True)

    taxonomy = build_taxonomy(load_summary())
    json_text = json.dumps(taxonomy, indent=2, ensure_ascii=False)
    (OUT_DIR / "research_direction_taxonomy.json").write_text(json_text + "\n", encoding="utf-8")
    (DOCS_DATA / "research_directions.json").write_text(json_text + "\n", encoding="utf-8")
    write_csv(taxonomy)
    write_markdown(taxonomy)
    write_svg(taxonomy)

    print(f"Wrote {OUT_DIR / 'research_direction_taxonomy.json'}")
    print(f"Wrote {OUT_DIR / 'research_direction_task_map.csv'}")
    print(f"Wrote {OUT_DIR / 'research_direction_summary.md'}")
    print(f"Wrote {DOCS_DATA / 'research_directions.json'}")
    print(f"Wrote {CHARTS / 'research_direction_coverage.svg'}")


if __name__ == "__main__":
    main()

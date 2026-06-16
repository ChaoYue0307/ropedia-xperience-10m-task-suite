#!/usr/bin/env python3
"""Build a unified 20-task radar chart for baseline and model-branch metrics."""

from __future__ import annotations

import html
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TASK_SUITE_PATH = ROOT / "docs/data/task_suite_20.json"
QWEN_V6_METRICS_PATH = (
    ROOT
    / "results/omni_finetune/verified_public"
    / "xperience10m_qwen3_omni_128ep_multiscale_cap96_v6_rank64_lr5e5_full8gpu_lora_eval_test_full"
    / "eval/metrics.json"
)
COSMOS_SUPER_REASONER_METRICS_PATH = (
    ROOT
    / "results/omni_finetune/verified_public"
    / "xperience10m_cosmos3_super_reasoner_128ep_test_full_20260607"
    / "eval/metrics.json"
)
COSMOS_NANO_METRICS_PATH = (
    ROOT
    / "results/omni_finetune/verified_public"
    / "xperience10m_cosmos3_nano_128ep_future_window_h5_compat_adapter_eval_test_full"
    / "eval/metrics.json"
)
COSMOS_SUPER_FD_METRICS_PATH = (
    ROOT
    / "results/omni_finetune/verified_public"
    / "xperience10m_cosmos3_super_forward_dynamics_lora_128ep_train1epoch_256_attn_full8gpu_20260608_eval_test_full_fsdp"
    / "eval/metrics.json"
)
METADATA128_BASELINE_DIR = ROOT / "results/omni_finetune/a100_128_metadata_task_baselines_20260616_v2"
OUTPUT_JSON = ROOT / "docs/data/unified_task_model_radar.json"
OUTPUT_SVG = ROOT / "docs/assets/charts/unified_task_model_radar.svg"


SERIES = {
    "minimal": {
        "label": "Minimal",
        "short_label": "Min",
        "color": "#ccffa0",
        "kind": "full_20_task_baseline",
        "scope": "1 public sample episode",
        "stroke_dasharray": None,
    },
    "neural_mlp": {
        "label": "Neural MLP",
        "short_label": "NN",
        "color": "#67e8d1",
        "kind": "full_20_task_baseline",
        "scope": "1 public sample episode",
        "stroke_dasharray": None,
    },
    "metadata128_simple": {
        "label": "128ep Metadata Simple",
        "short_label": "128-S",
        "color": "#ffd166",
        "kind": "partial_128_episode_metadata_baseline",
        "scope": "128 selected episodes, JSONL metadata/text only",
        "stroke_dasharray": "9 6",
    },
    "metadata128_neural_mlp": {
        "label": "128ep Metadata NN",
        "short_label": "128-NN",
        "color": "#f472b6",
        "kind": "partial_128_episode_metadata_baseline",
        "scope": "128 selected episodes, JSONL metadata/text only",
        "stroke_dasharray": "3 6",
    },
    "qwen3_omni_v6_lora": {
        "label": "Qwen3-Omni v6 LoRA",
        "short_label": "Qwen3",
        "color": "#9bb8ff",
        "kind": "partial_128_episode_foundation_model_overlay",
        "scope": "128 selected episodes, held-out test",
        "stroke_dasharray": "7 7",
    },
    "cosmos3_super_reasoner": {
        "label": "Cosmos3-Super Reasoner",
        "short_label": "C3-S",
        "color": "#ff9c7a",
        "kind": "partial_128_episode_foundation_model_overlay",
        "scope": "128 selected episodes, held-out test",
        "stroke_dasharray": "4 7",
    },
    "cosmos3_nano_future_window": {
        "label": "Cosmos3-Nano Future Window",
        "short_label": "C3-N",
        "color": "#d9c7ff",
        "kind": "partial_128_episode_world_model_overlay",
        "scope": "128 selected episodes, held-out test",
        "stroke_dasharray": "2 7",
    },
}

FOUNDATION_TASK_METRICS = {
    "timeline_action": {
        "qwen3_omni_v6_lora": "action_macro_f1",
        "cosmos3_super_reasoner": "action_macro_f1",
        "cosmos3_nano_future_window": "action_accuracy_from_retrieved_future",
    },
    "timeline_subtask": {
        "qwen3_omni_v6_lora": "subtask_accuracy",
        "cosmos3_super_reasoner": "subtask_accuracy",
    },
    "transition_detection": {
        "qwen3_omni_v6_lora": "transition_accuracy",
        "cosmos3_super_reasoner": "transition_accuracy",
        "cosmos3_nano_future_window": "transition_accuracy",
    },
    "next_action": {
        "qwen3_omni_v6_lora": "next_action_accuracy",
        "cosmos3_super_reasoner": "next_action_accuracy",
        "cosmos3_nano_future_window": "action_accuracy_from_retrieved_future",
    },
    "contact_prediction": {
        "qwen3_omni_v6_lora": "contact_accuracy",
        "cosmos3_super_reasoner": "contact_accuracy",
        "cosmos3_nano_future_window": "contact_accuracy",
    },
    "object_relevance": {
        "qwen3_omni_v6_lora": "object_micro_f1",
        "cosmos3_super_reasoner": "object_micro_f1",
    },
    "cross_modal_retrieval": {
        "cosmos3_nano_future_window": "future_retrieval_mrr",
    },
}

METADATA128_TASKS = {
    "timeline_action",
    "timeline_subtask",
    "transition_detection",
    "next_action",
    "contact_prediction",
    "object_relevance",
    "caption_grounding",
    "temporal_order",
}

SHORT_TASK_LABELS = {
    "timeline_action": "Action",
    "timeline_subtask": "Step",
    "transition_detection": "Boundary",
    "next_action": "Next act",
    "hand_trajectory_forecast": "Hand traj",
    "contact_prediction": "Contact",
    "object_relevance": "Objects",
    "caption_grounding": "Language",
    "cross_modal_retrieval": "X-modal",
    "modality_reconstruction": "Recon",
    "temporal_order": "Order",
    "misalignment_detection": "Sync",
    "long_horizon_next_action": "Long act",
    "next_subtask_forecast": "Long step",
    "interaction_text_prediction": "Interact txt",
    "action_object_relation": "Act+obj",
    "object_set_forecast": "Future obj",
    "imu_to_hand_pose": "IMU->hand",
    "camera_view_sync_retrieval": "Cam sync",
    "time_to_transition": "Time2bdry",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_a100_metadata_metric(task_id: str, *, neural: bool = False) -> dict[str, Any] | None:
    if task_id not in METADATA128_TASKS:
        return None
    path = METADATA128_BASELINE_DIR / ("neural_mlp" if neural else "") / task_id / "metrics.json"
    if not path.exists():
        return None
    payload = read_json(path)
    if payload.get("status") != "pass":
        return None
    score = payload.get("primary_score")
    if score is None:
        return None
    return {
        "raw": score,
        "metric_key": payload.get("primary_metric"),
        "source": str(path.relative_to(ROOT)),
        "scope": "multi_episode_128_metadata_baseline",
    }


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def score_from_raw(value: float | None, direction: str, best_lower: float | None = None) -> float | None:
    if value is None:
        return None
    if direction == "lower":
        if value <= 0:
            return 1.0
        if best_lower is None or best_lower <= 0:
            return None
        return clamp01(best_lower / value)
    return clamp01(value)


def format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    if abs(value) >= 10:
        return f"{value:.2f}"
    if abs(value) >= 1:
        return f"{value:.3f}"
    return f"{value:.4f}"


def point(cx: float, cy: float, radius: float, angle: float) -> tuple[float, float]:
    return cx + math.cos(angle) * radius, cy + math.sin(angle) * radius


def svg_text(
    x: float,
    y: float,
    text: str,
    *,
    size: int = 16,
    fill: str = "#f4f8ef",
    anchor: str = "start",
    weight: int | str = 600,
    opacity: float = 1.0,
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="Space Grotesk, Arial, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" opacity="{opacity:.3f}">{html.escape(text)}</text>'
    )


def polyline(points: list[tuple[float, float]], *, fill: str, stroke: str, opacity: float, stroke_width: float, dash: str | None = None) -> str:
    coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<polygon points="{coords}" fill="{fill}" fill-opacity="{opacity:.3f}" '
        f'stroke="{stroke}" stroke-opacity="0.92" stroke-width="{stroke_width}"{dash_attr}/>'
    )


def build_payload() -> dict[str, Any]:
    suite = read_json(TASK_SUITE_PATH)
    qwen = read_json(QWEN_V6_METRICS_PATH)
    cosmos_super = read_json(COSMOS_SUPER_REASONER_METRICS_PATH)
    cosmos_nano = read_json(COSMOS_NANO_METRICS_PATH)
    cosmos_fd = read_json(COSMOS_SUPER_FD_METRICS_PATH)
    foundation_metrics = {
        "qwen3_omni_v6_lora": qwen,
        "cosmos3_super_reasoner": cosmos_super,
        "cosmos3_nano_future_window": cosmos_nano,
    }

    tasks: list[dict[str, Any]] = []
    for row in suite.get("tasks", []):
        values: dict[str, dict[str, Any]] = {
            "minimal": {
                "raw": row.get("minimal_primary_metric"),
                "metric_key": row.get("metric_key"),
                "source": row.get("artifact_sources", {}).get("minimal_metrics"),
                "scope": "single_episode_public_sample",
            },
            "neural_mlp": {
                "raw": row.get("neural_primary_metric"),
                "metric_key": row.get("metric_key"),
                "source": row.get("artifact_sources", {}).get("neural_metrics"),
                "scope": "single_episode_public_sample",
            },
        }
        for series_id, metric_key in FOUNDATION_TASK_METRICS.get(row["task_id"], {}).items():
            raw = foundation_metrics.get(series_id, {}).get(metric_key)
            values[series_id] = {
                "raw": raw,
                "metric_key": metric_key,
                "source": str(
                    {
                        "qwen3_omni_v6_lora": QWEN_V6_METRICS_PATH,
                        "cosmos3_super_reasoner": COSMOS_SUPER_REASONER_METRICS_PATH,
                        "cosmos3_nano_future_window": COSMOS_NANO_METRICS_PATH,
                    }[series_id].relative_to(ROOT)
                ),
                "scope": "multi_episode_128_partial_model_overlay",
            }
        metadata_simple = read_a100_metadata_metric(row["task_id"], neural=False)
        if metadata_simple:
            values["metadata128_simple"] = metadata_simple
        metadata_neural = read_a100_metadata_metric(row["task_id"], neural=True)
        if metadata_neural:
            values["metadata128_neural_mlp"] = metadata_neural

        lower_values = [
            item["raw"]
            for item in values.values()
            if row.get("metric_direction") == "lower" and isinstance(item.get("raw"), (int, float)) and item["raw"] > 0
        ]
        best_lower = min(lower_values) if lower_values else None
        for item in values.values():
            item["normalized_score"] = score_from_raw(item.get("raw"), row.get("metric_direction", "higher"), best_lower)
            item["raw_text"] = format_metric(item.get("raw"))

        tasks.append(
            {
                "task_number": row["task_number"],
                "task_id": row["task_id"],
                "label": row.get("task_display_name", row["task_id"]),
                "short_label": SHORT_TASK_LABELS.get(row["task_id"], row["task_id"].replace("_", " ").title()),
                "origin": row.get("origin"),
                "metric_key": row.get("metric_key"),
                "metric_name": row.get("metric_name"),
                "metric_direction": row.get("metric_direction"),
                "values": values,
            }
        )

    series_records = []
    for series_id, spec in SERIES.items():
        covered = sum(1 for task in tasks if task["values"].get(series_id, {}).get("normalized_score") is not None)
        series_records.append(
            {
                "id": series_id,
                **spec,
                "covered_task_count": covered,
                "coverage_fraction": covered / max(len(tasks), 1),
            }
        )

    fd_loss = (cosmos_fd.get("loss_summary") or {}).get("mean")
    return {
        "title": "Unified 20-Task Model Radar",
        "status": "pass",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task_count": len(tasks),
        "normalization_policy": {
            "higher_is_better": "bounded metrics are plotted directly on 0-1 axes after clipping to [0, 1]",
            "lower_is_better": "lower-error metrics are converted to best_observed_value / raw_value within the same task",
            "raw_values": "raw metric values, metric keys, and sources are retained in this JSON; the SVG is an overview, not a replacement for the metric table",
            "foundation_model_overlay": "Qwen3/Cosmos points are plotted only on task-aligned axes. Missing axes mean the public result does not evaluate that task contract.",
            "metadata_128_overlay": "128-episode metadata baselines are plotted only where the public JSONL contains enough task labels without raw feature blocks.",
        },
        "series": series_records,
        "tasks": tasks,
        "model_branch_cards": [
            {
                "id": "metadata128_simple",
                "title": "128ep Metadata Simple",
                "status": "a100_rerun_pass",
                "coverage": f"{next(item for item in series_records if item['id'] == 'metadata128_simple')['covered_task_count']}/20 JSONL-supported axes",
                "headline": "34,269 rows; train/val/test 25,629/4,608/4,032",
                "source": str((METADATA128_BASELINE_DIR / "summary_report.json").relative_to(ROOT)),
            },
            {
                "id": "metadata128_neural_mlp",
                "title": "128ep Metadata NN",
                "status": "a100_rerun_pass",
                "coverage": f"{next(item for item in series_records if item['id'] == 'metadata128_neural_mlp')['covered_task_count']}/20 JSONL-supported axes",
                "headline": "compact MLP heads over metadata/text features",
                "source": str((METADATA128_BASELINE_DIR / "summary_report.json").relative_to(ROOT)),
            },
            {
                "id": "qwen3_omni_v6_lora",
                "title": "Qwen3-Omni v6 LoRA",
                "status": "verified",
                "task_aligned_axes": SERIES["qwen3_omni_v6_lora"]["short_label"],
                "coverage": f"{next(item for item in series_records if item['id'] == 'qwen3_omni_v6_lora')['covered_task_count']}/20 task-aligned axes",
                "headline": f"JSON validity {format_metric(qwen.get('json_validity_rate'))}; action macro-F1 {format_metric(qwen.get('action_macro_f1'))}",
                "source": str(QWEN_V6_METRICS_PATH.relative_to(ROOT)),
            },
            {
                "id": "cosmos3_super_reasoner",
                "title": "Cosmos3-Super Reasoner",
                "status": "verified_base_weight_eval",
                "coverage": f"{next(item for item in series_records if item['id'] == 'cosmos3_super_reasoner')['covered_task_count']}/20 task-aligned axes",
                "headline": f"JSON validity {format_metric(cosmos_super.get('json_validity_rate'))}; action macro-F1 {format_metric(cosmos_super.get('action_macro_f1'))}",
                "source": str(COSMOS_SUPER_REASONER_METRICS_PATH.relative_to(ROOT)),
            },
            {
                "id": "cosmos3_nano_future_window",
                "title": "Cosmos3-Nano Future Window",
                "status": "verified_compatibility_eval",
                "coverage": f"{next(item for item in series_records if item['id'] == 'cosmos3_nano_future_window')['covered_task_count']}/20 task-aligned axes",
                "headline": f"future retrieval MRR {format_metric(cosmos_nano.get('future_retrieval_mrr'))}; transition accuracy {format_metric(cosmos_nano.get('transition_accuracy'))}",
                "source": str(COSMOS_NANO_METRICS_PATH.relative_to(ROOT)),
            },
            {
                "id": "cosmos3_super_forward_dynamics_lora",
                "title": "Cosmos3-Super Forward-Dynamics LoRA",
                "status": "verified_finetuned_adapter",
                "coverage": "separate camera-pose proxy target, not plotted on the 20 task axes",
                "headline": f"test MSE {format_metric(fd_loss)} over 448 held-out rows",
                "source": str(COSMOS_SUPER_FD_METRICS_PATH.relative_to(ROOT)),
            },
        ],
    }


def render_svg(payload: dict[str, Any]) -> str:
    width, height = 1720, 1360
    cx, cy, radius = 570, 585, 330
    tasks = payload["tasks"]
    n = len(tasks)
    angles = [-math.pi / 2 + 2 * math.pi * i / n for i in range(n)]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        '<filter id="softGlow"><feGaussianBlur stdDeviation="5" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>',
        '<pattern id="dots" width="22" height="22" patternUnits="userSpaceOnUse"><circle cx="2" cy="2" r="1.15" fill="#ccffa0" opacity="0.16"/></pattern>',
        "</defs>",
        '<rect width="100%" height="100%" fill="#020502"/>',
        '<rect width="100%" height="100%" fill="url(#dots)" opacity="0.45"/>',
        '<rect x="28" y="28" width="1664" height="1304" rx="18" fill="#061006" fill-opacity="0.86" stroke="#ccffa0" stroke-opacity="0.22"/>',
        svg_text(70, 86, "Unified 20-Task Model Radar", size=34, weight=800),
        svg_text(70, 122, "Direction-aware normalized scores across the single-episode task suite, with 128ep metadata and Qwen3/Cosmos overlays.", size=17, fill="#a5afa2", weight=560),
        svg_text(70, 156, "Filled polygons: same 20 public-sample tasks. Points: 128-episode branches only where their public metrics map to that task.", size=15, fill="#a5afa2", weight=560),
    ]

    for level in range(1, 6):
        r = radius * level / 5
        ring = [point(cx, cy, r, angle) for angle in angles]
        parts.append(polyline(ring, fill="none", stroke="#ccffa0", opacity=0, stroke_width=1.1))
        parts[-1] = parts[-1].replace('fill="none" fill-opacity="0.000"', 'fill="none"').replace('stroke-opacity="0.92"', 'stroke-opacity="0.15"')
        parts.append(svg_text(cx + 8, cy - r + 4, f"{level / 5:.1f}", size=11, fill="#a5afa2", weight=600, opacity=0.75))

    for task, angle in zip(tasks, angles):
        x, y = point(cx, cy, radius, angle)
        parts.append(f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{x:.1f}" y2="{y:.1f}" stroke="#ccffa0" stroke-opacity="0.12" stroke-width="1"/>')
        lx, ly = point(cx, cy, radius + 58, angle)
        anchor = "middle"
        if math.cos(angle) > 0.25:
            anchor = "start"
        elif math.cos(angle) < -0.25:
            anchor = "end"
        parts.append(svg_text(lx, ly - 7, f"{task['task_number']:02d}", size=11, fill="#ccffa0", anchor=anchor, weight=800, opacity=0.9))
        parts.append(svg_text(lx, ly + 13, task["short_label"], size=12, fill="#dce8d7", anchor=anchor, weight=650))

    for series_id in ("minimal", "neural_mlp"):
        spec = SERIES[series_id]
        points = []
        for task, angle in zip(tasks, angles):
            score = task["values"].get(series_id, {}).get("normalized_score")
            points.append(point(cx, cy, radius * float(score or 0.0), angle))
        parts.append(polyline(points, fill=spec["color"], stroke=spec["color"], opacity=0.18 if series_id == "minimal" else 0.16, stroke_width=4.2))
        for x, y in points:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.0" fill="{spec["color"]}" stroke="#020502" stroke-width="1.1"/>')

    for series_id in ("metadata128_simple", "metadata128_neural_mlp", "qwen3_omni_v6_lora", "cosmos3_super_reasoner", "cosmos3_nano_future_window"):
        spec = SERIES[series_id]
        for task, angle in zip(tasks, angles):
            score = task["values"].get(series_id, {}).get("normalized_score")
            if score is None:
                continue
            x, y = point(cx, cy, radius * float(score), angle)
            radius_px = 6.5 if series_id.startswith("metadata128") else 8.0
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius_px:.1f}" fill="{spec["color"]}" fill-opacity="0.92" '
                f'stroke="#020502" stroke-width="2.0"/>'
            )

    legend_x, legend_y = 1105, 210
    parts.append(f'<rect x="{legend_x - 34}" y="{legend_y - 44}" width="520" height="900" rx="12" fill="#020502" fill-opacity="0.58" stroke="#ccffa0" stroke-opacity="0.20"/>')
    parts.append(svg_text(legend_x, legend_y, "How to read it", size=24, weight=800))
    parts.append(svg_text(legend_x, legend_y + 30, "Score radius is normalized by metric direction.", size=14, fill="#a5afa2", weight=560))
    parts.append(svg_text(legend_x, legend_y + 52, "Raw values stay in unified_task_model_radar.json.", size=14, fill="#a5afa2", weight=560))

    cursor = legend_y + 100
    for record in payload["series"]:
        color = record["color"]
        parts.append(f'<line x1="{legend_x}" y1="{cursor - 4}" x2="{legend_x + 48}" y2="{cursor - 4}" stroke="{color}" stroke-width="7" stroke-linecap="round"/>')
        if record["kind"].startswith("partial"):
            parts.append(f'<circle cx="{legend_x + 24}" cy="{cursor - 4}" r="7" fill="{color}" stroke="#020502" stroke-width="2"/>')
        parts.append(svg_text(legend_x + 64, cursor, record["label"], size=16, weight=800))
        parts.append(svg_text(legend_x + 64, cursor + 22, f"{record['covered_task_count']}/20 axes · {record['scope']}", size=12, fill="#a5afa2", weight=560))
        cursor += 50

    cursor += 10
    parts.append(svg_text(legend_x, cursor, "Model branch notes", size=20, weight=800))
    cursor += 28
    for card in payload["model_branch_cards"]:
        parts.append(f'<rect x="{legend_x}" y="{cursor - 18}" width="445" height="64" rx="8" fill="#081408" stroke="#ccffa0" stroke-opacity="0.15"/>')
        parts.append(svg_text(legend_x + 16, cursor + 3, card["title"], size=14, weight=800))
        parts.append(svg_text(legend_x + 16, cursor + 24, card["coverage"], size=11, fill="#a5afa2", weight=600))
        parts.append(svg_text(legend_x + 16, cursor + 45, card["headline"], size=11, fill="#dce8d7", weight=600))
        cursor += 74

    table_y = 1230
    parts.append(f'<rect x="70" y="{table_y - 35}" width="1540" height="86" rx="10" fill="#020502" fill-opacity="0.54" stroke="#ccffa0" stroke-opacity="0.16"/>')
    parts.append(svg_text(96, table_y - 8, "Caveat", size=15, fill="#ccffa0", weight=800))
    parts.append(svg_text(170, table_y - 8, "This chart compares normalized metric direction, not identical raw units.", size=14, fill="#dce8d7", weight=650))
    parts.append(svg_text(170, table_y + 18, "128-episode metadata, Qwen3, and Cosmos overlays are plotted only on semantically aligned task axes.", size=14, fill="#a5afa2", weight=560))
    parts.append(svg_text(170, table_y + 44, "Cosmos3-Super forward-dynamics LoRA is kept as a branch card because its camera-pose proxy MSE is not one of the 20 task metrics.", size=14, fill="#a5afa2", weight=560))

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> int:
    payload = build_payload()
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_SVG.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    OUTPUT_SVG.write_text(render_svg(payload), encoding="utf-8")
    print(f"PASS: wrote {OUTPUT_JSON}")
    print(f"PASS: wrote {OUTPUT_SVG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

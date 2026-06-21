#!/usr/bin/env python3
"""Build the unified 20-task public-sample task-suite index."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "docs/data/summary_metrics.json"
WALKTHROUGHS_PATH = ROOT / "docs/data/task_walkthroughs.json"
ADDITIONAL_TASKS_PATH = ROOT / "docs/data/tier2_task_suite.json"
OUTPUT_JSON = ROOT / "docs/data/task_suite_20.json"
OUTPUT_MD = ROOT / "TASK_SUITE_20.md"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_value(metrics: dict[str, Any], metric_key: str | None) -> float | None:
    if not metrics or not metric_key:
        return None
    if "primary_score" in metrics:
        return metrics.get("primary_score")
    return metrics.get(metric_key)


def count_fields(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "num_windows",
        "num_samples",
        "num_queries",
        "num_eval_windows",
        "num_train_windows",
        "num_test_windows",
        "num_train_samples",
        "num_test_samples",
        "num_classes",
        "num_labels",
    ]
    return {key: metrics[key] for key in keys if key in metrics}


def source_for(task_id: str, origin: str, neural: bool = False) -> str:
    if origin == "walkthrough_backed":
        prefix = "results/episode_task_suite/neural_mlp" if neural else "results/episode_task_suite"
        return f"{prefix}/{task_id}/metrics.json"
    prefix = "results/episode_task_suite/tier2_task_suite/neural_mlp" if neural else "results/episode_task_suite/tier2_task_suite"
    return f"{prefix}/{task_id}/metrics.json"


def build_core_tasks(summary: dict[str, Any], walkthroughs: dict[str, Any]) -> list[dict[str, Any]]:
    suite = summary["suite"]
    minimal_tasks = suite.get("tasks", {})
    neural_tasks = suite.get("neural_tasks", {})
    rows: list[dict[str, Any]] = []
    for task_id, walkthrough in walkthroughs["tasks"].items():
        metric = walkthrough.get("metric", {})
        metric_key = metric.get("key")
        minimal = minimal_tasks.get(task_id, {})
        neural = neural_tasks.get(task_id, {})
        rows.append(
            {
                "task_id": task_id,
                "task_display_name": walkthrough.get("display_name") or walkthrough.get("research_name") or task_id,
                "research_name": walkthrough.get("research_name"),
                "provenance_source": "walkthrough_backed_task_contract",
                "origin_count_label": "unified task",
                "family": walkthrough.get("task_family"),
                "architecture_family": walkthrough.get("architecture_family"),
                "primary_direction": walkthrough.get("primary_direction"),
                "input": walkthrough.get("input"),
                "input_short": walkthrough.get("input_short"),
                "process": walkthrough.get("process_short"),
                "output": walkthrough.get("output"),
                "output_short": walkthrough.get("output_short"),
                "metric_key": metric_key,
                "metric_name": metric.get("name"),
                "metric_direction": metric.get("direction"),
                "minimal_primary_metric": metric_value(minimal, metric_key),
                "neural_primary_metric": metric_value(neural, metric_key),
                "counts": count_fields(minimal),
                "meaning": walkthrough.get("card_blurb") or walkthrough.get("plain_goal"),
                "artifact_sources": {
                    "walkthrough": f"results/episode_task_suite/task_walkthroughs/{task_id}.md",
                    "minimal_metrics": source_for(task_id, "walkthrough_backed", neural=False),
                    "neural_metrics": source_for(task_id, "walkthrough_backed", neural=True),
                },
            }
        )
    return rows


def build_additional_tasks(additional: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task_id, spec in additional.get("task_specs", {}).items():
        result = additional.get("tasks", {}).get(task_id, {})
        minimal = result.get("minimal") or {}
        neural = result.get("neural_mlp") or {}
        metric_key = spec.get("metric_key")
        rows.append(
            {
                "task_id": task_id,
                "task_display_name": spec.get("name", task_id.replace("_", " ").title()),
                "research_name": spec.get("name", task_id.replace("_", " ").title()),
                "provenance_source": "historical_result_bundle",
                "origin_count_label": "unified task",
                "family": spec.get("family"),
                "architecture_family": minimal.get("model_family"),
                "primary_direction": spec.get("research_direction", "sample-supported extension"),
                "input": spec.get("input"),
                "input_short": spec.get("input"),
                "process": "shared window features -> task-specific target builder -> minimal/neural head",
                "output": spec.get("target"),
                "output_short": spec.get("target"),
                "metric_key": metric_key,
                "metric_name": spec.get("metric_name"),
                "metric_direction": spec.get("metric_direction"),
                "minimal_primary_metric": metric_value(minimal, metric_key),
                "neural_primary_metric": metric_value(neural, metric_key),
                "counts": count_fields(minimal),
                "meaning": spec.get("meaning"),
                "artifact_sources": {
                    "legacy_result_directory": "results/episode_task_suite/tier2_task_suite/",
                    "minimal_metrics": source_for(task_id, "historical_provenance", neural=False),
                    "neural_metrics": source_for(task_id, "historical_provenance", neural=True),
                },
            }
        )
    return rows


def build_payload() -> dict[str, Any]:
    summary = read_json(SUMMARY_PATH)
    walkthroughs = read_json(WALKTHROUGHS_PATH)
    additional = read_json(ADDITIONAL_TASKS_PATH)
    suite = summary["suite"]
    tasks = build_core_tasks(summary, walkthroughs) + build_additional_tasks(additional)
    for idx, row in enumerate(tasks, start=1):
        row["task_number"] = idx
        row["suite_label"] = f"Task {idx:02d}"

    return {
        "title": "Ropedia Xperience-10M Unified 20-Task Suite",
        "status": "pass",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task_count": len(tasks),
        "task_count_summary": {
            "total_unified_tasks": len(tasks),
            "public_framing": "all 20 task contracts are presented as one suite",
            "legacy_provenance_rows": len(tasks) - 12,
        },
        "unification_policy": {
            "public_framing": "The suite is presented as one 20-task benchmark surface. All task contracts share the same window, split, feature, baseline, and leakage-control language.",
            "legacy_path_note": "The directory and file name tier2_task_suite are retained only for backward-compatible artifact links; they are not a separate public benchmark tier.",
        },
        "dataset_scope": {
            "sample_episode_count": 1,
            "annotation": suite.get("annotation"),
            "num_frames": suite.get("num_frames"),
            "num_windows": suite.get("num_windows"),
            "feature_dim": suite.get("feature_dim"),
            "window_frames": suite.get("window_frames"),
            "stride_frames": suite.get("stride_frames"),
            "split_policy": "single_episode_chronological_70_30",
            "raw_hdf5_required_for_full_public_regeneration": True,
            "raw_data_redistributed": False,
        },
        "setup_alignment": {
            "same_window_unit": "20-frame aligned windows",
            "same_stride": "5 frames",
            "same_feature_manifest": "results/episode_task_suite/feature_manifest.json",
            "same_shared_tensor": "results/episode_task_suite/shared_windows.npz",
            "same_split": "chronological 70/30 train/test split within the public sample episode",
            "same_baseline_pattern": "minimal interpretable heads plus compact neural MLP heads",
            "same_leakage_policy": "Target-side future, contact, object, caption, relation, and interaction signals are excluded from inputs unless language is explicitly the query.",
        },
        "source_files": [
            "docs/data/summary_metrics.json",
            "docs/data/task_walkthroughs.json",
            "docs/data/tier2_task_suite.json",
            "results/episode_task_suite/summary_report.json",
            "results/episode_task_suite/tier2_task_suite/tier2_task_suite_results.json",
            "results/episode_task_suite/windows.csv",
            "results/episode_task_suite/feature_manifest.json",
        ],
        "tasks": tasks,
    }


def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def render_markdown(payload: dict[str, Any]) -> str:
    scope = payload["dataset_scope"]
    lines = [
        "# Unified 20-Task Suite",
        "",
        "The public Xperience-10M sample task surface is one unified set of 20 tasks.",
        "All task contracts are presented together under the same window, split,",
        "feature, baseline, and leakage-control contract.",
        "",
        "Historical artifact paths containing `tier2_task_suite` are kept for stable",
        "links, but they should be read as provenance directories inside the unified task suite, not",
        "as a separate benchmark tier.",
        "",
        "## Shared Setup",
        "",
        f"- Episode scope: `{scope['sample_episode_count']}` public sample episode.",
        f"- Frames/windows: `{scope['num_frames']:,}` frames and `{scope['num_windows']:,}` aligned windows.",
        f"- Windowing: `{scope['window_frames']}` frames per window, stride `{scope['stride_frames']}` frames.",
        f"- Feature vector: `{scope['feature_dim']:,}` dimensions from the shared feature manifest.",
        "- Split: chronological 70/30 train/test by time within the sample episode.",
        "- Baselines: minimal interpretable heads and compact neural MLP heads.",
        "- Raw data: MP4/HDF5/RRD files are not redistributed.",
        "",
        "## Task Table",
        "",
        "| # | Task | Artifact id | Input -> output | Primary metric | Minimal | Neural |",
        "| ---: | --- | --- | --- | --- | ---: | ---: |",
    ]
    for row in payload["tasks"]:
        metric_direction = "higher better" if row.get("metric_direction") == "higher" else "lower better"
        lines.append(
            "| {num} | {name} | `{task_id}` | {inp} -> {out} | {metric} ({direction}) | {minimal} | {neural} |".format(
                num=row["task_number"],
                name=row["task_display_name"],
                task_id=row["task_id"],
                inp=row.get("input_short") or row.get("input"),
                out=row.get("output_short") or row.get("output"),
                metric=row.get("metric_name") or row.get("metric_key"),
                direction=metric_direction,
                minimal=fmt(row.get("minimal_primary_metric")),
                neural=fmt(row.get("neural_primary_metric")),
            )
        )
    lines.extend(
        [
            "",
            "## Machine-Readable Copy",
            "",
            "The JSON mirror is `docs/data/task_suite_20.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    payload = build_payload()
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    OUTPUT_MD.write_text(render_markdown(payload), encoding="utf-8")
    print(f"PASS: wrote {OUTPUT_JSON}")
    print(f"PASS: wrote {OUTPUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

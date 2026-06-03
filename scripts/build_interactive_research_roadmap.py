#!/usr/bin/env python3
"""Build the interactive research-roadmap data contract for the public site."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DOCS_DATA = ROOT / "docs" / "data"
RESULTS = ROOT / "results" / "episode_task_suite"
GITHUB_BLOB = "https://github.com/ChaoYue0307/ropedia-xperience-10m-task-suite/blob/main"


def repo_link(path: str) -> str:
    return f"{GITHUB_BLOB}/{path}"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rounded_metric(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def metric_summary(metric: dict[str, Any] | None) -> dict[str, Any]:
    if not metric:
        return {}
    return {
        "key": metric.get("key"),
        "name": metric.get("name"),
        "direction": metric.get("direction"),
        "minimal": rounded_metric(metric.get("minimal")),
        "neural_mlp": rounded_metric(metric.get("neural_mlp")),
        "better_baseline": metric.get("better_baseline"),
    }


def task_evidence_links(task_id: str) -> list[dict[str, str]]:
    candidates = [
        ("Minimal metrics", f"results/episode_task_suite/{task_id}/metrics.json"),
        ("Neural metrics", f"results/episode_task_suite/neural_mlp/{task_id}/metrics.json"),
        ("Minimal predictions", f"results/episode_task_suite/{task_id}/predictions.csv"),
        ("Neural predictions", f"results/episode_task_suite/neural_mlp/{task_id}/predictions.csv"),
        ("Confusion matrix", f"results/episode_task_suite/{task_id}/confusion_matrix.csv"),
        ("Neural confusion matrix", f"results/episode_task_suite/neural_mlp/{task_id}/confusion_matrix.csv"),
    ]
    links = [
        {"label": "Task walkthrough", "href": "data/task_walkthroughs.json"},
        {"label": "Single-episode explorer", "href": "single_episode_explorer.html"},
    ]
    for label, relative_path in candidates:
        if (ROOT / relative_path).exists():
            links.append({"label": label, "href": repo_link(relative_path)})
    return links


def task_payload(
    task_id: str,
    direction_task: dict[str, Any],
    walkthrough: dict[str, Any],
) -> dict[str, Any]:
    metric = direction_task.get("metric") or walkthrough.get("metric") or {}
    return {
        "id": task_id,
        "display_name": walkthrough.get("display_name") or direction_task.get("name") or task_id,
        "research_name": walkthrough.get("research_name") or direction_task.get("name") or task_id,
        "family": direction_task.get("family") or walkthrough.get("task_family"),
        "architecture_family": walkthrough.get("architecture_family"),
        "primary_direction": direction_task.get("primary_direction"),
        "direction_roles": direction_task.get("direction_roles", {}),
        "modalities": walkthrough.get("modalities", []),
        "case_study": walkthrough.get("case_study"),
        "input": walkthrough.get("input"),
        "input_short": walkthrough.get("input_short"),
        "process_short": walkthrough.get("process_short"),
        "output_short": walkthrough.get("output_short"),
        "module_summary": walkthrough.get("module_summary"),
        "current_limit": direction_task.get("current_limit") or walkthrough.get("failure_mode"),
        "why": direction_task.get("why"),
        "metric": metric_summary(metric),
        "evidence_links": task_evidence_links(task_id),
    }


def phase_payload(phases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stage_map = {
        "implemented": "now",
        "active": "scale_up",
        "next": "omni",
        "planned": "future",
    }
    return [
        {
            "id": phase.get("id"),
            "name": phase.get("name"),
            "status": phase.get("status"),
            "stage": stage_map.get(str(phase.get("status", "")).lower(), "future"),
            "entry_condition": phase.get("entry_condition"),
            "deliverables": phase.get("deliverables", []),
            "completion_evidence": phase.get("completion_evidence", []),
            "reader_takeaway": phase.get("reader_takeaway"),
        }
        for phase in phases
    ]


def main() -> int:
    directions_doc = load_json(DOCS_DATA / "research_directions.json")
    walkthroughs = load_json(DOCS_DATA / "task_walkthroughs.json")
    roadmap = load_json(DOCS_DATA / "research_roadmap.json")
    foundation_plan = load_json(DOCS_DATA / "foundation_model_plan.json")
    summary_metrics = load_json(DOCS_DATA / "summary_metrics.json")
    episode_summary = load_json(RESULTS / "summary_report.json")
    feature_manifest = load_json(RESULTS / "feature_manifest.json")
    extension_doc = load_json(DOCS_DATA / "research_direction_extensions.json")

    tasks: dict[str, dict[str, Any]] = {}
    for task_id, direction_task in directions_doc.get("tasks", {}).items():
        tasks[task_id] = task_payload(
            task_id,
            direction_task,
            walkthroughs.get("tasks", {}).get(task_id, {}),
        )

    directions = []
    for code, direction in directions_doc.get("directions", {}).items():
        linked_tasks = [tasks[task_id] for task_id in direction.get("tasks", []) if task_id in tasks]
        extension_tasks = [
            {
                "id": task_id,
                "name": spec.get("name"),
                "family": spec.get("family"),
                "metric_name": spec.get("metric_name"),
                "current_limit": spec.get("current_limit"),
            }
            for task_id, spec in extension_doc.get("task_specs", {}).items()
            if spec.get("direction") == code
        ]
        directions.append(
            {
                "code": code,
                "id": direction.get("id"),
                "name": direction.get("name"),
                "focus": direction.get("focus"),
                "preferred_background": direction.get("preferred_background"),
                "current_status": direction.get("current_status"),
                "current_readout": direction.get("current_readout"),
                "next_steps": direction.get("next_steps", []),
                "counts": direction.get("counts", {}),
                "task_ids": direction.get("tasks", []),
                "tasks": linked_tasks,
                "extension_tasks": extension_tasks,
            }
        )

    omni = summary_metrics.get("omni_relay", {})
    payload = {
        "title": "Interactive Research Roadmap",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_files": [
            "docs/data/research_directions.json",
            "docs/data/task_walkthroughs.json",
            "docs/data/research_roadmap.json",
            "docs/data/foundation_model_plan.json",
            "docs/data/summary_metrics.json",
            "docs/data/research_direction_extensions.json",
            "results/episode_task_suite/summary_report.json",
            "results/episode_task_suite/feature_manifest.json",
        ],
        "scope": {
            "sample_episode_count": walkthroughs.get("scope", {}).get("episode_count", 1),
            "num_frames": episode_summary.get("num_frames"),
            "num_windows": episode_summary.get("num_windows"),
            "feature_dim": episode_summary.get("feature_dim"),
            "window_frames": episode_summary.get("window_frames"),
            "stride_frames": episode_summary.get("stride_frames"),
            "feature_blocks": len(feature_manifest),
            "warning": walkthroughs.get("scope", {}).get("warning"),
        },
        "baseline_summary": {
            "task_count": len(tasks),
            "baseline_heads": "minimal and neural MLP heads",
            "split": "chronological single-episode split for public-sample diagnostics",
            "current_use": "task design, data-contract validation, case studies, and baseline comparison",
        },
        "scale_up": {
            "target_episodes": omni.get("target_episodes"),
            "candidate_scan_top_level_sessions": omni.get("candidate_scan_top_level_sessions"),
            "valid_candidates": omni.get("valid_candidates"),
            "estimated_bytes": omni.get("estimated_bytes"),
            "status": omni.get("status"),
            "access_status": omni.get("access_status"),
            "exclude": omni.get("exclude", []),
            "selection_strategy": omni.get("selection_strategy"),
        },
        "omni_plan": {
            "backbone": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
            "adapter": "LoRA rank 16, alpha 32, dropout 0.05",
            "first_pilot": "32 held-out-episode pilot after valid episodes are staged",
            "training_unit": "episode-level split, window-level supervised examples",
            "evaluation": [
                "JSON validity",
                "action macro-F1",
                "subtask accuracy",
                "transition accuracy",
                "next-action accuracy",
                "contact accuracy",
                "object micro-F1",
                "held-out episode count",
            ],
        },
        "foundation_model_plan": {
            "status": foundation_plan.get("status"),
            "decision": foundation_plan.get("decision", {}),
            "model_families": foundation_plan.get("model_families", []),
            "execution_order": foundation_plan.get("execution_order", []),
            "evaluation_additions": foundation_plan.get("evaluation_additions", []),
            "source_links": foundation_plan.get("source_links", []),
        },
        "phases": phase_payload(roadmap.get("phases", [])),
        "directions": directions,
        "tasks": list(tasks.values()),
    }

    out_path = DOCS_DATA / "research_roadmap_interactive.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

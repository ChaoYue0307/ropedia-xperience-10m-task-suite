#!/usr/bin/env python3
"""Validate the public 12-task card and walkthrough surface.

This gate is deliberately about presentation integrity, not model quality. The
repo keeps snake_case artifact ids for reproducibility, but the public website
task cards and interactive player should use research-readable names and clear
input/process/output wording.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TASK_JSON = ROOT / "docs/data/task_walkthroughs.json"
WEBSITE = ROOT / "docs/index.html"
WALKTHROUGH_MD = ROOT / "results/episode_task_suite/task_walkthroughs/TASK_WALKTHROUGHS.md"
OUTPUT = ROOT / "docs/data/task_surface_integrity.json"

EXPECTED_TASKS = {
    "timeline_action": "Action Recognition",
    "timeline_subtask": "Procedure Step Recognition",
    "transition_detection": "Action Boundary Detection",
    "next_action": "Next-Action Prediction",
    "hand_trajectory_forecast": "Hand Trajectory Forecasting",
    "contact_prediction": "Contact State Prediction",
    "object_relevance": "Object Relevance Prediction",
    "caption_grounding": "Language Grounding",
    "cross_modal_retrieval": "Cross-Modal Retrieval",
    "modality_reconstruction": "Cross-Modal Reconstruction",
    "temporal_order": "Temporal Order Verification",
    "misalignment_detection": "Multimodal Synchronization Detection",
}

EXPECTED_EXTENSION_NAMES = {
    "body_motion_intensity": "Body and Hand Motion Intensity",
    "multi_view_consistency_retrieval": "Multi-View Consistency Retrieval",
    "action_phase_progress": "Action Phase Progress Estimation",
    "ego_motion_forecast": "Short-Horizon Ego-Motion Forecasting",
}

REQUIRED_TASK_FIELDS = {
    "display_name",
    "research_name",
    "task_family",
    "architecture_family",
    "primary_direction",
    "card_blurb",
    "input_short",
    "process_short",
    "output_short",
    "modalities",
    "poster_modality",
    "case_study",
    "input",
    "output",
    "middle_modules",
    "metric",
    "failure_mode",
    "artifact_id",
    "plain_goal",
}

DISPLAY_FIELDS = {
    "display_name",
    "research_name",
    "card_blurb",
    "input_short",
    "process_short",
    "output_short",
    "plain_goal",
}

ALLOWED_FAMILIES = {"supervised", "forecast", "retrieval", "diagnostic"}
MODALITY_ASSETS = {
    "video": "docs/assets/modalities/video.jpg",
    "audio": "docs/assets/modalities/audio.png",
    "depth": "docs/assets/modalities/depth.jpg",
    "pose_slam": "docs/assets/modalities/pose_slam.png",
    "motion_capture": "docs/assets/modalities/motion_capture.png",
    "inertial": "docs/assets/modalities/inertial.png",
    "language": "docs/assets/modalities/language.png",
}

RAW_ID_PATTERN = re.compile(r"\b[a-z]+(?:_[a-z0-9]+)+\b")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def check(condition: bool, name: str, failures: list[dict[str, Any]], **details: Any) -> dict[str, Any]:
    record = {"name": name, "status": "pass" if condition else "fail", **details}
    if not condition:
        failures.append(record)
    return record


def function_body(source: str, name: str) -> str:
    marker = f"function {name}("
    start = source.find(marker)
    if start < 0:
        return ""
    brace = source.find("{", start)
    if brace < 0:
        return ""
    depth = 0
    for index in range(brace, len(source)):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    return source[start:]


def validate_tasks(payload: dict[str, Any], failures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    tasks = payload.get("tasks", {})
    checks.append(check(isinstance(tasks, dict), "tasks_object_present", failures))
    if not isinstance(tasks, dict):
        return checks

    task_ids = set(tasks)
    checks.append(
        check(
            len(tasks) == len(EXPECTED_TASKS),
            "exactly_12_tasks",
            failures,
            observed=len(tasks),
            expected=len(EXPECTED_TASKS),
        )
    )
    checks.append(
        check(
            task_ids == set(EXPECTED_TASKS),
            "expected_task_ids_present",
            failures,
            missing=sorted(set(EXPECTED_TASKS) - task_ids),
            extra=sorted(task_ids - set(EXPECTED_TASKS)),
        )
    )

    for task_id, task in tasks.items():
        if not isinstance(task, dict):
            checks.append(check(False, f"{task_id}: task_record_object", failures))
            continue
        missing_fields = sorted(REQUIRED_TASK_FIELDS - set(task))
        checks.append(
            check(not missing_fields, f"{task_id}: required_fields", failures, missing=missing_fields)
        )
        expected_name = EXPECTED_TASKS.get(task_id)
        checks.append(
            check(
                task.get("display_name") == expected_name,
                f"{task_id}: human_readable_display_name",
                failures,
                expected=expected_name,
                observed=task.get("display_name"),
            )
        )
        checks.append(
            check(
                task.get("artifact_id") == task_id,
                f"{task_id}: artifact_id_matches_key",
                failures,
                observed=task.get("artifact_id"),
            )
        )
        for field in DISPLAY_FIELDS:
            value = str(task.get(field, ""))
            raw_hits = [hit for hit in RAW_ID_PATTERN.findall(value) if hit in EXPECTED_TASKS or hit in MODALITY_ASSETS]
            checks.append(
                check(
                    not raw_hits,
                    f"{task_id}: public_field_{field}_is_human_readable",
                    failures,
                    value=value,
                    raw_hits=raw_hits,
                )
            )
        family = task.get("task_family")
        checks.append(
            check(
                family in ALLOWED_FAMILIES,
                f"{task_id}: known_task_family",
                failures,
                observed=family,
                allowed=sorted(ALLOWED_FAMILIES),
            )
        )
        modalities = task.get("modalities", [])
        checks.append(
            check(
                isinstance(modalities, list) and modalities,
                f"{task_id}: modality_list_present",
                failures,
                observed=modalities,
            )
        )
        if isinstance(modalities, list):
            unknown = [item for item in modalities if item not in MODALITY_ASSETS]
            missing_assets = [
                MODALITY_ASSETS[item]
                for item in modalities
                if item in MODALITY_ASSETS and not (ROOT / MODALITY_ASSETS[item]).exists()
            ]
            checks.append(
                check(
                    not unknown,
                    f"{task_id}: known_modalities",
                    failures,
                    unknown=unknown,
                )
            )
            checks.append(
                check(
                    not missing_assets,
                    f"{task_id}: modality_assets_exist",
                    failures,
                    missing=missing_assets,
                )
            )
            checks.append(
                check(
                    task.get("poster_modality") in modalities,
                    f"{task_id}: poster_modality_in_task_modalities",
                    failures,
                    poster_modality=task.get("poster_modality"),
                    modalities=modalities,
                )
            )
        metric = task.get("metric", {})
        metric_ok = (
            isinstance(metric, dict)
            and isinstance(metric.get("name"), str)
            and isinstance(metric.get("direction"), str)
            and isinstance(metric.get("minimal"), (int, float))
            and isinstance(metric.get("neural_mlp"), (int, float))
        )
        checks.append(
            check(
                metric_ok,
                f"{task_id}: numeric_minimal_and_neural_metrics",
                failures,
                metric=metric,
            )
        )
        checks.append(
            check(
                isinstance(task.get("middle_modules"), list) and len(task.get("middle_modules", [])) >= 3,
                f"{task_id}: middle_modules_explain_process",
                failures,
                observed_count=len(task.get("middle_modules", [])) if isinstance(task.get("middle_modules"), list) else 0,
            )
        )
    return checks


def validate_markdown(source: str, tasks: dict[str, Any], failures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for task_id, display_name in EXPECTED_TASKS.items():
        expected_heading = f"### {display_name} (`{task_id}`)"
        checks.append(
            check(
                expected_heading in source,
                f"markdown_heading_present:{task_id}",
                failures,
                expected=expected_heading,
            )
        )
    checks.append(
        check(
            source.count("### ") == len(EXPECTED_TASKS),
            "markdown_has_12_task_sections",
            failures,
            observed=source.count("### "),
        )
    )
    checks.append(
        check(
            all(str(task.get("case_study", "")) in source for task in tasks.values()),
            "markdown_contains_case_studies",
            failures,
        )
    )
    return checks


def validate_website(source: str, failures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    required_markers = [
        'id="taskPlayer"',
        'id="taskGrid"',
        'id="walkthroughSelector"',
        'id="playerStoryboard"',
        'id="playerFrameChip"',
        'id="playerFrameCaption"',
        'id="playerScrub"',
        'fetch("data/task_walkthroughs.json"',
        'class="task-card"',
        'class="task-card-media"',
        'class="story-button',
        'class="flow-step',
        'id="playerPlay"',
        'id="playerPrev"',
        'id="playerNext"',
    ]
    for marker in required_markers:
        checks.append(
            check(marker in source, f"website_marker_present:{marker}", failures, marker=marker)
        )
    task_card_renderer = function_body(source, "renderTaskCards")
    selector_renderer = function_body(source, "renderSelector")
    player_renderer = function_body(source, "renderPlayer")
    checks.append(
        check(
            "artifact-id" not in source,
            "website_no_artifact_id_css_or_markup",
            failures,
        )
    )
    checks.append(
        check(
            "artifact_id" not in task_card_renderer,
            "task_cards_do_not_render_artifact_ids",
            failures,
        )
    )
    checks.append(
        check(
            "task.display_name" in task_card_renderer and "task.research_name" in task_card_renderer,
            "task_cards_render_human_names",
            failures,
        )
    )
    checks.append(
        check(
            "task.input_short" in task_card_renderer and "task.process_short" in task_card_renderer and "task.output_short" in task_card_renderer,
            "task_cards_render_input_process_output",
            failures,
        )
    )
    checks.append(
        check(
            "task.poster_modality" in task_card_renderer and "task-card-media" in task_card_renderer,
            "task_cards_use_representative_modality_thumbnail",
            failures,
        )
    )
    checks.append(
        check(
            all(
                needle in player_renderer
                for needle in ["playerPoster", "middle_modules"]
            )
            and all(needle in source for needle in ["playerProgress", "renderStageFrame(task, index)"])
            and all(needle in source for needle in ['id="playerPlay"', 'id="playerPrev"', 'id="playerNext"']),
            "interactive_player_wired_to_task_metadata",
            failures,
        )
    )
    checks.append(
        check(
            all(needle in source for needle in ["function setActiveStage", "function advancePlayer", "playerScrub"]),
            "interactive_video_storyboard_controls_present",
            failures,
        )
    )
    checks.append(
        check(
            "task.display_name" in selector_renderer and "artifact_id" not in selector_renderer,
            "selector_uses_human_names",
            failures,
        )
    )
    for artifact_id, display_name in EXPECTED_EXTENSION_NAMES.items():
        checks.append(
            check(
                f"<h3>{artifact_id}</h3>" not in source and display_name in source,
                f"extension_probe_uses_human_name:{artifact_id}",
                failures,
                expected=display_name,
            )
        )
    return checks


def build_report() -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []

    inputs_present = {
        "task_walkthroughs_json": TASK_JSON.exists(),
        "website_index": WEBSITE.exists(),
        "walkthrough_markdown": WALKTHROUGH_MD.exists(),
    }
    checks.append(
        check(
            all(inputs_present.values()),
            "required_task_surface_inputs_present",
            failures,
            inputs=inputs_present,
        )
    )
    if not all(inputs_present.values()):
        return {
            "status": "fail",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "summary": {"task_count": 0, "failure_count": len(failures)},
            "checks": checks,
            "failures": failures,
        }

    task_payload = load_json(TASK_JSON)
    website_source = WEBSITE.read_text(encoding="utf-8")
    markdown_source = WALKTHROUGH_MD.read_text(encoding="utf-8")
    tasks = task_payload.get("tasks", {}) if isinstance(task_payload.get("tasks", {}), dict) else {}

    checks.extend(validate_tasks(task_payload, failures))
    checks.extend(validate_markdown(markdown_source, tasks, failures))
    checks.extend(validate_website(website_source, failures))

    task_families = {}
    task_modalities = {}
    for task in tasks.values():
        family = task.get("task_family")
        if isinstance(family, str):
            task_families[family] = task_families.get(family, 0) + 1
        for modality in task.get("modalities", []):
            task_modalities[modality] = task_modalities.get(modality, 0) + 1

    return {
        "status": "pass" if not failures else "fail",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "summary": {
            "task_count": len(tasks),
            "expected_task_count": len(EXPECTED_TASKS),
            "task_family_counts": dict(sorted(task_families.items())),
            "modality_usage_counts": dict(sorted(task_modalities.items())),
            "interactive_surface": "task cards plus scrub/play/chapter walkthrough storyboard",
            "failure_count": len(failures),
        },
        "checks": checks,
        "failures": failures,
    }


def main() -> int:
    report = build_report()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{report['status'].upper()}: wrote {OUTPUT}")
    if report["status"] != "pass":
        for failure in report["failures"][:40]:
            print(f"- {failure['name']}")
        if len(report["failures"]) > 40:
            print(f"- ... {len(report['failures']) - 40} more failures")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

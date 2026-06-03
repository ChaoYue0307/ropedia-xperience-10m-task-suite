#!/usr/bin/env python3
"""Build the rendered website check report from browser observations."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = Path("/tmp/xperience_rendered_site_observations.json")
OUTPUT_JSON = ROOT / "docs/data/rendered_site_check.json"
OUTPUT_MD = ROOT / "RENDERED_SITE_CHECK.md"


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(
            f"Observation file missing: {path}. Run the browser flow and pass --input."
        ) from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Observation file is not valid JSON: {path}: {exc}") from exc


def check(name: str, passed: bool, reason: str, **detail: Any) -> dict[str, Any]:
    return {"name": name, "status": "pass" if passed else "fail", "reason": reason, **detail}


def build_report(observations: dict[str, Any]) -> dict[str, Any]:
    viewport = observations.get("viewport") or {}
    checks = [
        check(
            "page_identity",
            observations.get("title") == "Ropedia Xperience-10M Task Suite"
            and str(observations.get("url", "")).startswith("http://127.0.0.1:"),
            "The rendered page should load the expected local site title and URL.",
            title=observations.get("title"),
            url=observations.get("url"),
        ),
        check(
            "first_meaningful_content",
            observations.get("h1") == "Ropedia Xperience-10M Research Task Lab.",
            "The first meaningful heading should identify the research task lab.",
            h1=observations.get("h1"),
        ),
        check(
            "responsive_viewport_recorded",
            int(viewport.get("width", 0)) >= 360 and int(viewport.get("height", 0)) >= 600,
            "The browser run should record a narrow responsive viewport large enough to expose the mobile/tablet layout.",
            viewport=viewport,
        ),
        check(
            "tabbed_research_navigation",
            observations.get("projectTabCount") == 5
            and observations.get("panelCount", 0) >= 19
            and observations.get("selectedProjectTab") == "tab-data",
            "The rendered top-level tab system should switch to the Data tab for the walkthrough deep link.",
            project_tab_count=observations.get("projectTabCount"),
            panel_count=observations.get("panelCount"),
            selected_project_tab=observations.get("selectedProjectTab"),
        ),
        check(
            "task_and_modality_cards_render",
            observations.get("taskCardCount") == 12 and observations.get("atlasCardCount") == 7,
            "The rendered task and modality sections should expose all 12 task cards and seven modality cards.",
            task_card_count=observations.get("taskCardCount"),
            atlas_card_count=observations.get("atlasCardCount"),
        ),
        check(
            "walkthrough_deep_link",
            observations.get("visibleWalkthrough") is True
            and observations.get("selectorButtonCount") == 12
            and observations.get("storyButtonCount") == 4,
            "The walkthrough deep link should reveal the walkthrough player, all task selectors, and four chapter controls.",
            visible_walkthrough=observations.get("visibleWalkthrough"),
            selector_button_count=observations.get("selectorButtonCount"),
            story_button_count=observations.get("storyButtonCount"),
        ),
        check(
            "walkthrough_interaction",
            observations.get("activeSelectorCount") == 1
            and observations.get("activeStoryCount") == 1
            and observations.get("activeTask") == "Procedure Step Recognition"
            and observations.get("activeStory") == "Process"
            and observations.get("frameChip") == "Step 2 / 4 · Process",
            "Clicking Next and the Process chapter should update the active task, chapter, counter, and frame label.",
            active_task=observations.get("activeTask"),
            active_story=observations.get("activeStory"),
            player_counter=observations.get("playerCounter"),
            frame_chip=observations.get("frameChip"),
            active_selector_count=observations.get("activeSelectorCount"),
            active_story_count=observations.get("activeStoryCount"),
        ),
        check(
            "rendered_check_resource_link",
            observations.get("renderedSiteCheckLinkPresent") is True,
            "The rendered page should expose the rendered website check JSON from the resource section.",
            rendered_site_check_link_present=observations.get("renderedSiteCheckLinkPresent"),
        ),
        check(
            "console_health",
            observations.get("consoleWarningsOrErrors") == 0,
            "The rendered flow should complete without browser console warnings or errors.",
            console_warnings_or_errors=observations.get("consoleWarningsOrErrors"),
        ),
    ]
    return {
        "title": "Ropedia Xperience-10M Rendered Website Check",
        "status": "pass" if all(item["status"] == "pass" for item in checks) else "fail",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "flow": observations.get("exercised_flow"),
        "checked_at_local": observations.get("checked_at_local"),
        "screenshot_path": observations.get("screenshot_path"),
        "observations": observations,
        "checks": checks,
    }


def markdown(report: dict[str, Any]) -> str:
    observations = report["observations"]
    viewport = observations.get("viewport") or {}
    lines = [
        "# Rendered Website Check",
        "",
        "This report records the latest browser-level check of the local static website.",
        "",
        f"Current status: **{report['status']}**",
        "",
        "## Browser Flow",
        "",
        f"- URL: `{observations.get('url')}`",
        f"- Title: `{observations.get('title')}`",
        f"- Viewport: `{viewport.get('width')} x {viewport.get('height')}`",
        f"- Flow: {report.get('flow')}",
        f"- Screenshot: `{report.get('screenshot_path')}`",
        "",
        "## Checks",
        "",
        "| Check | Status | What it covers |",
        "| --- | --- | --- |",
    ]
    for item in report["checks"]:
        label = item["name"].replace("_", " ").title()
        lines.append(f"| {label} | `{item['status']}` | {item['reason']} |")
    lines.extend(
        [
            "",
            "## Regenerate",
            "",
            "Run the local static website, exercise the walkthrough in a browser, save the observation JSON, then rebuild this report:",
            "",
            "```bash",
            "python scripts/build_rendered_site_check.py --input /tmp/xperience_rendered_site_observations.json",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()

    report = build_report(load_json(args.input))
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    OUTPUT_MD.write_text(markdown(report), encoding="utf-8")
    print(f"{report['status'].upper()}: wrote {OUTPUT_JSON}")
    print(f"{report['status'].upper()}: wrote {OUTPUT_MD}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

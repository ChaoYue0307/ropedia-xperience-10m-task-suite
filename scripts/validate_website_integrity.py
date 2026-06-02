#!/usr/bin/env python3
"""Validate static website links, anchors, image assets, and JSON data.

This is a local integrity check for the GitHub Pages / Hugging Face static
website. It intentionally does not fetch external URLs; it verifies that the
published local surface is internally coherent.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCS = ROOT / "docs"
DEFAULT_OUTPUT = DEFAULT_DOCS / "data/website_integrity.json"
DEFAULT_SITE_BASE = "/ropedia-xperience-10m-task-suite/"
LOCAL_ATTRS = {"href", "src"}
SKIP_SCHEMES = {"http", "https", "mailto", "tel", "data", "javascript"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class Reference:
    source: Path
    tag: str
    attr: str
    raw: str
    path_part: str
    fragment: str


class SiteParser(HTMLParser):
    def __init__(self, source: Path):
        super().__init__(convert_charrefs=True)
        self.source = source
        self.ids: list[str] = []
        self.references: list[Reference] = []
        self.images: list[str] = []

    def handle_starttag(self, tag: str, attrs):
        attr_map = dict(attrs)
        element_id = attr_map.get("id") or attr_map.get("name")
        if element_id:
            self.ids.append(element_id)

        for attr in LOCAL_ATTRS:
            raw = attr_map.get(attr)
            if not raw:
                continue
            parsed = urlsplit(raw)
            if parsed.scheme in SKIP_SCHEMES or parsed.netloc:
                continue
            path_part = unquote(parsed.path)
            fragment = unquote(parsed.fragment)
            if not path_part and not fragment:
                continue
            self.references.append(Reference(self.source, tag, attr, raw, path_part, fragment))
            if tag == "img" and attr == "src":
                self.images.append(path_part)


def parse_html(path: Path) -> SiteParser:
    parser = SiteParser(path)
    parser.feed(path.read_text(encoding="utf-8", errors="ignore"))
    return parser


def normalize_path_part(path_part: str, site_base: str) -> str:
    if path_part in {"", "/"}:
        return path_part
    normalized_base = "/" + site_base.strip("/") + "/"
    if path_part == normalized_base:
        return "/index.html"
    if path_part.startswith(normalized_base):
        return "/" + path_part[len(normalized_base):]
    return path_part


def resolve_reference(docs_root: Path, source: Path, path_part: str, site_base: str) -> Path:
    path_part = normalize_path_part(path_part, site_base)
    if not path_part:
        return source
    base = docs_root if path_part.startswith("/") else source.parent
    resolved = (base / path_part.lstrip("/")).resolve()
    if resolved.is_dir():
        return resolved / "index.html"
    return resolved


def relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def duplicate_ids(ids: list[str]) -> list[dict]:
    counts: dict[str, int] = {}
    for item in ids:
        counts[item] = counts.get(item, 0) + 1
    return [{"id": item, "count": count} for item, count in sorted(counts.items()) if count > 1]


def image_record(path: Path, docs_root: Path) -> dict:
    suffix = path.suffix.lower()
    record = {
        "path": relative(path, docs_root),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else 0,
    }
    if not path.exists():
        return record
    if suffix in IMAGE_SUFFIXES:
        with Image.open(path) as image:
            record.update({
                "width": int(image.width),
                "height": int(image.height),
                "format": image.format,
            })
    elif suffix == ".svg":
        text = path.read_text(encoding="utf-8", errors="ignore")
        viewbox = re.search(r'viewBox=["\']([^"\']+)["\']', text)
        record["format"] = "SVG"
        record["has_viewbox"] = bool(viewbox)
    return record


def validate(docs_root: Path, site_base: str) -> dict:
    html_files = sorted(docs_root.glob("*.html"))
    parsers = {path: parse_html(path) for path in html_files}
    anchors_by_file = {path: set(parser.ids) for path, parser in parsers.items()}

    missing_targets = []
    missing_anchors = []
    local_references = []
    external_reference_count = 0
    image_paths: set[Path] = set()

    for path, parser in parsers.items():
        text = path.read_text(encoding="utf-8", errors="ignore")
        external_reference_count += len(re.findall(r'https?://', text))
        for ref in parser.references:
            target = resolve_reference(docs_root, path, ref.path_part, site_base)
            target_rel = relative(target, docs_root)
            source_rel = relative(ref.source, docs_root)
            local_references.append({
                "source": source_rel,
                "tag": ref.tag,
                "attr": ref.attr,
                "raw": ref.raw,
                "target": target_rel,
                "fragment": ref.fragment,
            })
            if not target.exists():
                missing_targets.append({
                    "source": source_rel,
                    "raw": ref.raw,
                    "target": target_rel,
                })
                continue
            if ref.tag == "img" and ref.attr == "src":
                image_paths.add(target)
            if ref.fragment:
                anchor_target = target if target.suffix.lower() == ".html" else path
                if ref.path_part and target.suffix.lower() != ".html":
                    continue
                anchors = anchors_by_file.get(anchor_target)
                if anchors is None and anchor_target.exists() and anchor_target.suffix.lower() == ".html":
                    anchors = set(parse_html(anchor_target).ids)
                    anchors_by_file[anchor_target] = anchors
                if anchors is not None and ref.fragment not in anchors:
                    missing_anchors.append({
                        "source": source_rel,
                        "raw": ref.raw,
                        "target": relative(anchor_target, docs_root),
                        "fragment": ref.fragment,
                    })

    json_records = []
    invalid_json = []
    for path in sorted((docs_root / "data").glob("*.json")):
        rel_path = relative(path, docs_root)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            invalid_json.append({"path": rel_path, "error": str(exc)})
            continue
        json_records.append({
            "path": rel_path,
            "bytes": path.stat().st_size,
            "top_level_type": type(payload).__name__,
        })

    images = []
    invalid_images = []
    for path in sorted(image_paths):
        try:
            record = image_record(path, docs_root)
            images.append(record)
            if not record.get("exists") or record.get("bytes", 0) <= 0:
                invalid_images.append(record)
            if path.suffix.lower() in IMAGE_SUFFIXES and (record.get("width", 0) <= 0 or record.get("height", 0) <= 0):
                invalid_images.append(record)
            if path.suffix.lower() == ".svg" and not record.get("has_viewbox"):
                invalid_images.append(record)
        except Exception as exc:  # noqa: BLE001 - report image validation failures.
            invalid_images.append({"path": relative(path, docs_root), "error": str(exc)})

    duplicate_id_records = [
        {"path": relative(path, docs_root), "duplicates": duplicate_ids(parser.ids)}
        for path, parser in parsers.items()
    ]
    duplicate_id_records = [item for item in duplicate_id_records if item["duplicates"]]

    semantic_checks = []
    semantic_layout_failures = []
    index_path = docs_root / "index.html"
    index_text = index_path.read_text(encoding="utf-8", errors="ignore") if index_path.exists() else ""

    def section_pos(section_id: str) -> int:
        match = re.search(rf'<section\b[^>]*\bid="{re.escape(section_id)}"', index_text)
        return match.start() if match else -1

    suite_start = section_pos("suite")
    suite_end = section_pos("pipeline")
    suite_text = index_text[suite_start:suite_end] if suite_start >= 0 and suite_end > suite_start else ""
    overview_pos = section_pos("overview")
    protocol_pos = section_pos("protocol")
    evidence_pos = section_pos("evidence")
    dataset_start = section_pos("dataset-card")
    dataset_end = section_pos("suite")
    dataset_text = index_text[dataset_start:dataset_end] if dataset_start >= 0 and dataset_end > dataset_start else ""
    semantic_rules = [
        (
            "project_tabs_have_five_groups",
            'data-tab-key=',
            None,
            "The long research page should be grouped into five top-level tabs.",
        ),
        (
            "project_sections_are_assigned_to_tabs",
            'data-project-tab=',
            None,
            "Every major research section should be assigned to a tab group.",
        ),
        (
            "project_hash_router_preserves_deep_links",
            'activateTabForHash',
            None,
            "Deep links should open the correct tab instead of landing on hidden content.",
        ),
        (
            "project_tabs_use_accessible_roles",
            'role="tab"',
            None,
            "The tabbed research dashboard should expose tablist/tab semantics.",
        ),
        (
            "project_sections_are_labeled_tabpanels",
            'role="tabpanel"',
            None,
            "Every tabbed research section should expose a labeled panel role.",
        ),
        (
            "project_tabs_update_selected_state",
            'aria-selected',
            None,
            "Tab activation should update selected state for assistive technology.",
        ),
        (
            "project_tabs_support_keyboard_navigation",
            'moveProjectTabFocus',
            None,
            "Keyboard users should be able to switch project tabs with arrow, Home, and End keys.",
        ),
        (
            "project_overview_precedes_progress_ledger",
            '<section id="overview">',
            '<section id="evidence">',
            "The project overview should appear before the deeper progress ledger.",
        ),
        (
            "project_status_links_json",
            'data/project_status.json',
            None,
            "The website should expose the machine-readable project status.",
        ),
        (
            "evaluation_protocol_between_overview_and_progress",
            '<section id="protocol">',
            '<section id="evidence">',
            "The evaluation protocol should appear before the deeper evidence ledger.",
        ),
        (
            "evaluation_protocol_links_json",
            'data/evaluation_protocol.json',
            None,
            "The website should expose the machine-readable evaluation protocol.",
        ),
        (
            "figure_index_links_json",
            'data/figure_index.json',
            None,
            "The website should expose the machine-readable figure index.",
        ),
        (
            "suite_task_map_precedes_modality_atlas",
            '<div class="figure-pan" id="task-suite-map">',
            '<div class="modality-atlas-panel"',
            "The Suite anchor should show the full 12-task map before the modality atlas.",
        ),
        (
            "suite_modality_atlas_contains_seven_cards",
            'class="atlas-card',
            None,
            "The modality atlas should expose seven sample modalities.",
        ),
        (
            "dataset_card_section_mentions_sample_license",
            'cc-by-nc-4.0',
            None,
            "The dataset-card section should preserve the public sample card license.",
        ),
        (
            "dataset_card_section_mentions_api_episode_listing",
            '12,103 episode folders',
            None,
            "The dataset-card section should distinguish HF API listing metadata from local data possession.",
        ),
        (
            "dataset_card_section_links_source_alignment_audit",
            'data/source_alignment_audit.json',
            None,
            "The dataset-card section should expose the generated source-alignment report.",
        ),
        (
            "task_player_surface_present",
            'id="taskPlayer"',
            None,
            "The website should expose the interactive task walkthrough/player.",
        ),
        (
            "task_player_uses_walkthrough_json",
            'data/task_walkthroughs.json',
            None,
            "The task player and task cards should read the generated walkthrough JSON.",
        ),
        (
            "task_cards_use_human_research_names",
            'Action Recognition',
            None,
            "The public task surface should use readable research task names.",
        ),
    ]
    for name, marker, after_marker, reason in semantic_rules:
        if name == "project_tabs_have_five_groups":
            tab_count = index_text.count(marker)
            passed = tab_count == 5
            detail = {"tab_count": tab_count}
        elif name == "project_sections_are_assigned_to_tabs":
            section_count = index_text.count(marker)
            passed = section_count >= 19
            detail = {"section_count": section_count}
        elif name == "project_hash_router_preserves_deep_links":
            marker_count = index_text.count(marker)
            passed = marker_count >= 2 and "sectionTabMap" in index_text
            detail = {"marker_count": marker_count, "has_section_tab_map": "sectionTabMap" in index_text}
        elif name == "project_tabs_use_accessible_roles":
            tab_role_count = index_text.count(marker)
            passed = 'role="tablist"' in index_text and tab_role_count == 5
            detail = {"tab_role_count": tab_role_count, "has_tablist": 'role="tablist"' in index_text}
        elif name == "project_sections_are_labeled_tabpanels":
            panel_count = index_text.count(marker)
            passed = panel_count >= 19 and index_text.count('aria-labelledby="tab-') >= 19
            detail = {
                "panel_count": panel_count,
                "labeled_panel_count": index_text.count('aria-labelledby="tab-'),
            }
        elif name == "project_tabs_update_selected_state":
            selected_count = index_text.count(marker)
            passed = selected_count >= 6 and 'setAttribute("aria-selected"' in index_text
            detail = {
                "selected_count": selected_count,
                "updates_selected_state": 'setAttribute("aria-selected"' in index_text,
            }
        elif name == "project_tabs_support_keyboard_navigation":
            marker_count = index_text.count(marker)
            passed = marker_count >= 2 and "ArrowRight" in index_text and "Home" in index_text and "End" in index_text
            detail = {
                "marker_count": marker_count,
                "has_arrow_navigation": "ArrowRight" in index_text and "ArrowLeft" in index_text,
                "has_home_end_navigation": "Home" in index_text and "End" in index_text,
            }
        elif name == "suite_modality_atlas_contains_seven_cards":
            card_count = len(re.findall(r'class="atlas-card(?:\s|")', suite_text))
            passed = card_count == 7
            detail = {"card_count": card_count}
        elif name.startswith("dataset_card_section_"):
            marker_count = dataset_text.count(marker)
            passed = marker_count >= 1
            detail = {"marker_count": marker_count}
        elif name == "project_overview_precedes_progress_ledger":
            passed = overview_pos >= 0 and evidence_pos >= 0 and overview_pos < evidence_pos
            detail = {"overview_index": overview_pos, "evidence_index": evidence_pos}
        elif name == "evaluation_protocol_between_overview_and_progress":
            passed = overview_pos >= 0 and protocol_pos >= 0 and evidence_pos >= 0 and overview_pos < protocol_pos < evidence_pos
            detail = {"overview_index": overview_pos, "protocol_index": protocol_pos, "evidence_index": evidence_pos}
        elif name in {
            "project_status_links_json",
            "figure_index_links_json",
            "task_player_surface_present",
            "task_player_uses_walkthrough_json",
            "task_cards_use_human_research_names",
        }:
            marker_count = index_text.count(marker)
            passed = marker_count >= 1
            detail = {"marker_count": marker_count}
        elif name == "evaluation_protocol_links_json":
            marker_count = index_text.count(marker)
            passed = marker_count >= 1
            detail = {"marker_count": marker_count}
        else:
            marker_pos = suite_text.find(marker)
            after_pos = suite_text.find(after_marker or "")
            passed = marker_pos >= 0 and after_pos >= 0 and marker_pos < after_pos
            detail = {"first_marker_index": marker_pos, "second_marker_index": after_pos}
        check = {"name": name, "status": "pass" if passed else "fail", "reason": reason, **detail}
        semantic_checks.append(check)
        if not passed:
            semantic_layout_failures.append(check)

    failures = {
        "missing_targets": missing_targets,
        "missing_anchors": missing_anchors,
        "duplicate_ids": duplicate_id_records,
        "invalid_json": invalid_json,
        "invalid_images": invalid_images,
        "semantic_layout": semantic_layout_failures,
    }
    failure_count = sum(len(items) for items in failures.values())

    return {
        "status": "pass" if failure_count == 0 else "fail",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "docs_root": str(docs_root),
        "site_base": site_base,
        "summary": {
            "html_pages": len(html_files),
            "local_references": len(local_references),
            "external_reference_count": external_reference_count,
            "json_files": len(json_records),
            "image_assets_referenced": len(images),
            "failure_count": failure_count,
        },
        "failures": failures,
        "semantic_checks": semantic_checks,
        "html_pages": [
            {
                "path": relative(path, docs_root),
                "id_count": len(parser.ids),
                "reference_count": len(parser.references),
                "image_count": len(parser.images),
            }
            for path, parser in parsers.items()
        ],
        "json_files": json_records,
        "images": images,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs-root", type=Path, default=DEFAULT_DOCS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--site-base", default=DEFAULT_SITE_BASE)
    args = parser.parse_args()

    report = validate(args.docs_root.resolve(), args.site_base)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{report['status'].upper()}: wrote {args.output}")
    if report["status"] != "pass":
        for kind, items in report["failures"].items():
            for item in items[:20]:
                print(f"- {kind}: {item}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

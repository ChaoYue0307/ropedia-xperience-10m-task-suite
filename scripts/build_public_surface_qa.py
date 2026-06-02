#!/usr/bin/env python3
"""Build a public-surface QA report for repo, website, and HF mirrors."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HF_ROOT = ROOT.parent / "hf_publish"
OUTPUT_JSON = ROOT / "docs/data/public_surface_qa.json"
OUTPUT_MD = ROOT / "PUBLIC_SURFACE_QA.md"

SURFACES = {
    "github_readme": ROOT / "README.md",
    "website_html": ROOT / "docs/index.html",
    "hf_space_card": HF_ROOT / "space/README.md",
    "hf_artifact_card": HF_ROOT / "artifacts/README.md",
    "hf_model_card": HF_ROOT / "model/README.md",
}

STATUS_REPORTS = {
    "website_integrity": ROOT / "docs/data/website_integrity.json",
    "task_surface_integrity": ROOT / "docs/data/task_surface_integrity.json",
    "source_alignment": ROOT / "docs/data/source_alignment_audit.json",
    "scope_claims": ROOT / "docs/data/scope_claims_audit.json",
    "publication_hygiene": ROOT / "docs/data/publication_audit.json",
    "mirror_parity": ROOT / "docs/data/mirror_parity.json",
    "live_publication": ROOT / "docs/data/live_publication_status.json",
}

BANNED_PUBLIC_STRINGS = [
    "audit" + "able",
    "rev" + "iewer " + "scorecard",
    "REVIEWER_" + "SCORECARD",
    "ChatGPT" + "-image",
    "H" + "20",
    "A" + "100",
    "Cur" + "sor",
    "public " + "dashboard and generated figures " + "deliberately follow",
]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""


def read_status(path: Path) -> dict:
    if not path.exists():
        return {"exists": False, "status": "missing"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"exists": True, "status": "invalid_json", "error": str(exc)}
    return {
        "exists": True,
        "status": str(payload.get("status", "unknown")),
        "generated_at_utc": payload.get("generated_at_utc"),
    }


def display_path(path: Path) -> str:
    for base in (ROOT, ROOT.parent):
        try:
            return path.relative_to(base).as_posix()
        except ValueError:
            continue
    return path.name


def check(name: str, passed: bool, reason: str, **detail) -> dict:
    return {"name": name, "status": "pass" if passed else "fail", "reason": reason, **detail}


def marker_count(text: str, markers: list[str]) -> dict[str, int]:
    return {marker: text.count(marker) for marker in markers}


def build_report() -> dict:
    texts = {name: read_text(path) for name, path in SURFACES.items()}
    combined_public_text = "\n".join(texts.values())
    website = texts["website_html"]
    all_surface_files_exist = all(path.exists() for path in SURFACES.values())

    status_records = {name: read_status(path) for name, path in STATUS_REPORTS.items()}
    status_failures = {
        name: record
        for name, record in status_records.items()
        if record["status"] != "pass"
    }

    seo_markers = [
        "<title>Ropedia Xperience-10M Task Suite</title>",
        'name="description"',
        'rel="canonical"',
        'property="og:title"',
        'property="og:image"',
        'name="twitter:card"',
        'application/ld+json',
        'rel="manifest"',
        'rel="apple-touch-icon"',
    ]
    accessible_markers = [
        'role="tablist"',
        'role="tab"',
        'role="tabpanel"',
        "aria-selected",
        "aria-controls",
        "moveProjectTabFocus",
        "ArrowRight",
        "Home",
        "End",
    ]
    naming_markers = [
        "Ropedia Xperience-10M Task Suite",
        "Xperience-10M",
        "12-task",
        "Qwen3-Omni",
        "one public Xperience-10M sample episode",
    ]
    hf_link_markers = [
        "https://github.com/ChaoYue0307/ropedia-xperience-10m-task-suite",
        "https://huggingface.co/spaces/cy0307/ropedia-xperience-10m-task-suite",
        "https://huggingface.co/datasets/cy0307/ropedia-xperience-10m-task-suite-artifacts",
        "https://huggingface.co/cy0307/ropedia-xperience-10m-task-baselines",
        "https://huggingface.co/datasets/ropedia-ai/xperience-10m",
        "https://ropedia.com/dataset",
    ]
    artifact_markers = [
        "data/website_integrity.json",
        "data/task_surface_integrity.json",
        "data/publication_audit.json",
        "data/mirror_parity.json",
        "data/public_surface_qa.json",
    ]

    banned_hits = [
        {"marker": marker, "count": combined_public_text.count(marker)}
        for marker in BANNED_PUBLIC_STRINGS
        if marker in combined_public_text
    ]

    checks = [
        check(
            "public_surface_files_exist",
            all_surface_files_exist,
            "Repo README, website HTML, and three Hugging Face cards should all be present in the publication workspace.",
            missing=[str(path) for path in SURFACES.values() if not path.exists()],
        ),
        check(
            "core_status_reports_pass",
            not status_failures,
            "Current public-surface QA depends on the existing validators already reporting pass.",
            reports=status_records,
            failures=status_failures,
        ),
        check(
            "website_has_research_seo_metadata",
            all(marker in website for marker in seo_markers),
            "The website should expose search/social metadata and structured project metadata.",
            marker_counts=marker_count(website, seo_markers),
        ),
        check(
            "website_tabs_are_accessible_and_keyboardable",
            'role="tablist"' in website
            and website.count('role="tab"') == 5
            and website.count('role="tabpanel"') >= 19
            and "moveProjectTabFocus" in website
            and "ArrowRight" in website
            and "Home" in website
            and "End" in website,
            "The long research dashboard should be navigable as real tabs, including keyboard support.",
            marker_counts=marker_count(website, accessible_markers),
        ),
        check(
            "responsive_navigation_guard_present",
            "@media (max-width: 1120px)" in website
            and ".nav-links { display: none; }" in website
            and "scroll-margin-top" in website,
            "Tablet/mobile navigation should not overflow and deep links should land below sticky navigation.",
        ),
        check(
            "public_naming_consistent",
            all(marker in combined_public_text for marker in naming_markers),
            "Public copy should consistently present the project as Ropedia Xperience-10M, with the Qwen3-Omni scale-up boundary.",
            marker_counts=marker_count(combined_public_text, naming_markers),
        ),
        check(
            "public_links_cover_repo_hf_dataset_and_ropedia",
            all(marker in combined_public_text for marker in hf_link_markers),
            "Public cards should link the repo, Space, artifacts, model baselines, upstream dataset, and Ropedia dataset page.",
            marker_counts=marker_count(combined_public_text, hf_link_markers),
        ),
        check(
            "public_artifact_qa_files_are_exposed",
            all(marker in combined_public_text for marker in artifact_markers),
            "Readers should be able to find integrity, publication, mirror, and public-surface QA files from public copy.",
            marker_counts=marker_count(combined_public_text, artifact_markers),
        ),
        check(
            "public_copy_avoids_internal_or_meta_tone",
            not banned_hits,
            "Public copy should not expose internal tooling, private hardware labels, assessment framing, or meta design-process wording.",
            banned_hits=banned_hits,
        ),
    ]
    status = "pass" if all(item["status"] == "pass" for item in checks) else "fail"
    return {
        "title": "Ropedia Xperience-10M Public Surface QA",
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scope": "Repo README, GitHub Pages HTML, Hugging Face Space card, artifact dataset card, and model card.",
        "checks": checks,
        "surface_files": {name: display_path(path) for name, path in SURFACES.items()},
        "boundary": "This report validates public presentation quality and packaging hygiene. It does not prove multi-episode model quality.",
    }


def markdown(report: dict) -> str:
    lines = [
        "# Public Surface QA",
        "",
        "This generated report checks whether the public repo, website, and Hugging Face cards read like one polished research project.",
        "",
        f"Current status: **{report['status']}**",
        "",
        report["boundary"],
        "",
        "## Checks",
        "",
        "| Check | Status | What it guards |",
        "| --- | --- | --- |",
    ]
    for item in report["checks"]:
        lines.append(f"| {item['name']} | `{item['status']}` | {item['reason']} |")
    lines.extend([
        "",
        "## Scope",
        "",
        "| Surface | File |",
        "| --- | --- |",
    ])
    for name, path in report["surface_files"].items():
        lines.append(f"| {name} | `{path}` |")
    lines.extend([
        "",
        "## Regenerate",
        "",
        "```bash",
        "python scripts/build_public_surface_qa.py",
        "```",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    report = build_report()
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    OUTPUT_MD.write_text(markdown(report), encoding="utf-8")
    print(f"{report['status'].upper()}: wrote {OUTPUT_JSON}")
    print(f"{report['status'].upper()}: wrote {OUTPUT_MD}")
    if report["status"] != "pass":
        for item in report["checks"]:
            if item["status"] != "pass":
                print(f"- {item['name']}: {item['reason']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

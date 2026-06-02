#!/usr/bin/env python3
"""Build the public quality-gate summary.

This is a presentation artifact over the existing validators. It does not
replace the validators; it makes the release gate readable in one file and one
machine-readable JSON bundle.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_JSON = ROOT / "docs/data/quality_gates.json"
OUTPUT_MD = ROOT / "QUALITY_GATES.md"


GATES = [
    {
        "id": "scope_claims",
        "title": "Scope claims guard",
        "command": "python scripts/validate_scope_claims.py",
        "report": "docs/data/scope_claims_audit.json",
        "blocks_if": "Historical 32ep readiness/provenance strings are presented as real 32-episode metrics.",
        "proves": "The public narrative does not overclaim the Qwen3-Omni readiness artifacts.",
    },
    {
        "id": "source_alignment",
        "title": "Source alignment",
        "command": "python scripts/validate_source_alignment.py",
        "report": "docs/data/source_alignment_audit.json",
        "blocks_if": "Official full-dataset facts, sample-card facts, API-listing caveats, or public-card boundary markers are missing or inconsistent.",
        "proves": "The repo, website, and Hugging Face cards preserve the Xperience-10M source facts and current project boundary.",
    },
    {
        "id": "website_integrity",
        "title": "Website integrity",
        "command": "python scripts/validate_website_integrity.py",
        "report": "docs/data/website_integrity.json",
        "blocks_if": "Local links, anchors, JSON bundles, or referenced image assets are missing or invalid.",
        "proves": "The GitHub Pages / HF static surface is internally coherent before upload.",
    },
    {
        "id": "task_surface_integrity",
        "title": "Task surface integrity",
        "command": "python scripts/validate_task_surface.py",
        "report": "docs/data/task_surface_integrity.json",
        "blocks_if": "Task cards expose raw artifact ids, human-readable task names drift, modality thumbnails are missing, or the interactive task player is not wired to the generated JSON.",
        "proves": "The public task cards and walkthrough/player stay aligned with generated 12-task metadata.",
    },
    {
        "id": "evaluation_protocol",
        "title": "Evaluation protocol",
        "command": "python scripts/build_evaluation_protocol.py",
        "report": "docs/data/evaluation_protocol.json",
        "blocks_if": "Windowing, split policy, leakage controls, task metrics, or unsupported interpretations are not explicit.",
        "proves": "The task evaluation protocol is generated from committed metric artifacts.",
    },
    {
        "id": "figure_index",
        "title": "Figure index",
        "command": "python scripts/build_figure_index.py",
        "report": "docs/data/figure_index.json",
        "blocks_if": "Public figures, charts, or modality thumbnails are missing, unreadable, or lack source-script provenance.",
        "proves": "Public visual assets have dimensions, SHA-256 hashes, source scripts, and presentation roles.",
    },
    {
        "id": "brand_assets",
        "title": "Brand assets",
        "command": "python scripts/build_brand_assets.py",
        "report": "docs/data/brand_assets.json",
        "blocks_if": "The generated logo system, favicon, social card, or app icons are missing or not reproducibly packaged.",
        "proves": "The same project logo is available for website header, favicon, README, Hugging Face cards, and social previews.",
    },
    {
        "id": "quality_gate_manifest",
        "title": "Quality-gate manifest",
        "command": "python scripts/build_quality_gates.py",
        "report": "docs/data/quality_gates.json",
        "blocks_if": "A public reader cannot see the current packaging gates in one place.",
        "proves": "The publication checklist is explicit, versioned, and mirrored with the repo.",
    },
    {
        "id": "artifact_index",
        "title": "Artifact index",
        "command": "python scripts/build_artifact_index.py",
        "report": "docs/data/artifact_index.json",
        "blocks_if": "Project-critical evidence files are missing from the indexed proof layer.",
        "proves": "Core proof artifacts exist and stable files have SHA-256 hashes.",
    },
    {
        "id": "publication_hygiene",
        "title": "Publication hygiene",
        "command": "python scripts/validate_publication_package.py",
        "report": "docs/data/publication_audit.json",
        "blocks_if": "Raw data, caches, heavy archives, token strings, missing required assets, or stale public-card figure references enter public bundles.",
        "proves": "The repo and prepared HF bundles are clean enough to publish.",
    },
    {
        "id": "public_surface_qa",
        "title": "Public surface QA",
        "command": "python scripts/build_public_surface_qa.py",
        "report": "docs/data/public_surface_qa.json",
        "blocks_if": "Repo, website, or Hugging Face presentation loses SEO/social metadata, accessible tab semantics, source links, QA links, or public-copy hygiene.",
        "proves": "The public repo, website, and Hugging Face cards read as one polished research project surface.",
    },
    {
        "id": "mirror_parity",
        "title": "Prepared mirror parity",
        "command": "python scripts/validate_mirror_parity.py",
        "report": "docs/data/mirror_parity.json",
        "blocks_if": "Prepared HF Space, artifact dataset, or model bundle diverges from the repo for critical files.",
        "proves": "The files staged for GitHub and Hugging Face are synchronized before upload.",
    },
]

POST_PUBLISH_CHECKS = [
    {
        "id": "live_publication_verifier",
        "title": "Live publication verifier",
        "evidence": "python scripts/verify_live_publication.py",
        "required_result": "live GitHub Pages, GitHub raw, HF Space, artifact dataset, and model mirrors match the current release assets",
    },
    {
        "id": "github_pages_deploy",
        "title": "GitHub Pages deployment",
        "evidence": "gh run list --repo ChaoYue0307/ropedia-xperience-10m-task-suite --limit 5",
        "required_result": "latest pages-build-deployment run succeeds",
    },
    {
        "id": "rendered_browser_check",
        "title": "Rendered browser check",
        "evidence": "Browser/Playwright page identity, nonblank render, console health, and one local interaction",
        "required_result": "no relevant console warnings/errors and target links work",
    },
]


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
    }


def build_payload() -> dict:
    gate_records = []
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for gate in GATES:
        if gate["id"] == "quality_gate_manifest":
            status = {"exists": True, "status": "pass"}
        else:
            status = read_status(ROOT / gate["report"])
        gate_records.append({**gate, "current_report": status})
    overall_status = "pass" if all(item["current_report"]["status"] == "pass" for item in gate_records) else "fail"
    return {
        "title": "Ropedia Xperience-10M Publication Quality Gates",
        "status": overall_status,
        "generated_at_utc": generated_at,
        "rule": "Do not present a release as current unless every automated gate passes, then verify live GitHub/HF mirrors after publishing.",
        "automated_gates": gate_records,
        "post_publish_checks": POST_PUBLISH_CHECKS,
        "scope_boundary": "These gates validate public packaging, claim boundaries, mirror parity, and website integrity. They do not prove cross-episode model quality.",
    }


def markdown(payload: dict) -> str:
    lines = [
        "# Publication Quality Gates",
        "",
        "This file is the release checklist for the Ropedia Xperience-10M Task Suite.",
        "",
        f"Current gate status: **{payload['status']}**",
        "",
        payload["rule"],
        "",
        "These gates validate public packaging, claim boundaries, mirror parity, website integrity, and task-surface clarity. They do not prove cross-episode model quality; the 32-episode Qwen3-Omni pilot remains gated on data access.",
        "",
        "## Automated Gates",
        "",
        "| Gate | Command | Report | Current report status | Blocks publication if |",
        "| --- | --- | --- | --- | --- |",
    ]
    for gate in payload["automated_gates"]:
        report_status = gate["current_report"]["status"]
        lines.append(
            f"| {gate['title']} | `{gate['command']}` | `{gate['report']}` | `{report_status}` | {gate['blocks_if']} |"
        )
    lines.extend([
        "",
        "## Post-Publish Checks",
        "",
        "| Check | Evidence | Required result |",
        "| --- | --- | --- |",
    ])
    for check in payload["post_publish_checks"]:
        lines.append(f"| {check['title']} | `{check['evidence']}` | {check['required_result']} |")
    lines.extend([
        "",
        "## Rerun Order",
        "",
        "```bash",
        "python scripts/validate_scope_claims.py",
        "python scripts/validate_source_alignment.py",
        "python scripts/build_evaluation_protocol.py",
        "python scripts/build_brand_assets.py",
        "python scripts/build_figure_index.py",
        "python scripts/validate_website_integrity.py",
        "python scripts/validate_task_surface.py",
        "python scripts/build_quality_gates.py",
        "python scripts/build_artifact_index.py",
        "python scripts/validate_publication_package.py",
        "python scripts/build_public_surface_qa.py",
        "python scripts/validate_mirror_parity.py",
        "```",
        "",
        "After Hugging Face bundle sync, rerun `validate_publication_package.py` and `validate_mirror_parity.py` once more before upload.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    payload = build_payload()
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    OUTPUT_MD.write_text(markdown(payload), encoding="utf-8")
    print(f"{payload['status'].upper()}: wrote {OUTPUT_JSON}")
    print(f"{payload['status'].upper()}: wrote {OUTPUT_MD}")
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

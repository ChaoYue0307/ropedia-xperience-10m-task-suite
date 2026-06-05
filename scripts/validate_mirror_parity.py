#!/usr/bin/env python3
"""Validate parity between the repo and prepared Hugging Face mirrors.

This is a publisher-side check. It compares critical website data, figures, and
validator scripts across the local repo, prepared HF Space bundle, prepared HF
artifact dataset bundle, and prepared HF model bundle before upload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HF_ROOT = ROOT.parent / "hf_publish"
DEFAULT_OUTPUT = ROOT / "docs/data/mirror_parity.json"

DATA_FILES = [
    "additional_development_directions.json",
    "audio_ablation_summary.json",
    "artifact_index.json",
    "brand_assets.json",
    "evidence_contract.json",
    "evaluation_protocol.json",
    "figure_index.json",
    "foundation_model_plan.json",
    "live_publication_status.json",
    "modality_atlas.json",
    "omni_finetune_verified_result.json",
    "project_brief.json",
    "project_manifest.json",
    "project_packet.json",
    "project_status.json",
    "publication_audit.json",
    "public_surface_qa.json",
    "quality_gates.json",
    "rendered_site_check.json",
    "reproducibility_matrix.json",
    "research_roadmap.json",
    "research_roadmap_interactive.json",
    "research_takeaways.json",
    "research_direction_extensions.json",
    "research_directions.json",
    "scope_claims_audit.json",
    "single_episode_explorer.json",
    "source_alignment_audit.json",
    "summary_metrics.json",
    "task_surface_integrity.json",
    "task_walkthroughs.json",
    "website_integrity.json",
    "xperience10m_dataset_card_alignment.json",
]

ASSET_FILES = [
    "charts/audio_ablation_delta.svg",
    "brand/xperience10m-logo-apple-touch.png",
    "brand/xperience10m-logo-favicon-32.png",
    "brand/xperience10m-logo-favicon-64.png",
    "brand/xperience10m-logo-mark.png",
    "brand/xperience10m-logo-mark-192.png",
    "brand/xperience10m-logo-mark-512.png",
    "brand/xperience10m-logo-social-card.png",
    "task_suite_infographic.png",
    "pipeline_diagram.png",
    "task_architectures.png",
    "modalities/audio.png",
    "modalities/depth.jpg",
    "modalities/inertial.png",
    "modalities/language.png",
    "modalities/motion_capture.png",
    "modalities/pose_slam.png",
    "modalities/video.jpg",
]

SCRIPT_FILES = [
    "audio_ablation_and_raw_upgrade.py",
    "build_artifact_index.py",
    "build_brand_assets.py",
    "build_evaluation_protocol.py",
    "build_figure_index.py",
    "build_quality_gates.py",
    "build_public_surface_qa.py",
    "build_rendered_site_check.py",
    "build_interactive_research_roadmap.py",
    "build_single_episode_explorer.py",
    "build_research_takeaways.py",
    "single_episode_diagnostics.py",
    "verify_live_publication.py",
    "validate_mirror_parity.py",
    "validate_publication_package.py",
    "validate_scope_claims.py",
    "validate_source_alignment.py",
    "validate_task_surface.py",
    "validate_website_integrity.py",
    "publish_hf_bundles.py",
]

WEBSITE_FILES = [
    "apple-touch-icon.png",
    "favicon.png",
    "index.html",
    "research_roadmap.html",
    "single_episode_explorer.html",
    "site.webmanifest",
]

RESULT_FILES = [
    "audio_ablation/AUDIO_ABLATION_SUMMARY.md",
    "audio_ablation/audio_ablation_metrics.csv",
    "audio_ablation/audio_ablation_summary.json",
    "audio_ablation/audio_delta_summary.csv",
    "audio_ablation/raw_logmel_fisheye_cam0_sr16000_mels64_fft512_hop160.npz",
    "single_episode_diagnostics/provenance.json",
    "single_episode_diagnostics/README.md",
    "single_episode_diagnostics/modality_ablation/ablation_metrics.csv",
    "single_episode_diagnostics/modality_ablation/ablation_summary.json",
    "single_episode_diagnostics/object_labels/object_vocab.json",
    "single_episode_diagnostics/object_labels/window_object_labels.csv",
    "single_episode_diagnostics/timeline_overlay/timeline_overlay.csv",
    "single_episode_diagnostics/alignment_stress/alignment_shift_metrics.csv",
    "single_episode_diagnostics/alignment_stress/alignment_stress_summary.json",
]

DOC_FILES = [
    "QUALITY_GATES.md",
    "EVALUATION_PROTOCOL.md",
    "FIGURE_INDEX.md",
    "FOUNDATION_MODEL_PLAN.md",
    "ADDITIONAL_DEVELOPMENT_DIRECTIONS.md",
    "PROJECT_BRIEF.md",
    "RENDERED_SITE_CHECK.md",
    "RESEARCH_ROADMAP.md",
    "PROJECT_STATUS.md",
    "PUBLIC_SURFACE_QA.md",
    "RESEARCH_TAKEAWAYS.md",
    "SOURCE_ALIGNMENT_AUDIT.md",
    "XPERIENCE10M_DATASET_CARD_ALIGNMENT.md",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path, hf_root: Path) -> str:
    resolved = path.resolve()
    bases = [
        ("hf_space", hf_root / "space"),
        ("hf_artifacts", hf_root / "artifacts"),
        ("hf_model", hf_root / "model"),
        ("repo", ROOT),
        ("hf_publish", hf_root),
    ]
    for label, base in bases:
        try:
            return f"{label}:{resolved.relative_to(base.resolve()).as_posix()}"
        except ValueError:
            continue
    return path.name


def file_record(path: Path, hf_root: Path) -> dict:
    record = {
        "path": display_path(path, hf_root),
        "exists": path.exists(),
    }
    if path.exists() and path.is_file():
        record["bytes"] = path.stat().st_size
        record["sha256"] = sha256(path)
    else:
        record["bytes"] = 0
        record["sha256"] = None
    return record


def parity_group(name: str, local_path: Path, mirrors: dict[str, Path], hf_root: Path) -> dict:
    local = file_record(local_path, hf_root)
    mirror_records = {surface: file_record(path, hf_root) for surface, path in mirrors.items()}
    failures = []
    if not local["exists"]:
        failures.append({"surface": "repo", "kind": "missing", "path": local["path"]})
    for surface, record in mirror_records.items():
        if not record["exists"]:
            failures.append({"surface": surface, "kind": "missing", "path": record["path"]})
            continue
        if local["exists"] and record["sha256"] != local["sha256"]:
            failures.append(
                {
                    "surface": surface,
                    "kind": "hash_mismatch",
                    "path": record["path"],
                    "expected_sha256": local["sha256"],
                    "actual_sha256": record["sha256"],
                }
            )
    return {
        "name": name,
        "status": "pass" if not failures else "fail",
        "local": local,
        "mirrors": mirror_records,
        "failures": failures,
    }


def build_report(hf_root: Path) -> dict:
    groups = []

    for filename in DATA_FILES:
        groups.append(
            parity_group(
                f"data/{filename}",
                ROOT / "docs/data" / filename,
                {
                    "hf_space": hf_root / "space/data" / filename,
                    "hf_artifacts": hf_root / "artifacts/docs/data" / filename,
                    "hf_model": hf_root / "model/metrics" / filename,
                },
                hf_root,
            )
        )

    for filename in ASSET_FILES:
        groups.append(
            parity_group(
                f"assets/{filename}",
                ROOT / "docs/assets" / filename,
                {
                    "hf_space": hf_root / "space/assets" / filename,
                    "hf_artifacts_docs": hf_root / "artifacts/docs/assets" / filename,
                    "hf_artifacts_card": hf_root / "artifacts/assets" / filename,
                    "hf_model": hf_root / "model/assets" / filename,
                },
                hf_root,
            )
        )

    for filename in SCRIPT_FILES:
        groups.append(
            parity_group(
                f"scripts/{filename}",
                ROOT / "scripts" / filename,
                {
                    "hf_artifacts": hf_root / "artifacts/scripts" / filename,
                    "hf_model": hf_root / "model/scripts" / filename,
                },
                hf_root,
            )
        )

    for filename in WEBSITE_FILES:
        groups.append(
            parity_group(
                f"website/{filename}",
                ROOT / "docs" / filename,
                {
                    "hf_space": hf_root / "space" / filename,
                    "hf_artifacts_docs": hf_root / "artifacts/docs" / filename,
                },
                hf_root,
            )
        )

    for filename in RESULT_FILES:
        groups.append(
            parity_group(
                f"results/{filename}",
                ROOT / "results" / filename,
                {
                    "hf_space": hf_root / "space/results" / filename,
                    "hf_artifacts": hf_root / "artifacts/results" / filename,
                    "hf_model": hf_root / "model/results" / filename,
                },
                hf_root,
            )
        )

    for filename in DOC_FILES:
        groups.append(
            parity_group(
                f"docs/{filename}",
                ROOT / filename,
                {
                    "hf_space": hf_root / "space" / filename,
                    "hf_artifacts": hf_root / "artifacts" / filename,
                    "hf_model": hf_root / "model" / filename,
                },
                hf_root,
            )
        )

    failures = [
        {"group": group["name"], **failure}
        for group in groups
        for failure in group["failures"]
    ]
    by_surface: dict[str, int] = {}
    for failure in failures:
        by_surface[failure["surface"]] = by_surface.get(failure["surface"], 0) + 1

    return {
        "status": "pass" if not failures else "fail",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hf_root": "hf_publish",
        "summary": {
            "group_count": len(groups),
            "failure_count": len(failures),
            "failures_by_surface": by_surface,
        },
        "checks": [
            {
                "name": "repo_hf_space_artifact_model_data_parity",
                "status": "pass"
                if not any(failure["group"].startswith("data/") for failure in failures)
                else "fail",
            },
            {
                "name": "repo_hf_visual_asset_parity",
                "status": "pass"
                if not any(failure["group"].startswith("assets/") for failure in failures)
                else "fail",
            },
            {
                "name": "repo_hf_validator_script_parity",
                "status": "pass"
                if not any(failure["group"].startswith("scripts/") for failure in failures)
                else "fail",
            },
            {
                "name": "repo_hf_website_html_parity",
                "status": "pass"
                if not any(failure["group"].startswith("website/") for failure in failures)
                else "fail",
            },
            {
                "name": "repo_hf_diagnostic_result_parity",
                "status": "pass"
                if not any(failure["group"].startswith("results/") for failure in failures)
                else "fail",
            },
            {
                "name": "repo_hf_quality_doc_parity",
                "status": "pass"
                if not any(failure["group"].startswith("docs/") for failure in failures)
                else "fail",
            },
        ],
        "groups": groups,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-root", type=Path, default=DEFAULT_HF_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    report = build_report(args.hf_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{report['status'].upper()}: wrote {args.output}")
    if report["status"] != "pass":
        for failure in report["failures"][:40]:
            print(f"- {failure['group']}: {failure['surface']} {failure['kind']} {failure['path']}")
        if len(report["failures"]) > 40:
            print(f"- ... {len(report['failures']) - 40} more failures")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

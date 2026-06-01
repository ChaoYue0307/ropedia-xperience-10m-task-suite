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
    "artifact_index.json",
    "evidence_contract.json",
    "modality_atlas.json",
    "project_manifest.json",
    "publication_audit.json",
    "reproducibility_matrix.json",
    "research_direction_extensions.json",
    "research_directions.json",
    "reviewer_packet.json",
    "scope_claims_audit.json",
    "summary_metrics.json",
    "task_walkthroughs.json",
    "website_integrity.json",
]

ASSET_FILES = [
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
    "build_artifact_index.py",
    "validate_mirror_parity.py",
    "validate_publication_package.py",
    "validate_scope_claims.py",
    "validate_website_integrity.py",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict:
    record = {
        "path": str(path),
        "exists": path.exists(),
    }
    if path.exists() and path.is_file():
        record["bytes"] = path.stat().st_size
        record["sha256"] = sha256(path)
    else:
        record["bytes"] = 0
        record["sha256"] = None
    return record


def parity_group(name: str, local_path: Path, mirrors: dict[str, Path]) -> dict:
    local = file_record(local_path)
    mirror_records = {surface: file_record(path) for surface, path in mirrors.items()}
    failures = []
    if not local["exists"]:
        failures.append({"surface": "repo", "kind": "missing", "path": str(local_path)})
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
        "hf_root": str(hf_root),
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

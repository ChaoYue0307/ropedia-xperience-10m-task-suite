#!/usr/bin/env python3
"""Validate Xperience-10M source-description alignment.

This is an offline gate over committed source-alignment facts. It checks that
the repo distinguishes the gated full dataset, the public sample card, and this
project's one-episode boundary across the main repo, website, and HF cards.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HF_ROOT = ROOT.parent / "hf_publish"
OUTPUT_JSON = ROOT / "docs/data/source_alignment_audit.json"
OUTPUT_MD = ROOT / "SOURCE_ALIGNMENT_AUDIT.md"
ALIGNMENT_JSON = ROOT / "docs/data/xperience10m_dataset_card_alignment.json"

EXPECTED_FULL_DATASET = {
    "repo_id": "ropedia-ai/xperience-10m",
    "repo_sha": "ce943cf271a758b60240084892d05cf6dc12dd90",
    "last_modified": "2026-04-21T05:03:45.000Z",
    "gated": "manual",
    "license": "other",
    "task_categories": {
        "video-classification",
        "image-to-text",
        "depth-estimation",
        "robotics",
    },
    "modalities": {"3d", "audio", "video"},
}

EXPECTED_API_LISTING = {
    "sibling_count": 85258,
    "session_folder_count": 803,
    "episode_folder_count": 12103,
    "annotation_hdf5_count": 12103,
    "mp4_count": 72612,
    "visualization_rrd_count": 541,
}

EXPECTED_SAMPLE = {
    "repo_id": "ropedia-ai/xperience-10m-sample",
    "pretty_name": "Xperience-10M-Sample",
    "license": "cc-by-nc-4.0",
    "tooling": {"HOMIE Toolkit", "Rerun 0.29.0 for visualization.rrd"},
}

MODALITY_MARKERS = [
    "six RGB video streams",
    "audio",
    "stereo depth",
    "camera pose",
    "SLAM",
    "two-hand motion capture",
    "full-body motion capture",
    "inertial",
    "language",
    "metadata",
    "calibration",
]

BOUNDARY_MARKERS = [
    "full audio-visual learning",
    "caption generation",
    "depth-pixel estimation",
    "SLAM estimation",
    "neural rendering",
    "policy learning",
    "cross-episode generalization",
    "real 32-episode Qwen3-Omni model quality",
]

PRESENTATION_MARKERS = {
    "README.md": [
        "ropedia-ai/xperience-10m",
        "ropedia-ai/xperience-10m-sample",
        "SOURCE_ALIGNMENT_AUDIT.md",
        "source_alignment_audit.json",
        "cc-by-nc-4.0",
        "HOMIE Toolkit",
        "Rerun 0.29.0",
        "12,103 episode folders",
        "metadata only",
    ],
    "XPERIENCE10M_DATASET_CARD_ALIGNMENT.md": [
        "ropedia-ai/xperience-10m",
        "ropedia-ai/xperience-10m-sample",
        "cc-by-nc-4.0",
        "HOMIE Toolkit",
        "Rerun 0.29.0",
        "12,103 episode folders",
        "metadata only",
    ],
    "DATA_NOTICE.md": [
        "ropedia-ai/xperience-10m",
        "ropedia-ai/xperience-10m-sample",
        "cc-by-nc-4.0",
        "HOMIE Toolkit",
        "Rerun 0.29.0",
        "does not redistribute",
    ],
    "docs/index.html": [
        "ropedia-ai/xperience-10m",
        "xperience-10m-sample",
        "data/source_alignment_audit.json",
        "cc-by-nc-4.0",
        "HOMIE Toolkit",
        "Rerun 0.29.0",
        "12,103 episode folders",
        "not local data possession",
    ],
}

HF_PRESENTATION_MARKERS = {
    "space/README.md": [
        "xperience10m_dataset_card_alignment.json",
        "source_alignment_audit.json",
        "cc-by-nc-4.0",
        "HOMIE Toolkit",
        "Rerun 0.29.0",
        "12,103 episode folders",
        "source-listing facts only",
    ],
    "artifacts/README.md": [
        "xperience10m_dataset_card_alignment.json",
        "source_alignment_audit.json",
        "cc-by-nc-4.0",
        "HOMIE Toolkit",
        "Rerun 0.29.0",
        "12,103 episode folders",
        "metadata only",
    ],
    "artifacts/PROJECT_README.md": [
        "ropedia-ai/xperience-10m-sample",
        "SOURCE_ALIGNMENT_AUDIT.md",
        "source_alignment_audit.json",
        "cc-by-nc-4.0",
        "HOMIE Toolkit",
        "Rerun 0.29.0",
        "12,103 episode folders",
    ],
    "model/README.md": [
        "xperience10m_dataset_card_alignment.json",
        "source_alignment_audit.json",
        "cc-by-nc-4.0",
        "HOMIE",
        "Toolkit",
        "Rerun 0.29.0",
        "12,103 episode folders",
        "upstream metadata facts",
    ],
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def check(name: str, passed: bool, detail: str, evidence: list[str]) -> dict:
    return {
        "name": name,
        "status": "pass" if passed else "fail",
        "detail": detail,
        "evidence": evidence,
    }


def marker_record(base: Path, relative_path: str, markers: list[str]) -> dict:
    path = base / relative_path
    text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
    missing = [marker for marker in markers if marker not in text]
    return {
        "path": relative_path,
        "exists": path.exists(),
        "required_marker_count": len(markers),
        "missing_markers": missing,
        "status": "pass" if path.exists() and not missing else "fail",
    }


def render_markdown(payload: dict) -> str:
    alignment = payload["alignment_summary"]
    lines = [
        "# Source Alignment Audit",
        "",
        "This file is generated by `scripts/validate_source_alignment.py`. It checks",
        "that public repo, website, and HF cards preserve the same Xperience-10M",
        "source facts and boundary language.",
        "",
        f"Current status: **{payload['status']}**",
        "",
        "## Source Facts",
        "",
        "| Layer | Current value |",
        "| --- | --- |",
        f"| Full dataset repo | `{alignment['full_dataset_repo']}` |",
        f"| Full dataset access | {alignment['full_dataset_access']} |",
        f"| API episode listing | {alignment['api_episode_folders']:,} episode folders with `annotation.hdf5` as upstream metadata only |",
        f"| Public sample repo | `{alignment['sample_repo']}` |",
        f"| Public sample license | `{alignment['sample_license']}` |",
        f"| Current verified project data | {alignment['current_project_scope']} |",
        "",
        "## Checks",
        "",
        "| Check | Status | Evidence |",
        "| --- | --- | --- |",
    ]
    for item in payload["checks"]:
        evidence = ", ".join(f"`{path}`" for path in item["evidence"])
        lines.append(f"| {item['name']} | {item['status']} | {evidence} |")
    lines.extend([
        "",
        "## Boundary",
        "",
        "- HF API file counts are source-listing metadata, not local data possession.",
        "- The public sample license is preserved separately from the gated full dataset license field.",
        "- Raw MP4, HDF5, RRD, private gated data, and full Qwen weights are not redistributed.",
        "- Current model evidence remains one public sample episode, not cross-episode generalization.",
        "",
    ])
    return "\n".join(lines)


def build_report(hf_root: Path) -> dict:
    alignment = load_json(ALIGNMENT_JSON)
    checks: list[dict] = []

    metadata = alignment.get("hf_repo_metadata_observed", {})
    api_listing = metadata.get("api_file_listing_observed", {})
    sample = alignment.get("public_sample_card_observed", {})
    current = alignment.get("current_repo_alignment", {})

    checks.append(
        check(
            "full_dataset_metadata_matches_observed_snapshot",
            metadata.get("repo_id") == EXPECTED_FULL_DATASET["repo_id"]
            and metadata.get("repo_sha") == EXPECTED_FULL_DATASET["repo_sha"]
            and metadata.get("last_modified") == EXPECTED_FULL_DATASET["last_modified"]
            and metadata.get("gated") == EXPECTED_FULL_DATASET["gated"]
            and metadata.get("license") == EXPECTED_FULL_DATASET["license"]
            and set(metadata.get("task_categories", [])) == EXPECTED_FULL_DATASET["task_categories"]
            and set(metadata.get("modalities", [])) == EXPECTED_FULL_DATASET["modalities"],
            "gated full-dataset metadata matches the recorded HF API snapshot",
            ["docs/data/xperience10m_dataset_card_alignment.json"],
        )
    )
    checks.append(
        check(
            "api_listing_snapshot_is_consistent",
            all(api_listing.get(key) == value for key, value in EXPECTED_API_LISTING.items()),
            "HF API file-listing counts remain internally consistent in the committed alignment JSON",
            ["docs/data/xperience10m_dataset_card_alignment.json"],
        )
    )
    checks.append(
        check(
            "sample_card_metadata_is_preserved",
            sample.get("repo_id") == EXPECTED_SAMPLE["repo_id"]
            and sample.get("pretty_name") == EXPECTED_SAMPLE["pretty_name"]
            and sample.get("license") == EXPECTED_SAMPLE["license"]
            and set(sample.get("tooling", [])) == EXPECTED_SAMPLE["tooling"],
            "public sample card license and tooling are recorded separately from the gated full dataset",
            ["docs/data/xperience10m_dataset_card_alignment.json"],
        )
    )

    modality_text = "\n".join(alignment.get("official_modalities", []))
    missing_modalities = [marker for marker in MODALITY_MARKERS if marker not in modality_text]
    checks.append(
        check(
            "official_modality_description_is_complete",
            not missing_modalities,
            f"missing modality markers={missing_modalities}",
            ["docs/data/xperience10m_dataset_card_alignment.json"],
        )
    )

    not_claimed = set(current.get("not_yet_claimed", []))
    checks.append(
        check(
            "current_project_boundary_is_explicit",
            current.get("validated_episode_count") == 1
            and current.get("validated_frames") == 5821
            and current.get("validated_windows") == 1161
            and current.get("current_feature_dim") == 8378
            and current.get("raw_data_redistributed") is False
            and "not extracted into the current baseline feature vector" in current.get("audio_feature_status", "")
            and set(BOUNDARY_MARKERS).issubset(not_claimed),
            "one-episode scope, audio boundary, raw-data exclusion, and unsupported claims are present",
            ["docs/data/xperience10m_dataset_card_alignment.json"],
        )
    )

    repo_marker_records = [marker_record(ROOT, path, markers) for path, markers in PRESENTATION_MARKERS.items()]
    hf_marker_records = [marker_record(hf_root, path, markers) for path, markers in HF_PRESENTATION_MARKERS.items()]
    checks.append(
        check(
            "repo_public_surfaces_preserve_source_markers",
            all(item["status"] == "pass" for item in repo_marker_records),
            "README, data notice, alignment doc, and website expose official/full/sample/source-boundary markers",
            [item["path"] for item in repo_marker_records],
        )
    )
    checks.append(
        check(
            "hf_public_cards_preserve_source_markers",
            all(item["status"] == "pass" for item in hf_marker_records),
            "HF Space, artifact dataset, model card, and mirrored project README expose source-boundary markers",
            [item["path"] for item in hf_marker_records],
        )
    )

    failures = [item for item in checks if item["status"] != "pass"]
    payload = {
        "title": "Ropedia Xperience-10M Source Alignment Audit",
        "status": "pass" if not failures else "fail",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "alignment_json": "docs/data/xperience10m_dataset_card_alignment.json",
        "alignment_summary": {
            "full_dataset_repo": metadata.get("repo_id"),
            "full_dataset_access": metadata.get("gated"),
            "api_episode_folders": api_listing.get("episode_folder_count"),
            "sample_repo": sample.get("repo_id"),
            "sample_license": sample.get("license"),
            "current_project_scope": "1 public sample episode, 5,821 frames, 1,161 windows, 8,378 current features",
        },
        "checks": checks,
        "repo_marker_records": repo_marker_records,
        "hf_marker_records": hf_marker_records,
        "failures": failures,
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-root", type=Path, default=DEFAULT_HF_ROOT)
    parser.add_argument("--output-json", type=Path, default=OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=OUTPUT_MD)
    args = parser.parse_args()

    payload = build_report(args.hf_root.resolve())
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(payload), encoding="utf-8")
    print(f"{payload['status'].upper()}: wrote {args.output_json}")
    print(f"{payload['status'].upper()}: wrote {args.output_md}")
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

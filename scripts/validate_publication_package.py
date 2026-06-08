#!/usr/bin/env python3
"""Validate public package contents for the repo and HF bundles.

This check scans the GitHub repo plus the prepared Hugging Face
Space/artifact/model folders for generated Python caches, raw Xperience-10M
data, heavyweight checkpoint formats that do not belong in this public package,
and accidental credential text.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HF_ROOT = ROOT.parent / "hf_publish"

BANNED_DIR_NAMES = {"__pycache__"}
BANNED_FILE_NAMES = {".DS_Store"}
BANNED_SUFFIXES = {".pyc", ".pyo"}
RAW_DATA_SUFFIXES = {".mp4", ".hdf5", ".h5", ".rrd"}
HEAVY_MODEL_SUFFIXES = {".safetensors", ".bin", ".tar"}
TEXT_SUFFIXES = {
    "",
    ".cff",
    ".csv",
    ".html",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".svg",
    ".txt",
    ".webmanifest",
    ".xml",
    ".yaml",
    ".yml",
}
TOKEN_PATTERN = re.compile(r"hf_[A-Za-z0-9]{20,}")
STALE_PRESENTATION_STRINGS = {
    "xperience10m-" + "modalities-v9-large-atlas": "old task-suite infographic cache key",
    "xperience10m-" + "taskfirst-v10": "older task-suite infographic cache key",
    "Start with the large native " + "modality atlas": "old suite-section hierarchy copy",
    "ChatGPT" + "-image": "internal image-generation tool wording in public copy",
    "H" + "20": "private compute infrastructure wording in public copy",
    "A" + "100": "private compute infrastructure wording in public copy",
    "Cur" + "sor": "editor/work-session wording in public copy",
    "public " + "dashboard and generated figures " + "deliberately " + "follow": "meta design-process wording in public copy",
}
LOCAL_PATH_PATTERNS = {
    "/" + "Users/": "local macOS user path in public text",
    "/" + "private/": "temporary local path in public text",
}
CARD_FRESHNESS_EXPECTATIONS = [
    {
        "surface": "github_repo",
        "relative_path": "README.md",
        "required": [
            "XPERIENCE10M_DATASET_CARD_ALIGNMENT.md",
            "EVALUATION_PROTOCOL.md",
            "PROJECT_STATUS.md",
            "RESEARCH_TAKEAWAYS.md",
            "xperience10m-logo-social-card.png",
            "Dataset Context",
            "Qwen3-Omni",
            "Cosmos 3",
            "12 human-readable tasks",
            "interactive scrub/play walkthrough storyboard",
        ],
    },
    {
        "surface": "hf_space_bundle",
        "relative_path": "README.md",
        "required": [
            "XPERIENCE10M_DATASET_CARD_ALIGNMENT.md",
            "EVALUATION_PROTOCOL.md",
            "PROJECT_STATUS.md",
            "RESEARCH_TAKEAWAYS.md",
            "xperience10m-logo-social-card.png",
            "Dataset Context",
            "Qwen3-Omni",
            "Cosmos 3",
            "12 human-readable tasks",
            "interactive scrub/play walkthrough storyboard",
        ],
    },
    {
        "surface": "hf_artifact_bundle",
        "relative_path": "README.md",
        "required": [
            "What To Open First",
            "Dataset Boundary",
            "Related Hub Repositories",
            "xperience10m-logo-social-card.png",
            "derived artifacts",
            "Raw Xperience-10M videos",
            "Qwen3-Omni",
        ],
    },
    {
        "surface": "hf_artifact_bundle",
        "relative_path": "PROJECT_README.md",
        "required": [
            "XPERIENCE10M_DATASET_CARD_ALIGNMENT.md",
            "EVALUATION_PROTOCOL.md",
            "PROJECT_STATUS.md",
            "RESEARCH_TAKEAWAYS.md",
            "xperience10m-logo-social-card.png",
            "Dataset Context",
            "Qwen3-Omni",
            "Cosmos 3",
            "12 human-readable tasks",
            "interactive scrub/play walkthrough storyboard",
        ],
    },
    {
        "surface": "hf_model_bundle",
        "relative_path": "README.md",
        "required": [
            "XPERIENCE10M_DATASET_CARD_ALIGNMENT.md",
            "EVALUATION_PROTOCOL.md",
            "PROJECT_STATUS.md",
            "RESEARCH_TAKEAWAYS.md",
            "xperience10m-logo-social-card.png",
            "Dataset Context",
            "Qwen3-Omni",
            "Cosmos 3",
            "12 human-readable tasks",
            "interactive scrub/play walkthrough storyboard",
        ],
    },
]


def rel(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


def git_public_paths(root: Path) -> list[Path] | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return [root / line for line in result.stdout.splitlines() if line.strip()]


def iter_public_files(root: Path, paths: list[Path] | None = None):
    if paths is not None:
        for path in paths:
            if path.exists():
                yield path
        return
    if not root.exists():
        return
    for path in root.rglob("*"):
        parts = set(path.parts)
        if ".git" in parts or ".venv" in parts or "venv" in parts:
            continue
        yield path


def scan(root: Path, *, paths: list[Path] | None = None, display_root: str | None = None) -> dict:
    violations: list[dict] = []
    text_files = 0
    total_files = 0
    largest_file = {"path": None, "bytes": 0}

    for path in iter_public_files(root, paths):
        path_rel = rel(path, root)
        if path.is_dir():
            if path.name in BANNED_DIR_NAMES:
                violations.append({"kind": "generated_cache_dir", "path": path_rel})
            continue

        total_files += 1
        size = path.stat().st_size
        if size > largest_file["bytes"]:
            largest_file = {"path": path_rel, "bytes": size}

        suffix = path.suffix.lower()
        if path.name in BANNED_FILE_NAMES or suffix in BANNED_SUFFIXES:
            violations.append({"kind": "generated_cache_file", "path": path_rel})
        if suffix in RAW_DATA_SUFFIXES:
            violations.append({"kind": "raw_xperience10m_data", "path": path_rel})
        allowed_model_weight = (
            display_root == "hf_publish/model"
            and path_rel == "pytorch_model.bin"
            and size < 150 * 1024 * 1024
        )
        if suffix in HEAVY_MODEL_SUFFIXES and not allowed_model_weight:
            violations.append({"kind": "heavy_model_or_archive", "path": path_rel})

        if suffix in TEXT_SUFFIXES:
            text_files += 1
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if TOKEN_PATTERN.search(text):
                violations.append({"kind": "possible_hf_token", "path": path_rel})
            for needle, reason in LOCAL_PATH_PATTERNS.items():
                if needle in text:
                    violations.append({
                        "kind": "local_filesystem_path",
                        "path": path_rel,
                        "detail": reason,
                    })
            for needle, reason in STALE_PRESENTATION_STRINGS.items():
                if path_rel == ".mailmap":
                    continue
                if needle in text:
                    violations.append({
                        "kind": "stale_presentation_copy",
                        "path": path_rel,
                        "detail": reason,
                    })

    return {
        "root": display_root or rel(root, ROOT.parent),
        "exists": root.exists(),
        "file_count": total_files,
        "text_file_count": text_files,
        "largest_file": largest_file,
        "violations": violations,
    }


def required_assets(root: Path) -> dict[str, bool]:
    required = [
        "README.md",
        "CITATION.cff",
        "LICENSE",
        "codemeta.json",
        "ARTIFACT_GUIDE.md",
        "PROJECT_STATUS.md",
        "RESEARCH_ROADMAP.md",
        "RESEARCH_TAKEAWAYS.md",
        "TASK_SUITE_ENHANCEMENT_128.md",
        "QUALITY_GATES.md",
        "PUBLIC_SURFACE_QA.md",
        "RENDERED_SITE_CHECK.md",
        "EVALUATION_PROTOCOL.md",
        "FIGURE_INDEX.md",
        "SOURCE_ALIGNMENT_AUDIT.md",
        "XPERIENCE10M_DATASET_CARD_ALIGNMENT.md",
        "REPRODUCIBILITY.md",
        "EVIDENCE_CONTRACT.md",
        "DATA_NOTICE.md",
        "docs/404.html",
        "docs/apple-touch-icon.png",
        "docs/favicon.svg",
        "docs/favicon.png",
        "docs/index.html",
        "docs/research_roadmap.html",
        "docs/robots.txt",
        "docs/site.webmanifest",
        "docs/sitemap.xml",
        "docs/data/brand_assets.json",
        "docs/data/evidence_contract.json",
        "docs/data/evaluation_protocol.json",
        "docs/data/figure_index.json",
        "docs/data/source_alignment_audit.json",
        "docs/data/artifact_index.json",
        "docs/data/live_publication_status.json",
        "docs/data/quality_gates.json",
        "docs/data/project_manifest.json",
        "docs/data/project_packet.json",
        "docs/data/project_status.json",
        "docs/data/research_roadmap.json",
        "docs/data/research_roadmap_interactive.json",
        "docs/data/research_takeaways.json",
        "docs/data/xperience10m_dataset_card_alignment.json",
        "docs/data/reproducibility_matrix.json",
        "docs/data/modality_atlas.json",
        "docs/data/mirror_parity.json",
        "docs/data/public_surface_qa.json",
        "docs/data/rendered_site_check.json",
        "docs/data/scope_claims_audit.json",
        "docs/data/task_surface_integrity.json",
        "docs/data/website_integrity.json",
        "docs/data/summary_metrics.json",
        "docs/data/task_suite_enhancement_128.json",
        "docs/assets/modalities/video.jpg",
        "docs/assets/modalities/audio.png",
        "docs/assets/modalities/depth.jpg",
        "docs/assets/modalities/pose_slam.png",
        "docs/assets/modalities/motion_capture.png",
        "docs/assets/modalities/inertial.png",
        "docs/assets/modalities/language.png",
        "docs/assets/brand/xperience10m-logo-apple-touch.png",
        "docs/assets/brand/xperience10m-logo-favicon-32.png",
        "docs/assets/brand/xperience10m-logo-favicon-64.png",
        "docs/assets/brand/xperience10m-logo-mark.png",
        "docs/assets/brand/xperience10m-logo-mark-192.png",
        "docs/assets/brand/xperience10m-logo-mark-512.png",
        "docs/assets/brand/xperience10m-logo-social-card.png",
        "docs/assets/task_suite_infographic.png",
        "docs/assets/pipeline_diagram.png",
        "docs/assets/task_architectures.png",
        "results/episode_task_suite/summary_report.json",
        "results/episode_task_suite/feature_manifest.json",
        "results/episode_task_suite/neural_mlp/timeline_action/metrics.json",
        "results/omni_finetune/DATA_ACCESS_STATUS.md",
        "results/omni_finetune/MULTI_EPISODE_ACCESS_STATUS.md",
        "scripts/episode_task_suite.py",
        "scripts/neural_task_models.py",
        "scripts/build_artifact_index.py",
        "scripts/build_brand_assets.py",
        "scripts/build_evaluation_protocol.py",
        "scripts/build_figure_index.py",
        "scripts/build_quality_gates.py",
        "scripts/build_public_surface_qa.py",
        "scripts/build_rendered_site_check.py",
        "scripts/build_interactive_research_roadmap.py",
        "scripts/verify_live_publication.py",
        "scripts/validate_mirror_parity.py",
        "scripts/validate_scope_claims.py",
        "scripts/validate_source_alignment.py",
        "scripts/validate_task_surface.py",
        "scripts/validate_website_integrity.py",
        "scripts/publish_hf_bundles.py",
        "scripts/omni/build_task_suite_enhancement_128.py",
        "scripts/omni/train_qwen3_omni_lora.py",
        "results/omni_finetune/task_suite_enhancement_128_v1_20260608/enhancement_plan.json",
        "results/omni_finetune/task_suite_enhancement_128_v1_20260608/ENHANCEMENT_REPORT.md",
    ]
    return {item: (root / item).exists() for item in required}


def public_card_freshness(roots: dict[str, Path]) -> list[dict]:
    records = []
    for item in CARD_FRESHNESS_EXPECTATIONS:
        surface = item["surface"]
        path = roots[surface] / item["relative_path"]
        text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
        missing = [marker for marker in item["required"] if marker not in text]
        records.append({
            "surface": surface,
            "path": item["relative_path"],
            "exists": path.exists(),
            "required_marker_count": len(item["required"]),
            "missing_markers": missing,
            "status": "pass" if path.exists() and not missing else "fail",
        })
    return records


def build_report(hf_root: Path) -> dict:
    roots = {
        "github_repo": ROOT,
        "hf_space_bundle": hf_root / "space",
        "hf_artifact_bundle": hf_root / "artifacts",
        "hf_model_bundle": hf_root / "model",
    }
    root_labels = {
        "github_repo": "repo",
        "hf_space_bundle": "hf_publish/space",
        "hf_artifact_bundle": "hf_publish/artifacts",
        "hf_model_bundle": "hf_publish/model",
    }
    scans = {}
    for name, path in roots.items():
        public_paths = git_public_paths(path) if name == "github_repo" else None
        scans[name] = scan(path, paths=public_paths, display_root=root_labels[name])
    assets = required_assets(ROOT)
    card_freshness = public_card_freshness(roots)
    missing_assets = [path for path, present in assets.items() if not present]
    violations = [
        {"root": name, **violation}
        for name, result in scans.items()
        for violation in result["violations"]
    ]
    checks = [
        {
            "name": "required_publication_assets_present",
            "status": "pass" if not missing_assets else "fail",
            "missing": missing_assets,
        },
        {
            "name": "no_generated_python_caches",
            "status": "pass"
            if not any(v["kind"].startswith("generated_cache") for v in violations)
            else "fail",
            "count": sum(1 for v in violations if v["kind"].startswith("generated_cache")),
        },
        {
            "name": "no_raw_xperience10m_data",
            "status": "pass" if not any(v["kind"] == "raw_xperience10m_data" for v in violations) else "fail",
            "count": sum(1 for v in violations if v["kind"] == "raw_xperience10m_data"),
        },
        {
            "name": "no_heavy_model_archives",
            "status": "pass" if not any(v["kind"] == "heavy_model_or_archive" for v in violations) else "fail",
            "count": sum(1 for v in violations if v["kind"] == "heavy_model_or_archive"),
        },
        {
            "name": "no_hf_tokens_in_public_text",
            "status": "pass" if not any(v["kind"] == "possible_hf_token" for v in violations) else "fail",
            "count": sum(1 for v in violations if v["kind"] == "possible_hf_token"),
        },
        {
            "name": "no_local_filesystem_paths_in_public_text",
            "status": "pass" if not any(v["kind"] == "local_filesystem_path" for v in violations) else "fail",
            "count": sum(1 for v in violations if v["kind"] == "local_filesystem_path"),
        },
        {
            "name": "no_stale_task_suite_presentation_copy",
            "status": "pass" if not any(v["kind"] == "stale_presentation_copy" for v in violations) else "fail",
            "count": sum(1 for v in violations if v["kind"] == "stale_presentation_copy"),
        },
        {
            "name": "public_cards_reference_taskfirst_figure",
            "status": "pass" if all(item["status"] == "pass" for item in card_freshness) else "fail",
            "failures": [item for item in card_freshness if item["status"] != "pass"],
        },
    ]
    status = "pass" if all(check["status"] == "pass" for check in checks) else "fail"
    return {
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checks": checks,
        "required_assets": assets,
        "public_card_freshness": card_freshness,
        "scans": scans,
        "violations": violations,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-root", type=Path, default=DEFAULT_HF_ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/data/publication_audit.json")
    args = parser.parse_args()

    report = build_report(args.hf_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{report['status'].upper()}: wrote {args.output}")
    if report["status"] != "pass":
        for violation in report["violations"][:40]:
            print(f"- {violation['root']}: {violation['kind']} {violation['path']}")
        if len(report["violations"]) > 40:
            print(f"- ... {len(report['violations']) - 40} more violations")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

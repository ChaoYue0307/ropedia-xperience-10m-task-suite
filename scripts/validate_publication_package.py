#!/usr/bin/env python3
"""Validate public publication hygiene for the repo and HF bundles.

This check is intentionally conservative: it scans the GitHub repo plus the
prepared Hugging Face Space/artifact/model folders for generated Python caches,
raw Xperience-10M data, heavyweight checkpoint formats that should not be
published here, and accidental Hugging Face token strings.
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
    ".xml",
    ".yaml",
    ".yml",
}
TOKEN_PATTERN = re.compile(r"hf_[A-Za-z0-9]{20,}")


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


def scan(root: Path, *, paths: list[Path] | None = None) -> dict:
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
        if suffix in HEAVY_MODEL_SUFFIXES:
            violations.append({"kind": "heavy_model_or_archive", "path": path_rel})

        if suffix in TEXT_SUFFIXES:
            text_files += 1
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if TOKEN_PATTERN.search(text):
                violations.append({"kind": "possible_hf_token", "path": path_rel})

    return {
        "root": str(root),
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
        "REPRODUCIBILITY.md",
        "EVIDENCE_CONTRACT.md",
        "DATA_NOTICE.md",
        "docs/404.html",
        "docs/favicon.svg",
        "docs/index.html",
        "docs/robots.txt",
        "docs/sitemap.xml",
        "docs/data/evidence_contract.json",
        "docs/data/artifact_index.json",
        "docs/data/project_manifest.json",
        "docs/data/reviewer_packet.json",
        "docs/data/reproducibility_matrix.json",
        "docs/data/modality_atlas.json",
        "docs/data/mirror_parity.json",
        "docs/data/scope_claims_audit.json",
        "docs/data/website_integrity.json",
        "docs/data/summary_metrics.json",
        "docs/assets/modalities/video.jpg",
        "docs/assets/modalities/audio.png",
        "docs/assets/modalities/depth.jpg",
        "docs/assets/modalities/pose_slam.png",
        "docs/assets/modalities/motion_capture.png",
        "docs/assets/modalities/inertial.png",
        "docs/assets/modalities/language.png",
        "docs/assets/task_suite_infographic.png",
        "docs/assets/pipeline_diagram.png",
        "docs/assets/task_architectures.png",
        "results/episode_task_suite/summary_report.json",
        "results/episode_task_suite/feature_manifest.json",
        "results/episode_task_suite/neural_mlp/timeline_action/metrics.json",
        "results/omni_finetune/DATA_BLOCKER_REPORT.md",
        "results/omni_finetune/A100_HF_RELAY_STATUS.md",
        "scripts/episode_task_suite.py",
        "scripts/neural_task_models.py",
        "scripts/build_artifact_index.py",
        "scripts/validate_mirror_parity.py",
        "scripts/validate_scope_claims.py",
        "scripts/validate_website_integrity.py",
        "scripts/omni/train_qwen3_omni_lora.py",
    ]
    return {item: (root / item).exists() for item in required}


def build_report(hf_root: Path) -> dict:
    roots = {
        "github_repo": ROOT,
        "hf_space_bundle": hf_root / "space",
        "hf_artifact_bundle": hf_root / "artifacts",
        "hf_model_bundle": hf_root / "model",
    }
    scans = {}
    for name, path in roots.items():
        public_paths = git_public_paths(path) if name == "github_repo" else None
        scans[name] = scan(path, paths=public_paths)
    assets = required_assets(ROOT)
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
    ]
    status = "pass" if all(check["status"] == "pass" for check in checks) else "fail"
    return {
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checks": checks,
        "required_assets": assets,
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

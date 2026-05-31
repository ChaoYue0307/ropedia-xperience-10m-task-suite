#!/usr/bin/env python3
"""Build a compact source-of-truth artifact index for reviewers.

The index is intentionally selective. It lists the files that prove the public
claims, not every prediction array or checkpoint in the repository.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/data/artifact_index.json"

ARTIFACTS = [
    {
        "id": "evidence_contract",
        "title": "Evidence contract",
        "path": "EVIDENCE_CONTRACT.md",
        "kind": "claim_boundary",
        "surface": "repo",
        "proves": "Defines what is verified, what is smoke-only, and what must not be inferred.",
    },
    {
        "id": "reviewer_packet",
        "title": "Reviewer packet",
        "path": "docs/data/reviewer_packet.json",
        "kind": "review_path",
        "surface": "website_hf",
        "proves": "Gives a short audit path with scope status and public surfaces.",
    },
    {
        "id": "artifact_index_builder",
        "title": "Artifact index builder",
        "path": "scripts/build_artifact_index.py",
        "kind": "review_path",
        "surface": "repo_hf",
        "proves": "Generates the selective proof-artifact catalog from local files.",
    },
    {
        "id": "publication_audit",
        "title": "Publication audit",
        "path": "docs/data/publication_audit.json",
        "kind": "hygiene_report",
        "surface": "website_hf",
        "volatile": True,
        "proves": "Confirms public bundles pass raw-data, cache, archive, and token-string checks.",
    },
    {
        "id": "project_manifest",
        "title": "Project manifest",
        "path": "docs/data/project_manifest.json",
        "kind": "metadata",
        "surface": "website_hf",
        "proves": "Lists public URLs, upstream sources, and machine-readable project metadata.",
    },
    {
        "id": "task_summary",
        "title": "12-task summary report",
        "path": "results/episode_task_suite/summary_report.json",
        "kind": "metrics_source",
        "surface": "repo_hf",
        "proves": "Stores the task definitions, splits, feature dimension, and minimal/neural metrics.",
    },
    {
        "id": "website_metrics_bundle",
        "title": "Website metrics bundle",
        "path": "docs/data/summary_metrics.json",
        "kind": "website_data",
        "surface": "website_hf",
        "proves": "Mirrors task metrics for the static dashboard.",
    },
    {
        "id": "feature_manifest",
        "title": "Feature manifest",
        "path": "results/episode_task_suite/feature_manifest.json",
        "kind": "data_contract",
        "surface": "repo_hf",
        "proves": "Maps the 8,378-dimensional window vector back to source feature blocks.",
    },
    {
        "id": "available_modalities",
        "title": "Available modalities",
        "path": "results/episode_task_suite/available_modalities.json",
        "kind": "data_contract",
        "surface": "repo_hf",
        "proves": "Documents which sample modalities entered the current extracted feature contract.",
    },
    {
        "id": "windows_table",
        "title": "Aligned windows table",
        "path": "results/episode_task_suite/windows.csv",
        "kind": "data_contract",
        "surface": "repo_hf",
        "proves": "Lists the 1,161 aligned windows and their frame/action/subtask labels.",
    },
    {
        "id": "neural_mlp_directory",
        "title": "Neural MLP task-head results",
        "path": "results/episode_task_suite/neural_mlp",
        "kind": "result_directory",
        "surface": "repo_hf_model",
        "proves": "Stores matching PyTorch MLP results for the 12 task contracts.",
    },
    {
        "id": "research_direction_taxonomy",
        "title": "Research direction taxonomy",
        "path": "results/episode_task_suite/research_directions/research_direction_taxonomy.json",
        "kind": "taxonomy",
        "surface": "repo_hf",
        "proves": "Maps the 12 tasks to the four Ropedia research directions as direct/proxy/diagnostic.",
    },
    {
        "id": "research_direction_extensions",
        "title": "Research direction extension probes",
        "path": "results/episode_task_suite/research_direction_extensions/research_direction_extension_results.json",
        "kind": "metrics_source",
        "surface": "repo_hf",
        "proves": "Stores one coded extension probe per research direction with minimal and neural metrics.",
    },
    {
        "id": "task_walkthroughs",
        "title": "Task walkthroughs",
        "path": "results/episode_task_suite/task_walkthroughs/TASK_WALKTHROUGHS.md",
        "kind": "onboarding_doc",
        "surface": "repo_hf",
        "proves": "Explains every task with case study, input, process modules, output, and limitation.",
    },
    {
        "id": "task_suite_infographic",
        "title": "12-task suite infographic",
        "path": "docs/assets/task_suite_infographic.png",
        "kind": "generated_figure",
        "surface": "website_hf",
        "proves": "Presents the task suite and sample modality thumbnails with metrics generated from committed files.",
    },
    {
        "id": "pipeline_figure",
        "title": "Pipeline figure",
        "path": "docs/assets/pipeline_diagram.png",
        "kind": "generated_figure",
        "surface": "website_hf",
        "proves": "Shows the raw-episode to artifact pipeline with verified labels.",
    },
    {
        "id": "architecture_figure",
        "title": "Architecture figure",
        "path": "docs/assets/task_architectures.png",
        "kind": "generated_figure",
        "surface": "website_hf",
        "proves": "Shows the shared feature pipeline and minimal/neural head families.",
    },
    {
        "id": "qwen_data_blocker",
        "title": "Qwen3-Omni data blocker report",
        "path": "results/omni_finetune/DATA_BLOCKER_REPORT.md",
        "kind": "blocker_report",
        "surface": "repo_hf",
        "proves": "Documents why no 32-episode Qwen3-Omni result is claimed yet.",
    },
    {
        "id": "a100_relay_status",
        "title": "A100 relay status",
        "path": "results/omni_finetune/A100_HF_RELAY_STATUS.md",
        "kind": "scaleup_status",
        "surface": "repo_hf",
        "proves": "Documents the pending A100-to-H20 data relay and 32-session pilot selection.",
    },
    {
        "id": "citation",
        "title": "Citation metadata",
        "path": "CITATION.cff",
        "kind": "citation",
        "surface": "repo_hf",
        "proves": "Makes the project externally citable.",
    },
    {
        "id": "license",
        "title": "License and data terms",
        "path": "LICENSE",
        "kind": "license",
        "surface": "repo_hf",
        "proves": "Separates MIT-scoped code from original Xperience-10M data terms.",
    },
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_stats(path: Path) -> dict:
    files = [item for item in path.rglob("*") if item.is_file()]
    return {
        "file_count": len(files),
        "bytes": sum(item.stat().st_size for item in files),
    }


def artifact_entry(item: dict) -> dict:
    path = ROOT / item["path"]
    entry = {
        **item,
        "exists": path.exists(),
    }
    if path.is_file():
        entry["bytes"] = path.stat().st_size
        if item.get("volatile"):
            entry["hash_policy"] = "existence_and_size_only"
        else:
            entry["sha256"] = sha256(path)
    elif path.is_dir():
        entry.update(directory_stats(path))
    else:
        entry.update({"bytes": 0})
    return entry


def main() -> int:
    entries = [artifact_entry(item) for item in ARTIFACTS]
    missing = [entry["path"] for entry in entries if not entry["exists"]]
    by_kind: dict[str, int] = {}
    for entry in entries:
        by_kind[entry["kind"]] = by_kind.get(entry["kind"], 0) + 1

    report = {
        "title": "Ropedia Xperience-10M Task Suite Artifact Index",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "pass" if not missing else "fail",
        "artifact_count": len(entries),
        "missing": missing,
        "by_kind": by_kind,
        "artifacts": entries,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{report['status'].upper()}: wrote {OUTPUT}")
    if missing:
        for path in missing:
            print(f"- missing: {path}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

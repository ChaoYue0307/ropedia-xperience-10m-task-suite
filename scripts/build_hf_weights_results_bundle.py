#!/usr/bin/env python3
"""Build the consolidated Hugging Face weights/results bundle.

This stages a public-safe model repository under ``../hf_publish/weights_results``.
It deliberately includes adapter weights and baseline-task weights, but not raw
Xperience-10M files, base model checkpoints, or generated Hugging Face cache
state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HF_ROOT = ROOT.parent / "hf_publish"
DEFAULT_OUTPUT = DEFAULT_HF_ROOT / "weights_results"
DEFAULT_REPO_ID = "cy0307/ropedia-xperience-10m-weights-results"

QWEN_LEGACY_DIR = ROOT / "results/omni_finetune/hf_upload"
QWEN_V6_DIR = ROOT / "results/omni_finetune/hf_upload_qwen3_128ep_v6_rank64"
COSMOS3_SUPER_DIR = (
    ROOT
    / "results/omni_finetune"
    / "xperience10m_cosmos3_super_forward_dynamics_lora_128ep_train1epoch_256_attn_full8gpu_20260608"
    / "adapter_lora"
)

ANALYSIS_FILES = [
    "PROJECT_BRIEF.md",
    "PROJECT_STATUS.md",
    "PUBLIC_READER_MAP.md",
    "QUALITY_GATES.md",
    "PUBLIC_SURFACE_QA.md",
    "SOURCE_ALIGNMENT_AUDIT.md",
    "TWO_EVIDENCE_LINES.md",
    "TWO_EVIDENCE_LINE_RESULT_SUMMARY.md",
    "QWEN3_OMNI_RUN_LINEAGE.md",
    "TASK_METHOD_20_RESULT_MATRIX.md",
    "TASK_METHOD_20_GAP_AUDIT.md",
    "TASK_METHOD_20_SOURCE_AUDIT.md",
    "TASK_SUITE_20.md",
    "XPERIENCE10M_128_EPISODE_FEATURE_INDEX.md",
    "THREE_FOUNDATION_PIPELINES.md",
    "docs/data/task_method_20_result_matrix.json",
    "docs/data/task_method_20_gap_audit.json",
    "docs/data/task_method_20_source_audit.json",
    "docs/data/omni_model_comparison.json",
    "docs/data/unified_task_model_radar.json",
    "docs/data/episode128_task_model_radar.json",
    "docs/data/single_episode_task_model_radar.json",
    "docs/data/project_status.json",
    "docs/data/mirror_parity.json",
    "docs/data/publication_audit.json",
    "docs/data/quality_gates.json",
    "docs/data/public_surface_qa.json",
    "docs/data/scope_claims_audit.json",
    "docs/data/source_alignment_audit.json",
    "docs/data/task_surface_integrity.json",
    "docs/data/two_evidence_lines.json",
    "docs/data/two_evidence_line_result_summary.json",
    "docs/data/qwen3_omni_run_lineage.json",
    "docs/data/website_integrity.json",
    "docs/data/xperience10m_128_episode_feature_index.json",
    "docs/assets/charts/two_evidence_line_map.svg",
]

VERIFIED_RESULT_GLOBS = [
    "results/omni_finetune/verified_public/**/verified_result_summary.json",
    "results/omni_finetune/verified_public/**/package_audit.json",
    "results/omni_finetune/verified_public/**/metrics.json",
    "results/omni_finetune/verified_public/**/RUN_REPORT.md",
    "results/omni_finetune/model_output_task_probes_20260616/**/metrics.json",
    "results/omni_finetune/model_output_task_probes_20260616/summary.json",
    "results/omni_finetune/xperience10m_qwen3_omni_v6_*/*/metrics.json",
    "results/omni_finetune/xperience10m_qwen3_omni_v6_*/summary.json",
    "results/omni_finetune/xperience10m_qwen3_omni_v6_*/RUN_REPORT.md",
    "results/omni_finetune/xperience10m_cosmos3_super_*/*/metrics.json",
    "results/omni_finetune/xperience10m_cosmos3_super_*/summary.json",
    "results/omni_finetune/xperience10m_cosmos3_super_*/RUN_REPORT.md",
    "results/omni_finetune/xperience10m_cosmos3_nano_*/*/metrics.json",
    "results/omni_finetune/xperience10m_cosmos3_nano_*/summary.json",
    "results/omni_finetune/xperience10m_cosmos3_nano_*/RUN_REPORT.md",
]

IGNORE_NAMES = {
    ".DS_Store",
    "__pycache__",
    ".git",
    ".cache",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-root", type=Path, default=DEFAULT_HF_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    return parser.parse_args()


def ignore_generated(dir_path: str, names: list[str]) -> set[str]:
    ignored = {name for name in names if name in IGNORE_NAMES}
    ignored.update(name for name in names if name.endswith((".log", ".pid", ".pyc")))
    return ignored


def copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    shutil.copytree(src, dst, ignore=ignore_generated, dirs_exist_ok=True)


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_files(root: Path) -> list[dict]:
    rows = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if "/.cache/" in path.as_posix():
            continue
        rel = path.relative_to(root).as_posix()
        rows.append(
            {
                "path": rel,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return rows


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_readme(path: Path, repo_id: str, manifest: dict) -> None:
    scored = manifest["result_matrix"]["scored_method_task_count"]
    total = manifest["result_matrix"]["method_task_record_count"]
    text = f"""---
license: mit
library_name: pytorch
tags:
  - embodied-ai
  - robotics
  - multimodal
  - xperience-10m
  - evaluation
  - lora
  - qwen3-omni
  - cosmos
datasets:
  - ropedia-ai/xperience-10m-sample
  - ropedia-ai/xperience-10m
metrics:
  - accuracy
  - f1
  - precision
  - recall
---

# Ropedia Xperience-10M Weights, Results, and Analysis

This repository is the consolidated public-safe weight and result package for
the Ropedia Xperience-10M task suite. It is intended to make the current model
artifacts traceable from one Hugging Face location while preserving the
canonical per-adapter repositories.

## What Is Included

- `baselines_and_analysis_snapshot/`: the published baseline-model snapshot,
  including minimal numpy weights, neural MLP checkpoints, task metrics, public
  docs, result JSON, and analysis cards from `cy0307/ropedia-xperience-10m-task-baselines`.
- `weights/qwen3_omni_legacy_lora/`: the earlier Qwen3-Omni LoRA adapter copy
  retained for provenance because it is byte-identical between the old
  `adapter_lora` and `hf_upload` local folders.
- `weights/qwen3_omni_v6_rank64_lora/`: the current verified Qwen3-Omni v6
  rank64 LoRA adapter for the selected 128-episode diagnostic branch.
- `weights/cosmos3_super_forward_dynamics_lora/`: the Cosmos3-Super
  forward-dynamics LoRA adapter over camera-pose proxy targets.
- `analysis/`: compact analysis files and the 20-task result matrix.
- `analysis/QWEN3_OMNI_RUN_LINEAGE.md` and
  `analysis/docs/data/qwen3_omni_run_lineage.json`: the Qwen3-Omni v1-v6
  run lineage, where v6 is the current 20-task matrix row and v5 is the
  pinned prior release.
- `results/`: verified public result summaries, package audits, metrics, and
  run reports that correspond to the included Qwen3-Omni and Cosmos3 artifacts.
- `manifest.json`: file-level size and SHA-256 manifest for this staged bundle.

## Current Result Coverage

- Method-task records: `{scored}/{total}` scored.
- Evidence line 1: 1 public sample episode, 2 methods x 20 tasks = 40/40 direct scores.
- Evidence line 2: 128 selected episodes, 7 methods x 20 tasks = 140/140 scores; 134 direct + 6 compact-proxy.
- Task surface: 20 tasks x 9 method families = 180/180 public records.
- Scope boundary: public sample for reproducible task construction; selected public-safe 128-episode artifacts for same-split comparison and model diagnostics.
- Excluded: raw Xperience-10M MP4/HDF5/RRD files, Qwen3 base weights and Cosmos3 base weights,
  full fine-tune checkpoints, private data, logs with machine-local state, and
  generated Hugging Face cache files.

## Canonical Public Repos

- Consolidated package: `https://huggingface.co/{repo_id}`
- Project website: `https://chaoyue0307.github.io/ropedia-xperience-10m-task-suite/`
- GitHub source: `https://github.com/ChaoYue0307/ropedia-xperience-10m-task-suite`
- HF Space: `https://huggingface.co/spaces/cy0307/ropedia-xperience-10m-task-suite`
- Derived artifact dataset: `https://huggingface.co/datasets/cy0307/ropedia-xperience-10m-task-suite-artifacts`
- Baseline model repo: `https://huggingface.co/cy0307/ropedia-xperience-10m-task-baselines`
- Qwen3-Omni LoRA repo: `https://huggingface.co/cy0307/ropedia-qwen3-omni-lora-128ep`
- Cosmos3-Super LoRA repo: `https://huggingface.co/cy0307/ropedia-cosmos3-super-forward-dynamics-lora-128ep`

## Reproducibility

The bundle is rebuilt by:

```bash
python3 scripts/build_hf_weights_results_bundle.py
python3 scripts/publish_hf_bundles.py --skip-space --skip-artifacts --skip-model
```

Use `manifest.json` to verify byte-level integrity after upload.
"""
    path.write_text(text, encoding="utf-8")


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def main() -> int:
    args = parse_args()
    hf_root = args.hf_root.resolve()
    output = args.output.resolve()
    baseline_snapshot = hf_root / "model"
    if not baseline_snapshot.exists():
        raise SystemExit(f"Missing baseline HF snapshot: {baseline_snapshot}")

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    copy_tree(baseline_snapshot, output / "baselines_and_analysis_snapshot")
    copy_tree(QWEN_LEGACY_DIR, output / "weights/qwen3_omni_legacy_lora")
    copy_tree(QWEN_V6_DIR, output / "weights/qwen3_omni_v6_rank64_lora")
    copy_tree(COSMOS3_SUPER_DIR, output / "weights/cosmos3_super_forward_dynamics_lora")

    for relative in ANALYSIS_FILES:
        src = ROOT / relative
        if src.exists():
            copy_file(src, output / "analysis" / relative)

    seen_results: set[str] = set()
    for pattern in VERIFIED_RESULT_GLOBS:
        for src in ROOT.glob(pattern):
            if not src.is_file():
                continue
            relative = src.relative_to(ROOT).as_posix()
            if relative in seen_results:
                continue
            seen_results.add(relative)
            copy_file(src, output / relative)

    matrix = json.loads((ROOT / "docs/data/task_method_20_result_matrix.json").read_text(encoding="utf-8"))
    manifest = {
        "title": "Ropedia Xperience-10M consolidated weights/results bundle",
        "repo_id": args.repo_id,
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source_git_commit": git_commit(),
        "result_matrix": {
            "task_count": matrix.get("task_count"),
            "method_count": matrix.get("method_count"),
            "method_task_record_count": matrix.get("method_task_record_count"),
            "scored_method_task_count": matrix.get("scored_method_task_count"),
            "status": matrix.get("status"),
        },
        "canonical_repos": {
            "github": "https://github.com/ChaoYue0307/ropedia-xperience-10m-task-suite",
            "space": "https://huggingface.co/spaces/cy0307/ropedia-xperience-10m-task-suite",
            "artifact_dataset": "https://huggingface.co/datasets/cy0307/ropedia-xperience-10m-task-suite-artifacts",
            "baseline_model_repo": "https://huggingface.co/cy0307/ropedia-xperience-10m-task-baselines",
            "qwen3_lora_repo": "https://huggingface.co/cy0307/ropedia-qwen3-omni-lora-128ep",
            "cosmos3_super_lora_repo": "https://huggingface.co/cy0307/ropedia-cosmos3-super-forward-dynamics-lora-128ep",
        },
        "included_weight_roots": [
            "baselines_and_analysis_snapshot/artifacts",
            "baselines_and_analysis_snapshot/pytorch_model.bin",
            "weights/qwen3_omni_legacy_lora",
            "weights/qwen3_omni_v6_rank64_lora",
            "weights/cosmos3_super_forward_dynamics_lora",
        ],
        "exclusions": [
            "raw Xperience-10M MP4/HDF5/RRD files",
            "Qwen3 base model weights and Cosmos3 base model weights",
            "full fine-tune checkpoints",
            "machine-local logs and PID files",
            "Hugging Face cache state",
        ],
    }
    write_readme(output / "README.md", args.repo_id, manifest)
    manifest["files"] = collect_files(output)
    manifest["file_count"] = len(manifest["files"])
    manifest["total_bytes"] = sum(row["bytes"] for row in manifest["files"])
    write_json(output / "manifest.json", manifest)
    print(f"Staged {manifest['file_count']} files ({manifest['total_bytes']} bytes) at {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

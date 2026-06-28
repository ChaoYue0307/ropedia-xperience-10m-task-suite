#!/usr/bin/env python3
"""Build a complete navigable resource map for scripts, results, docs, and HF bundles."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HF_ROOT = ROOT.parent / "hf_publish"
JSON_OUTPUT = ROOT / "docs/data/project_resource_map.json"
MD_OUTPUT = ROOT / "PROJECT_RESOURCE_MAP.md"
GITHUB_BLOB = "https://github.com/ChaoYue0307/ropedia-xperience-10m-task-suite/blob/main"
GITHUB_RAW = "https://raw.githubusercontent.com/ChaoYue0307/ropedia-xperience-10m-task-suite/main"
PAGES_BASE = "https://chaoyue0307.github.io/ropedia-xperience-10m-task-suite"

SCAN_ROOTS = [
    "scripts",
    "docs",
    "docs/data",
    "docs/assets",
    "results",
    "configs",
    "notes",
]

TOP_LEVEL_PATTERNS = [
    "*.md",
    "*.json",
    "*.toml",
    "*.txt",
    "Dockerfile",
    "requirements.txt",
    "package.json",
]

SKIP_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    "test-results",
}

HF_BUNDLES = {
    "space": {
        "label": "HF Space bundle",
        "repo_type": "space",
        "repo_id": "cy0307/ropedia-xperience-10m-task-suite",
        "raw_base": "https://huggingface.co/spaces/cy0307/ropedia-xperience-10m-task-suite/raw/main",
    },
    "artifacts": {
        "label": "HF artifact dataset bundle",
        "repo_type": "dataset",
        "repo_id": "cy0307/ropedia-xperience-10m-task-suite-artifacts",
        "raw_base": "https://huggingface.co/datasets/cy0307/ropedia-xperience-10m-task-suite-artifacts/raw/main",
    },
    "model": {
        "label": "HF baseline model bundle",
        "repo_type": "model",
        "repo_id": "cy0307/ropedia-xperience-10m-task-baselines",
        "raw_base": "https://huggingface.co/cy0307/ropedia-xperience-10m-task-baselines/raw/main",
    },
    "weights_results": {
        "label": "HF weights/results bundle",
        "repo_type": "model",
        "repo_id": "cy0307/ropedia-xperience-10m-weights-results",
        "raw_base": "https://huggingface.co/cy0307/ropedia-xperience-10m-weights-results/raw/main",
    },
    "qwen3_lora_128ep": {
        "label": "Qwen3-Omni LoRA bundle",
        "repo_type": "model",
        "repo_id": "cy0307/ropedia-qwen3-omni-lora-128ep",
        "raw_base": "https://huggingface.co/cy0307/ropedia-qwen3-omni-lora-128ep/raw/main",
    },
    "cosmos3_super_forward_dynamics_lora_128ep": {
        "label": "Cosmos3-Super LoRA bundle",
        "repo_type": "model",
        "repo_id": "cy0307/ropedia-cosmos3-super-forward-dynamics-lora-128ep",
        "raw_base": "https://huggingface.co/cy0307/ropedia-cosmos3-super-forward-dynamics-lora-128ep/raw/main",
    },
}


def run_git_lines(*args: str) -> set[str]:
    try:
        output = subprocess.check_output(["git", "-C", str(ROOT), *args], text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return set()
    return {line.strip() for line in output.splitlines() if line.strip()}


def is_skipped(path: Path) -> bool:
    return any(part in SKIP_PARTS for part in path.parts)


def repo_files() -> list[Path]:
    files: set[Path] = set()
    for root_name in SCAN_ROOTS:
        root = ROOT / root_name
        if root.exists():
            files.update(path for path in root.rglob("*") if path.is_file() and not is_skipped(path))
    for pattern in TOP_LEVEL_PATTERNS:
        files.update(path for path in ROOT.glob(pattern) if path.is_file())
    return sorted(files, key=lambda path: path.relative_to(ROOT).as_posix())


def hf_files(hf_root: Path) -> list[tuple[str, Path]]:
    if not hf_root.exists():
        return []
    files: list[tuple[str, Path]] = []
    for bundle_dir in sorted(hf_root.iterdir()):
        if not bundle_dir.is_dir() or is_skipped(bundle_dir):
            continue
        bundle = bundle_dir.name
        for path in bundle_dir.rglob("*"):
            if path.is_file() and not is_skipped(path):
                files.append((bundle, path))
    return sorted(files, key=lambda item: (item[0], item[1].relative_to(hf_root / item[0]).as_posix()))


def file_kind(path: str, source_root: str) -> str:
    if source_root.startswith("hf_publish"):
        return "hf_bundle"
    if path.startswith("scripts/omni/"):
        return "omni_script"
    if path.startswith("scripts/"):
        return "script"
    if path.startswith("results/omni_finetune/"):
        return "omni_result"
    if path.startswith("results/"):
        return "result"
    if path.startswith("docs/data/"):
        return "structured_data"
    if path.startswith("docs/assets/"):
        return "visual_asset"
    if path.startswith("configs/"):
        return "config"
    if path.startswith("notes/"):
        return "note"
    if path.endswith(".md"):
        return "documentation"
    return "resource"


def script_role(path: str) -> str:
    name = Path(path).name
    stem = Path(path).stem
    if name.endswith(".md"):
        return "runbook"
    if stem.startswith("validate_"):
        return "validator"
    if stem.startswith("build_"):
        return "builder"
    if stem.startswith(("render_", "generate_")):
        return "renderer"
    if stem.startswith(("sync_", "publish_", "upload_")):
        return "publisher"
    if stem.startswith(("train_", "run_train", "run_")):
        return "runner"
    if stem.startswith(("eval_", "score_")):
        return "evaluator"
    if stem.startswith(("collect_", "merge_", "package_", "prepare_")):
        return "packager"
    if stem.startswith(("watch_", "monitor_", "defer_", "launch_", "auto_start_")):
        return "orchestrator"
    if stem.startswith(("analyze_", "audit_", "probe_", "diagnose_")):
        return "auditor"
    if stem.startswith(("export_", "extract_", "download_", "stage_", "transfer_")):
        return "data-prep"
    return "utility"


def purpose_for(path: str, kind: str) -> str:
    name = Path(path).name
    stem = Path(path).stem.replace("_", " ")
    if kind in {"script", "omni_script"}:
        return f"{script_role(path).replace('-', ' ').title()} script for {stem}."
    if kind in {"result", "omni_result"}:
        return "Committed or local result artifact: metrics, predictions, model weights, logs, manifests, or generated analysis."
    if kind == "structured_data":
        return "Website and Hugging Face structured data mirror."
    if kind == "visual_asset":
        return "Website figure, chart, icon, preview, or generated visual asset."
    if kind == "hf_bundle":
        return "Prepared Hugging Face upload bundle file; publish with scripts/publish_hf_bundles.py."
    if kind == "config":
        return "Configuration used by training, evaluation, packaging, or public-surface generation."
    if name.endswith(".md"):
        return "Human-readable project note or public documentation."
    return "Project resource."


def repo_access(path: str, tracked: bool) -> dict:
    access: dict[str, str | bool] = {
        "local_path": f"repo:{path}",
        "tracked_in_git": tracked,
    }
    if tracked:
        access["github"] = f"{GITHUB_BLOB}/{path}"
        access["raw"] = f"{GITHUB_RAW}/{path}"
    if path.startswith("docs/") and tracked:
        site_path = path.removeprefix("docs/")
        access["site"] = f"{PAGES_BASE}/{site_path}"
    return access


def hf_access(bundle: str, relative_path: str, path: Path) -> dict:
    meta = HF_BUNDLES.get(bundle, {})
    access: dict[str, str | bool] = {
        "local_path": f"hf_publish/{bundle}:{relative_path}",
        "bundle": bundle,
        "published_repo": meta.get("repo_id", ""),
    }
    raw_base = meta.get("raw_base")
    if raw_base:
        access["hf_raw"] = f"{raw_base}/{relative_path}"
    return access


def file_record(path: Path, rel: str, *, source_root: str, tracked: bool = False, bundle: str | None = None) -> dict:
    stat = path.stat()
    kind = file_kind(rel, source_root)
    record = {
        "path": rel,
        "name": path.name,
        "kind": kind,
        "role": script_role(rel) if kind in {"script", "omni_script"} else kind.replace("_", " "),
        "source_root": source_root,
        "purpose": purpose_for(rel, kind),
        "size_bytes": stat.st_size,
        "size_human": human_size(stat.st_size),
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).replace(microsecond=0).isoformat(),
    }
    if bundle:
        record["bundle"] = bundle
        record["access"] = hf_access(bundle, rel, path)
    else:
        record["access"] = repo_access(rel, tracked)
    return record


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{size} B"


def build_payload(hf_root: Path) -> dict:
    tracked = run_git_lines("ls-files")
    entries: list[dict] = []
    for path in repo_files():
        rel = path.relative_to(ROOT).as_posix()
        entries.append(file_record(path, rel, source_root="repo", tracked=rel in tracked))
    for bundle, path in hf_files(hf_root):
        rel = path.relative_to(hf_root / bundle).as_posix()
        entries.append(file_record(path, rel, source_root=f"hf_publish/{bundle}", bundle=bundle))

    counts_by_kind = Counter(entry["kind"] for entry in entries)
    counts_by_root = Counter(entry["source_root"] for entry in entries)
    counts_by_script_role = Counter(entry["role"] for entry in entries if entry["kind"] in {"script", "omni_script"})
    public_repo_files = sum(1 for entry in entries if entry.get("access", {}).get("github"))
    site_files = sum(1 for entry in entries if entry.get("access", {}).get("site"))
    hf_bundle_files = sum(1 for entry in entries if entry["kind"] == "hf_bundle")
    local_untracked_repo_files = sum(1 for entry in entries if entry["source_root"] == "repo" and not entry.get("access", {}).get("tracked_in_git"))

    return {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "repo_root": ".",
        "hf_publish_root": "../hf_publish",
        "summary": {
            "total_files": len(entries),
            "script_files": counts_by_kind["script"] + counts_by_kind["omni_script"],
            "result_files": counts_by_kind["result"] + counts_by_kind["omni_result"],
            "structured_data_files": counts_by_kind["structured_data"],
            "visual_asset_files": counts_by_kind["visual_asset"],
            "hf_bundle_files": hf_bundle_files,
            "github_linked_files": public_repo_files,
            "site_linked_files": site_files,
            "local_untracked_repo_files": local_untracked_repo_files,
        },
        "counts_by_kind": dict(sorted(counts_by_kind.items())),
        "counts_by_source_root": dict(sorted(counts_by_root.items())),
        "counts_by_script_role": dict(sorted(counts_by_script_role.items())),
        "quick_routes": [
            {
                "question": "Which script builds or validates a public artifact?",
                "filter": "kind=script or kind=omni_script",
                "best_view": "Use script role filters: builder, validator, renderer, publisher, runner, evaluator, packager.",
            },
            {
                "question": "Where are the current metrics, predictions, and model-output files?",
                "filter": "kind=result or kind=omni_result",
                "best_view": "Use the result rows and open the GitHub link when tracked, or local path when a file is still local-only.",
            },
            {
                "question": "Which public files feed the website and Hugging Face mirrors?",
                "filter": "kind=structured_data or kind=visual_asset or kind=hf_bundle",
                "best_view": "Open site links for docs files and HF raw links for staged bundle files.",
            },
        ],
        "entries": entries,
    }


def md_link(label: str, href: str | None) -> str:
    if not href:
        return label
    return f"[{label}]({href})"


def entry_link(entry: dict) -> str:
    access = entry.get("access", {})
    for key in ("site", "github", "hf_raw", "raw"):
        href = access.get(key)
        if href:
            return md_link(entry["path"], href)
    return f"`{entry['path']}`"


def table_rows(entries: list[dict], *, kinds: set[str] | None = None, limit: int | None = None) -> list[str]:
    rows = []
    selected = [entry for entry in entries if kinds is None or entry["kind"] in kinds]
    for entry in selected[:limit]:
        access = entry.get("access", {})
        public = []
        if access.get("site"):
            public.append(md_link("site", str(access["site"])))
        if access.get("github"):
            public.append(md_link("repo", str(access["github"])))
        if access.get("hf_raw"):
            public.append(md_link("HF raw", str(access["hf_raw"])))
        if not public:
            public.append("local/staged")
        rows.append(
            "| "
            + " | ".join(
                [
                    entry_link(entry),
                    entry["kind"],
                    entry.get("role", ""),
                    entry["size_human"],
                    ", ".join(public),
                ]
            )
            + " |"
        )
    return rows


def write_markdown(payload: dict, path: Path) -> None:
    entries = payload["entries"]
    lines = [
        "# Project Resource Map",
        "",
        "Generated inventory for scripts, results, website data, visual assets, and prepared Hugging Face bundles.",
        "",
        "This file does not move or rewrite result artifacts. It indexes them in place so readers and maintainers can open the right file directly.",
        "",
        "## Summary",
        "",
    ]
    for key, value in payload["summary"].items():
        lines.append(f"- **{key.replace('_', ' ').title()}**: {value}")
    lines.extend(
        [
            "",
            "## Script Role Counts",
            "",
            "| Role | Files |",
            "|---|---:|",
        ]
    )
    for role, count in payload["counts_by_script_role"].items():
        lines.append(f"| {role} | {count} |")
    lines.extend(
        [
            "",
            "## Complete Script Inventory",
            "",
            "| File | Kind | Role | Size | Direct access |",
            "|---|---|---|---:|---|",
        ]
    )
    lines.extend(table_rows(entries, kinds={"script", "omni_script"}))
    lines.extend(
        [
            "",
            "## Complete Result Inventory",
            "",
            "| File | Kind | Role | Size | Direct access |",
            "|---|---|---|---:|---|",
        ]
    )
    lines.extend(table_rows(entries, kinds={"result", "omni_result"}))
    lines.extend(
        [
            "",
            "## Website Data And Assets",
            "",
            "| File | Kind | Role | Size | Direct access |",
            "|---|---|---|---:|---|",
        ]
    )
    lines.extend(table_rows(entries, kinds={"structured_data", "visual_asset"}))
    lines.extend(
        [
            "",
            "## Prepared Hugging Face Bundles",
            "",
            "| File | Kind | Role | Size | Direct access |",
            "|---|---|---|---:|---|",
        ]
    )
    lines.extend(table_rows(entries, kinds={"hf_bundle"}))
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-root", type=Path, default=HF_ROOT)
    parser.add_argument("--json-output", type=Path, default=JSON_OUTPUT)
    parser.add_argument("--md-output", type=Path, default=MD_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_payload(args.hf_root)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(payload, args.md_output)
    print(
        "wrote "
        f"{args.json_output.relative_to(ROOT)} and {args.md_output.relative_to(ROOT)} "
        f"({payload['summary']['total_files']} files indexed)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Sync repo publication files into the prepared Hugging Face bundles.

The upload step publishes ../hf_publish/{space,artifacts,model}; this helper
keeps those staging folders aligned with the same file groups checked by
validate_mirror_parity.py.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HF_ROOT = ROOT.parent / "hf_publish"
PARITY_SCRIPT = ROOT / "scripts/validate_mirror_parity.py"
STALE_MIRROR_FILES = [
    "artifacts/scripts/omni/collect_qwen3_v4_publication_artifacts.py",
    "model/scripts/omni/collect_qwen3_v4_publication_artifacts.py",
]
GENERATED_REPORT_DATA_FILES = [
    # The parity validator rewrites this report, so it is synced after checks
    # rather than included in the self-referential hash parity file set.
    "mirror_parity.json",
]
ENHANCEMENT_MARKER = "docs/data/task_suite_enhancement_128.json"
ENHANCEMENT_CARD_BLOCK = """
## 128-Episode Enhancement Pack

The no-new-episode suite push is recorded in `TASK_SUITE_ENHANCEMENT_128.md`
and `docs/data/task_suite_enhancement_128.json`. It recommends
`multiscale_20s10_40s20_80s40`, hierarchical action/subtask targets,
label-normalized scoring, and compact raw-feature shards before adding more
episodes.
"""
TIER2_MARKER = "docs/data/task_suite_20.json"
TIER2_CARD_BLOCK = """
## Unified 20-Task Suite

The public-sample task surface is now one unified 20-task suite in
`TASK_SUITE_20.md` and `docs/data/task_suite_20.json`. Tasks 1-12 are the
original sample tasks; Tasks 13-20 reuse the same 20-frame windows, 5-frame
stride, feature manifest, chronological split, and minimal/neural head pattern.
The historical `tier2_task_suite` path is retained only for stable artifact
links to tasks 13-20. The unified radar chart is published as
`docs/assets/charts/unified_task_model_radar.svg` with values in
`docs/data/unified_task_model_radar.json`; the 9-method by 20-task completion
matrix is in `docs/data/task_method_20_result_matrix.json`, with the explicit
gap audit in `docs/data/task_method_20_gap_audit.json`. Split radars for
the one-episode baselines and selected 128-episode methods are published as
`docs/assets/charts/single_episode_task_model_radar.svg` and
`docs/assets/charts/episode128_task_model_radar.svg`.
"""
READER_MAP_MARKER = "docs/data/public_reader_map.json"
READER_MAP_ARTIFACT_ROW = (
    "| Choose the right public surface | `PUBLIC_READER_MAP.md`, "
    "`docs/data/public_reader_map.json` |"
)
READER_MAP_CARD_BLOCK = """
## Public Surface Map

Use `PUBLIC_READER_MAP.md` and `docs/data/public_reader_map.json` to choose
between the GitHub repo, GitHub Pages dashboard, HF Space, artifact dataset,
baseline model repo, Qwen3/Cosmos model repos, and release-health checks without
losing the full evidence trail.
"""
QWEN_COMPARISON_MARKER = "docs/data/qwen3_v5_v6_comparison.json"
QWEN_COMPARISON_ROW = (
    "| Compare Qwen3 v5/v6 diagnostic branches | "
    "`docs/data/qwen3_v5_v6_comparison.json` |"
)
QWEN_ARTIFACT_OLD_BULLET = """- A current verified Qwen3-Omni strict-label v3 held-out package for the
  selected 96/16/16 episode split, with 100.00% JSON validity and weak
  action/subtask quality documented as the next error-analysis target."""
QWEN_ARTIFACT_CURRENT_BULLET = """- The latest verified Qwen3-Omni LoRA v6 diagnostic package for the selected
  96/16/16 episode split includes 34,269 exported windows, 4,032 held-out test
  predictions, 99.90% JSON validity, and public-safe metrics/predictions."""
README_QWEN_OLD_PARAGRAPH = """The current verified diagnostic package uses the same selected split and 8-GPU
training path, records validation loss over 512 validation windows, and keeps
the held-out test split sealed for final evaluation. The next pass should keep
this package contract while tightening JSON decoding, target formatting, and
action/subtask error analysis."""
README_QWEN_CURRENT_PARAGRAPH = """The latest verified diagnostic package uses the same selected split and 8-GPU
training path, includes the full held-out evaluation with 4,032 predictions and
99.90% JSON validity, and keeps raw data plus full Qwen weights out of the
public repos. The next pass should keep this package contract while improving
action/subtask target quality and error analysis."""


def load_parity_module():
    spec = importlib.util.spec_from_file_location("validate_mirror_parity", PARITY_SCRIPT)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not load {PARITY_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def copy_file(src: Path, destinations: list[Path], *, dry_run: bool) -> list[dict]:
    records = []
    if not src.is_file():
        raise SystemExit(f"Missing source file: {src}")
    for dst in destinations:
        records.append({"source": src.relative_to(ROOT).as_posix(), "dest": dst.as_posix()})
        if dry_run:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-root", type=Path, default=DEFAULT_HF_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true", help="print machine-readable copy records")
    return parser.parse_args()


def prune_stale_files(hf_root: Path, *, dry_run: bool) -> list[str]:
    removed = []
    for relative_path in STALE_MIRROR_FILES:
        path = hf_root / relative_path
        if not path.exists():
            continue
        removed.append(path.as_posix())
        if not dry_run:
            path.unlink()
    return removed


def ensure_enhancement_card_links(hf_root: Path, *, dry_run: bool) -> list[str]:
    updated = []
    for relative_path in ("artifacts/README.md", "model/README.md"):
        path = hf_root / relative_path
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        if ENHANCEMENT_MARKER in text:
            continue
        insert_before = "\n## Dataset Boundary" if relative_path.startswith("artifacts/") else "\n## Start Here"
        if insert_before in text:
            text = text.replace(insert_before, ENHANCEMENT_CARD_BLOCK + insert_before, 1)
        else:
            text = text.rstrip() + "\n" + ENHANCEMENT_CARD_BLOCK
        updated.append(relative_path)
        if not dry_run:
            path.write_text(text, encoding="utf-8")
    return updated


def ensure_tier2_card_links(hf_root: Path, *, dry_run: bool) -> list[str]:
    updated = []
    for relative_path in ("space/README.md", "artifacts/README.md", "model/README.md"):
        path = hf_root / relative_path
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        text = text.replace(
            "original sample tasks; tasks 13-20 reuse",
            "original sample tasks; Tasks 13-20 reuse",
        )
        if "docs/data/unified_task_model_radar.json" not in text:
            text = text.replace(
                "links to tasks 13-20.\n",
                "links to tasks 13-20. The unified radar chart is published as\n"
                "`docs/assets/charts/unified_task_model_radar.svg` with values in\n"
                "`docs/data/unified_task_model_radar.json`; the 9-method by\n"
                "20-task completion matrix is in\n"
                "`docs/data/task_method_20_result_matrix.json`, with the explicit\n"
                "gap audit in `docs/data/task_method_20_gap_audit.json`. Split radars are in\n"
                "`docs/assets/charts/single_episode_task_model_radar.svg` and\n"
                "`docs/assets/charts/episode128_task_model_radar.svg`.\n",
            )
        if (
            "docs/data/unified_task_model_radar.json" in text
            and "docs/data/task_method_20_result_matrix.json" not in text
        ):
            text = text.replace(
                "`docs/data/unified_task_model_radar.json`.",
                "`docs/data/unified_task_model_radar.json`; the 9-method by 20-task\n"
                "completion matrix is in `docs/data/task_method_20_result_matrix.json`,\n"
                "with the explicit gap audit in `docs/data/task_method_20_gap_audit.json`.",
            )
        if (
            "docs/data/task_method_20_result_matrix.json" in text
            and "docs/data/task_method_20_gap_audit.json" not in text
        ):
            text = text.replace(
                "`docs/data/task_method_20_result_matrix.json`.",
                "`docs/data/task_method_20_result_matrix.json`, with the explicit\n"
                "gap audit in `docs/data/task_method_20_gap_audit.json`.",
            )
        if (
            "docs/data/task_method_20_result_matrix.json" in text
            and "docs/assets/charts/single_episode_task_model_radar.svg" not in text
        ):
            text = text.replace(
                "`docs/data/task_method_20_result_matrix.json`.",
                "`docs/data/task_method_20_result_matrix.json`. Split radars are in\n"
                "`docs/assets/charts/single_episode_task_model_radar.svg` and\n"
                "`docs/assets/charts/episode128_task_model_radar.svg`.",
            )
        if TIER2_MARKER in text:
            if not dry_run:
                path.write_text(text, encoding="utf-8")
            continue
        insert_before = "\n## Dataset Boundary" if relative_path.startswith("artifacts/") else "\n## Start Here"
        if insert_before in text:
            text = text.replace(insert_before, TIER2_CARD_BLOCK + insert_before, 1)
        else:
            text = text.rstrip() + "\n" + TIER2_CARD_BLOCK
        updated.append(relative_path)
        if not dry_run:
            path.write_text(text, encoding="utf-8")
    return updated


def ensure_reader_map_card_links(hf_root: Path, *, dry_run: bool) -> list[str]:
    updated = []
    artifacts_readme = hf_root / "artifacts/README.md"
    if not artifacts_readme.exists():
        return updated
    text = artifacts_readme.read_text(encoding="utf-8")
    original = text
    if READER_MAP_ARTIFACT_ROW not in text:
        table_anchor = "| Reader goal | Artifact |\n| --- | --- |\n"
        if table_anchor in text:
            text = text.replace(table_anchor, table_anchor + READER_MAP_ARTIFACT_ROW + "\n", 1)
        else:
            text = text.rstrip() + "\n\n" + READER_MAP_ARTIFACT_ROW + "\n"
    if "## Public Surface Map" not in text:
        insert_before = "\n## Dataset Boundary"
        if insert_before in text:
            text = text.replace(insert_before, READER_MAP_CARD_BLOCK + insert_before, 1)
        else:
            text = text.rstrip() + "\n" + READER_MAP_CARD_BLOCK
    if text != original:
        updated.append("artifacts/README.md")
        if not dry_run:
            artifacts_readme.write_text(text, encoding="utf-8")
    return updated


def split_hf_frontmatter(text: str) -> tuple[str, str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return "", text
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            frontmatter = "\n".join(lines[: idx + 1]).rstrip() + "\n\n"
            body = "\n".join(lines[idx + 1 :]).lstrip()
            return frontmatter, body
    return "", text


def refresh_project_readme_cards(hf_root: Path, *, dry_run: bool) -> list[str]:
    """Keep HF project cards aligned with the current repo README body."""

    project_readme = (ROOT / "README.md").read_text(encoding="utf-8").rstrip() + "\n"
    updated = []
    for relative_path in ("space/PROJECT_README.md", "artifacts/PROJECT_README.md", "model/PROJECT_README.md"):
        path = hf_root / relative_path
        if not path.exists():
            continue
        original = path.read_text(encoding="utf-8")
        if original == project_readme:
            continue
        updated.append(relative_path)
        if not dry_run:
            path.write_text(project_readme, encoding="utf-8")

    for relative_path in ("space/README.md", "model/README.md"):
        path = hf_root / relative_path
        if not path.exists():
            continue
        original = path.read_text(encoding="utf-8")
        frontmatter, _body = split_hf_frontmatter(original)
        refreshed = frontmatter + project_readme
        if original == refreshed:
            continue
        updated.append(relative_path)
        if not dry_run:
            path.write_text(refreshed, encoding="utf-8")
    return updated


def read_current_scaleup_line() -> str:
    for line in (ROOT / "README.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("| Scale-up |"):
            return line
    raise SystemExit("Could not find current Scale-up row in README.md")


def ensure_current_qwen_card_links(hf_root: Path, *, dry_run: bool) -> list[str]:
    updated = []

    artifacts_readme = hf_root / "artifacts/README.md"
    if artifacts_readme.exists():
        text = artifacts_readme.read_text(encoding="utf-8")
        original = text
        if QWEN_COMPARISON_ROW not in text:
            anchor = "| Compare current versions and model groups | `docs/data/omni_model_comparison.json` |"
            text = text.replace(anchor, anchor + "\n" + QWEN_COMPARISON_ROW, 1)
        if QWEN_ARTIFACT_OLD_BULLET in text:
            text = text.replace(QWEN_ARTIFACT_OLD_BULLET, QWEN_ARTIFACT_CURRENT_BULLET, 1)
        if "99.90% JSON validity" not in text:
            text = text.rstrip() + "\n\n" + QWEN_ARTIFACT_CURRENT_BULLET + "\n"
        if QWEN_COMPARISON_MARKER not in text:
            text = text.rstrip() + f"\n\nQwen v5/v6 comparison: `{QWEN_COMPARISON_MARKER}`.\n"
        if text != original:
            updated.append("artifacts/README.md")
            if not dry_run:
                artifacts_readme.write_text(text, encoding="utf-8")

    scaleup_line = read_current_scaleup_line()
    for relative_path in ("model/README.md", "model/PROJECT_README.md"):
        path = hf_root / relative_path
        if not path.exists():
            continue
        original = path.read_text(encoding="utf-8")
        text = original
        if README_QWEN_OLD_PARAGRAPH in text:
            text = text.replace(README_QWEN_OLD_PARAGRAPH, README_QWEN_CURRENT_PARAGRAPH, 1)
        lines = text.splitlines()
        changed = text != original
        for idx, line in enumerate(lines):
            if line.startswith("| Scale-up |"):
                if line != scaleup_line:
                    lines[idx] = scaleup_line
                    changed = True
                break
        else:
            lines.append(scaleup_line)
            changed = True
        if changed:
            updated.append(relative_path)
            if not dry_run:
                path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return updated


def main() -> int:
    args = parse_args()
    hf_root = args.hf_root.expanduser().resolve()
    parity = load_parity_module()

    removed = prune_stale_files(hf_root, dry_run=args.dry_run)
    copied: list[dict] = []
    for filename in parity.DATA_FILES:
        src = ROOT / "docs/data" / filename
        copied += copy_file(
            src,
            [
                hf_root / "space/data" / filename,
                hf_root / "artifacts/data" / filename,
                hf_root / "artifacts/docs/data" / filename,
                hf_root / "model/data" / filename,
                hf_root / "model/docs/data" / filename,
                hf_root / "model/metrics" / filename,
            ],
            dry_run=args.dry_run,
        )

    for filename in GENERATED_REPORT_DATA_FILES:
        src = ROOT / "docs/data" / filename
        copied += copy_file(
            src,
            [
                hf_root / "space/data" / filename,
                hf_root / "artifacts/data" / filename,
                hf_root / "artifacts/docs/data" / filename,
                hf_root / "model/data" / filename,
                hf_root / "model/docs/data" / filename,
                hf_root / "model/metrics" / filename,
            ],
            dry_run=args.dry_run,
        )

    for filename in parity.ASSET_FILES:
        src = ROOT / "docs/assets" / filename
        copied += copy_file(
            src,
            [
                hf_root / "space/assets" / filename,
                hf_root / "artifacts/docs/assets" / filename,
                hf_root / "artifacts/assets" / filename,
                hf_root / "model/assets" / filename,
            ],
            dry_run=args.dry_run,
        )

    for filename in parity.SCRIPT_FILES:
        src = ROOT / "scripts" / filename
        copied += copy_file(
            src,
            [
                hf_root / "artifacts/scripts" / filename,
                hf_root / "model/scripts" / filename,
            ],
            dry_run=args.dry_run,
        )

    for filename in parity.WEBSITE_FILES:
        src = ROOT / "docs" / filename
        copied += copy_file(
            src,
            [
                hf_root / "space" / filename,
                hf_root / "artifacts" / filename,
                hf_root / "artifacts/docs" / filename,
                hf_root / "model" / filename,
                hf_root / "model/docs" / filename,
            ],
            dry_run=args.dry_run,
        )

    result_files = sorted(
        set(parity.RESULT_FILES)
        | set(parity.verified_public_result_files())
        | set(parity.tier2_result_files())
        | set(parity.a100_128_metadata_result_files())
        | set(parity.a100_128_raw20_result_files())
        | set(parity.model_output_task_probe_result_files())
        | set(parity.qwen3_future_task_probe_result_files())
    )
    for filename in result_files:
        src = ROOT / "results" / filename
        copied += copy_file(
            src,
            [
                hf_root / "space/results" / filename,
                hf_root / "artifacts/results" / filename,
                hf_root / "model/results" / filename,
            ],
            dry_run=args.dry_run,
        )

    for filename in parity.DOC_FILES:
        src = ROOT / filename
        copied += copy_file(
            src,
            [
                hf_root / "space" / filename,
                hf_root / "artifacts" / filename,
                hf_root / "model" / filename,
            ],
            dry_run=args.dry_run,
        )

    card_updates = refresh_project_readme_cards(hf_root, dry_run=args.dry_run)
    card_updates += ensure_reader_map_card_links(hf_root, dry_run=args.dry_run)
    card_updates += ensure_enhancement_card_links(hf_root, dry_run=args.dry_run)
    card_updates += ensure_tier2_card_links(hf_root, dry_run=args.dry_run)
    card_updates += ensure_current_qwen_card_links(hf_root, dry_run=args.dry_run)
    summary = {
        "status": "dry_run" if args.dry_run else "synced",
        "hf_root": hf_root.as_posix(),
        "copy_count": len(copied),
        "removed_stale_count": len(removed),
        "removed_stale": removed,
        "card_updates": card_updates,
        "records": copied,
    }
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(
            f"{summary['status'].upper()}: copied {summary['copy_count']} files into {hf_root}; "
            f"removed {summary['removed_stale_count']} stale files"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

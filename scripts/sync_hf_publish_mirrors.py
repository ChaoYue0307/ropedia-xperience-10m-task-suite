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
                hf_root / "artifacts/docs/data" / filename,
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
                hf_root / "artifacts/docs" / filename,
            ],
            dry_run=args.dry_run,
        )

    result_files = sorted(set(parity.RESULT_FILES) | set(parity.verified_public_result_files()))
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

    summary = {
        "status": "dry_run" if args.dry_run else "synced",
        "hf_root": hf_root.as_posix(),
        "copy_count": len(copied),
        "removed_stale_count": len(removed),
        "removed_stale": removed,
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

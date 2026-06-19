#!/usr/bin/env python3
"""Merge parallel raw Xperience-10M caption-extraction shards."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-file", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--input-jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--input-manifest", type=Path, nargs="*", default=[])
    return parser.parse_args()


def read_requested(paths_file: Path) -> list[str]:
    return [line.strip() for line in paths_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_rows_by_path(paths: list[Path], requested: set[str]) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    rows_by_path: dict[str, list[dict[str, Any]]] = {}
    errors: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            errors.append({"path": str(path), "status": "missing_input_jsonl"})
            continue
        grouped: dict[str, list[dict[str, Any]]] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append({"path": str(path), "line": line_number, "status": "json_error", "error": str(exc)})
                    continue
                rel_path = row.get("annotation_path")
                if not isinstance(rel_path, str) or rel_path not in requested:
                    continue
                grouped.setdefault(rel_path, []).append(row)
        for rel_path, rows in grouped.items():
            rows_by_path.setdefault(rel_path, rows)
    return rows_by_path, errors


def manifest_file_statuses(paths: list[Path]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    statuses: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []
    for path in paths:
        payload = read_json(path)
        if not payload:
            errors.append({"path": str(path), "status": "missing_input_manifest"})
            continue
        if payload.get("status") not in {"pass", "running"}:
            errors.append({"path": str(path), "status": "input_manifest_not_pass", "manifest_status": payload.get("status")})
        for item in payload.get("files", []):
            rel_path = item.get("annotation_path")
            if isinstance(rel_path, str):
                statuses.setdefault(rel_path, item)
    return statuses, errors


def main() -> int:
    args = parse_args()
    requested = read_requested(args.paths_file)
    requested_set = set(requested)
    rows_by_path, row_errors = read_rows_by_path(args.input_jsonl, requested_set)
    manifest_statuses, manifest_errors = manifest_file_statuses(args.input_manifest)

    output_rows: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    missing: list[str] = []
    for rel_path in requested:
        rows = rows_by_path.get(rel_path, [])
        manifest_row = manifest_statuses.get(rel_path, {})
        if rows:
            output_rows.extend(rows)
            files.append(
                {
                    "annotation_path": rel_path,
                    "status": "extracted",
                    "output_row_count": len(rows),
                    "source": "merged_jsonl",
                }
            )
        elif manifest_row.get("status") in {"extracted", "skipped_existing"} and int(manifest_row.get("output_row_count", 0) or 0) == 0:
            files.append(
                {
                    "annotation_path": rel_path,
                    "status": "extracted",
                    "output_row_count": 0,
                    "source": "merged_manifest_zero_rows",
                }
            )
        else:
            missing.append(rel_path)
            files.append({"annotation_path": rel_path, "status": "missing"})

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    tmp_jsonl = args.output_jsonl.with_suffix(args.output_jsonl.suffix + ".tmp")
    with tmp_jsonl.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    tmp_jsonl.replace(args.output_jsonl)

    errors = row_errors + manifest_errors + [
        {"annotation_path": rel_path, "status": "missing_merged_rows"} for rel_path in missing
    ]
    manifest = {
        "status": "pass" if not errors else "fail",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "requested_file_count": len(requested),
        "processed_file_count": len(requested) - len(missing),
        "error_count": len(errors),
        "output_jsonl": str(args.output_jsonl),
        "output_row_count": len(output_rows),
        "input_jsonl": [str(path) for path in args.input_jsonl],
        "input_manifest": [str(path) for path in args.input_manifest],
        "files": files,
        "errors": errors,
    }
    args.manifest_json.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ["status", "processed_file_count", "requested_file_count", "error_count", "output_row_count"]}, sort_keys=True))
    return 0 if manifest["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Validate that scored matrix rows agree with their JSON metric sources."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = ROOT / "docs/data/task_method_20_result_matrix.json"
DEFAULT_OUTPUT_JSON = ROOT / "docs/data/task_method_20_source_audit.json"
DEFAULT_OUTPUT_MD = ROOT / "TASK_METHOD_20_SOURCE_AUDIT.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-json", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--relative-tolerance", type=float, default=1e-9)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-12)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def resolve_source(source: str) -> Path:
    path = Path(source)
    return path if path.is_absolute() else ROOT / path


def numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def check_record(record: dict[str, Any], args: argparse.Namespace) -> tuple[str, dict[str, Any] | None]:
    source = record.get("source")
    metric_key = record.get("metric_key")
    raw = numeric(record.get("raw"))
    base = {
        "task_id": record.get("task_id"),
        "task_number": record.get("task_number"),
        "series_id": record.get("series_id"),
        "method": record.get("method"),
        "metric_key": metric_key,
        "source": source,
        "raw": record.get("raw"),
    }
    if not record.get("scored"):
        return "unscored", None
    if raw is None or not metric_key or not source:
        return "skipped_non_numeric_or_missing_source", base

    source_path = resolve_source(str(source))
    if not source_path.exists():
        return "missing_source", {**base, "resolved_source": rel(source_path)}
    if source_path.suffix.lower() != ".json":
        return "skipped_non_json_source", base

    try:
        payload = read_json(source_path)
    except json.JSONDecodeError as exc:
        return "invalid_json_source", {**base, "resolved_source": rel(source_path), "error": str(exc)}
    source_key = str(metric_key)
    source_value = numeric(payload.get(source_key)) if isinstance(payload, dict) else None
    if source_value is None and isinstance(payload, dict):
        primary_metric = payload.get("primary_metric")
        primary_score = numeric(payload.get("primary_score"))
        if primary_score is not None and (primary_metric in {metric_key, None} or str(primary_metric or "") == str(metric_key)):
            source_key = "primary_score"
            source_value = primary_score
        elif primary_score is not None and "primary_score" in payload:
            source_key = "primary_score"
            source_value = primary_score
    if source_value is None:
        return "missing_metric_key", {
            **base,
            "resolved_source": rel(source_path),
            "available_numeric_keys": sorted(
                key for key, value in payload.items() if numeric(value) is not None
            )
            if isinstance(payload, dict)
            else [],
        }
    if not math.isclose(raw, source_value, rel_tol=args.relative_tolerance, abs_tol=args.absolute_tolerance):
        return "value_mismatch", {
            **base,
            "resolved_source": rel(source_path),
            "source_value": source_value,
            "delta": raw - source_value,
        }
    return "checked", {**base, "resolved_source": rel(source_path), "source_key": source_key, "source_value": source_value}


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    matrix = read_json(args.matrix_json)
    records = matrix.get("records", [])
    checked: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}

    for record in records:
        status, detail = check_record(record, args)
        status_counts[status] = status_counts.get(status, 0) + 1
        if detail is None:
            continue
        detail = {"status": status, **detail}
        if status == "checked":
            checked.append(detail)
        elif status.startswith("skipped"):
            skipped.append(detail)
        else:
            failures.append(detail)

    return {
        "title": "Task Method 20 Matrix Source Audit",
        "status": "pass" if not failures else "fail",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_matrix": rel(args.matrix_json),
        "method_task_record_count": matrix.get("method_task_record_count"),
        "scored_method_task_count": matrix.get("scored_method_task_count"),
        "checked_json_metric_count": len(checked),
        "skipped_record_count": len(skipped),
        "failure_count": len(failures),
        "status_counts": dict(sorted(status_counts.items())),
        "failures": failures,
        "skipped_records": skipped[:100],
        "rule": (
            "Every scored row that declares a JSON metric source must have the same "
            "numeric value under that row's metric_key."
        ),
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    failures = report["failures"]
    lines = [
        "# Task Method 20 Matrix Source Audit",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        f"Status: **{report['status']}**",
        "",
        report["rule"],
        "",
        "## Summary",
        "",
        f"- Source matrix: `{report['source_matrix']}`",
        f"- Scored rows: `{report['scored_method_task_count']}/{report['method_task_record_count']}`",
        f"- JSON metric rows checked: `{report['checked_json_metric_count']}`",
        f"- Skipped non-JSON/non-numeric rows: `{report['skipped_record_count']}`",
        f"- Failures: `{report['failure_count']}`",
        "",
    ]
    if failures:
        lines.extend([
            "## Failures",
            "",
            "| Method | Task | Metric | Matrix value | Source value | Source |",
            "| --- | --- | --- | ---: | ---: | --- |",
        ])
        for row in failures:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("series_id")),
                        str(row.get("task_id")),
                        str(row.get("metric_key")),
                        str(row.get("raw")),
                        str(row.get("source_value")),
                        str(row.get("source")),
                    ]
                )
                + " |"
            )
    else:
        lines.append("No JSON source/value mismatches were found.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(args.markdown_output, report)
    print(f"{report['status'].upper()}: wrote {args.output}")
    print(f"{report['status'].upper()}: wrote {args.markdown_output}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

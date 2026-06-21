#!/usr/bin/env python3
"""Build a concise result summary for the two public evidence lines."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MATRIX_JSON = ROOT / "docs/data/task_method_20_result_matrix.json"
LINES_JSON = ROOT / "docs/data/two_evidence_lines.json"
OUTPUT_JSON = ROOT / "docs/data/two_evidence_line_result_summary.json"
OUTPUT_MD = ROOT / "TWO_EVIDENCE_LINE_RESULT_SUMMARY.md"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        escaped = [str(cell).replace("\n", " ").replace("|", "\\|") for cell in row]
        lines.append("| " + " | ".join(escaped) + " |")
    return "\n".join(lines)


def line_for_series(scope: str) -> str:
    if scope.startswith("1 public sample episode"):
        return "single_public_sample_episode"
    if scope.startswith("128 selected episodes"):
        return "selected_128_episode_surface"
    raise ValueError(f"Cannot map series scope to evidence line: {scope}")


def build_payload(matrix: dict, lines: dict) -> dict:
    line_meta = {line["id"]: line for line in lines["lines"]}
    line_rows: dict[str, dict] = {
        line_id: {
            "id": line_id,
            "label": meta["label"],
            "data_unit": meta["data_unit"],
            "primary_use": meta["best_use"],
            "task_count": matrix["task_count"],
            "method_count": 0,
            "method_task_record_count": 0,
            "scored_method_task_count": 0,
            "direct_scored_method_task_count": 0,
            "proxy_scored_method_task_count": 0,
            "methods": [],
            "artifact_entry_points": meta["primary_artifacts"],
        }
        for line_id, meta in line_meta.items()
    }

    series_to_line: dict[str, str] = {}
    for series in matrix["series"]:
        line_id = line_for_series(series["scope"])
        series_to_line[series["id"]] = line_id
        line = line_rows[line_id]
        line["method_count"] += 1
        line["method_task_record_count"] += series["result_record_count"]
        line["scored_method_task_count"] += series["scored_task_count"]
        line["proxy_scored_method_task_count"] += series.get("proxy_scored_task_count", 0)
        line["direct_scored_method_task_count"] += (
            series["scored_task_count"] - series.get("proxy_scored_task_count", 0)
        )
        line["methods"].append(
            {
                "id": series["id"],
                "label": series["label"],
                "scope": series["scope"],
                "method_detail": series.get("method_detail"),
                "scored_task_count": series["scored_task_count"],
                "result_record_count": series["result_record_count"],
                "direct_scored_task_count": (
                    series["scored_task_count"] - series.get("proxy_scored_task_count", 0)
                ),
                "proxy_scored_task_count": series.get("proxy_scored_task_count", 0),
                "status_counts": series.get("status_counts", {}),
            }
        )

    proxy_records = []
    for record in matrix["records"]:
        if not record.get("proxy_scored"):
            continue
        proxy_records.append(
            {
                "line_id": series_to_line[record["series_id"]],
                "task_number": record["task_number"],
                "task_id": record["task_id"],
                "task_label": record["task_label"],
                "series_id": record["series_id"],
                "method": record["method"],
                "metric_key": record.get("metric_key"),
                "source": record.get("source"),
                "reason": record.get("reason"),
            }
        )

    lines_out = list(line_rows.values())
    total_records = sum(line["method_task_record_count"] for line in lines_out)
    total_scored = sum(line["scored_method_task_count"] for line in lines_out)
    total_direct = sum(line["direct_scored_method_task_count"] for line in lines_out)
    total_proxy = sum(line["proxy_scored_method_task_count"] for line in lines_out)

    return {
        "title": "Two Evidence-Line Result Summary",
        "status": "pass",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_matrix": "docs/data/task_method_20_result_matrix.json",
        "source_lines": "docs/data/two_evidence_lines.json",
        "interpretation_rule": lines["interpretation_rule"],
        "summary": {
            "line_count": len(lines_out),
            "task_count": matrix["task_count"],
            "method_count": matrix["method_count"],
            "method_task_record_count": total_records,
            "scored_method_task_count": total_scored,
            "direct_scored_method_task_count": total_direct,
            "proxy_scored_method_task_count": total_proxy,
        },
        "lines": lines_out,
        "proxy_records": proxy_records,
        "reader_policy": {
            "single_public_sample_episode": (
                "Use for task construction, raw-file inspection, local reproducibility, "
                "and controlled Minimal-vs-Neural baseline behavior."
            ),
            "selected_128_episode_surface": (
                "Use for held-out comparison, metadata/raw-feature baselines, Qwen3/Cosmos "
                "branches, and scale-up decisions."
            ),
            "proxy_policy": (
                "Proxy-scored cells stay numeric only when the source artifact and reason "
                "are attached; they should not be read as direct raw-target measurements."
            ),
        },
    }


def write_markdown(payload: dict) -> None:
    summary = payload["summary"]
    line_rows = []
    for line in payload["lines"]:
        method_labels = ", ".join(method["label"] for method in line["methods"])
        line_rows.append(
            [
                line["label"],
                str(line["method_count"]),
                str(line["task_count"]),
                f"{line['scored_method_task_count']}/{line['method_task_record_count']}",
                str(line["direct_scored_method_task_count"]),
                str(line["proxy_scored_method_task_count"]),
                method_labels,
            ]
        )

    proxy_rows = [
        [
            row["task_number"],
            row["task_label"],
            row["method"],
            row.get("metric_key") or "",
            row.get("reason") or "",
        ]
        for row in payload["proxy_records"]
    ]

    text = f"""# Two Evidence-Line Result Summary

Generated: `{payload['generated_at_utc']}`.

Source matrix: [`{payload['source_matrix']}`]({payload['source_matrix']})

Interpretation rule: {payload['interpretation_rule']}

## Public Score Totals

- Lines: {summary['line_count']}
- Tasks per method: {summary['task_count']}
- Methods: {summary['method_count']}
- Scored records: {summary['scored_method_task_count']}/{summary['method_task_record_count']}
- Direct scores: {summary['direct_scored_method_task_count']}
- Compact-proxy scores: {summary['proxy_scored_method_task_count']}

## Line Ledger

{markdown_table(['Line', 'Methods', 'Tasks', 'Scored records', 'Direct scores', 'Proxy scores', 'Method families'], line_rows)}

## Proxy-Scored Cells

{markdown_table(['Task', 'Task label', 'Method', 'Metric', 'Reason'], proxy_rows)}

## Reader Policy

- 1 sample episode: {payload['reader_policy']['single_public_sample_episode']}
- 128 selected episodes: {payload['reader_policy']['selected_128_episode_surface']}
- Proxy scores: {payload['reader_policy']['proxy_policy']}
"""
    OUTPUT_MD.write_text(text, encoding="utf-8")


def main() -> int:
    payload = build_payload(read_json(MATRIX_JSON), read_json(LINES_JSON))
    write_json(OUTPUT_JSON, payload)
    write_markdown(payload)
    print(f"Wrote {OUTPUT_JSON.relative_to(ROOT)} and {OUTPUT_MD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

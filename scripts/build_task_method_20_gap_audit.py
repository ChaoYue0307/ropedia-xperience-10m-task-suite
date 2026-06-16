#!/usr/bin/env python3
"""Build an explicit score-gap audit for the 9-method x 20-task matrix."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MATRIX_JSON = ROOT / "docs/data/task_method_20_result_matrix.json"
OUTPUT_JSON = ROOT / "docs/data/task_method_20_gap_audit.json"
OUTPUT_MD = ROOT / "TASK_METHOD_20_GAP_AUDIT.md"


STATUS_NEXT_STEPS = {
    "not_supported_by_metadata_only_package": (
        "Run the task with raw sensor-feature blocks or add a task-specific "
        "metadata target builder before assigning a numeric score."
    ),
    "unsupported_without_required_target": (
        "Export the missing target field for this 128-episode method, then "
        "rerun the same train/validation/test split."
    ),
    "not_evaluated_in_verified_package": (
        "Generate verified model outputs for this task contract and score them "
        "against the held-out labels."
    ),
}


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
        clean = [str(cell).replace("\n", " ").replace("|", "\\|") for cell in row]
        lines.append("| " + " | ".join(clean) + " |")
    return "\n".join(lines)


def compact_record(record: dict) -> dict:
    return {
        "task_number": record["task_number"],
        "task_id": record["task_id"],
        "task_label": record["task_label"],
        "series_id": record["series_id"],
        "method": record["method"],
        "status": record["status"],
        "status_label": record.get("status_label"),
        "metric_key": record.get("metric_key"),
        "scope": record.get("scope"),
        "reason": record.get("reason"),
        "recommended_next_step": STATUS_NEXT_STEPS.get(
            record["status"], "Review the matrix status and source artifact before scoring."
        ),
    }


def build_payload(matrix: dict) -> dict:
    records = matrix["records"]
    missing_records = [compact_record(row) for row in records if not row.get("scored")]
    proxy_records = [
        {
            "task_number": row["task_number"],
            "task_id": row["task_id"],
            "task_label": row["task_label"],
            "series_id": row["series_id"],
            "method": row["method"],
            "metric_key": row.get("metric_key"),
            "source": row.get("source"),
            "reason": row.get("reason"),
        }
        for row in records
        if row.get("proxy_scored")
    ]

    missing_by_status = Counter(row["status"] for row in missing_records)
    missing_by_method = Counter(row["series_id"] for row in missing_records)
    missing_by_task = defaultdict(list)
    for row in missing_records:
        missing_by_task[f"{row['task_number']:02d} {row['task_label']}"].append(row["series_id"])

    methods = {
        series["id"]: {
            "label": series["label"],
            "scope": series["scope"],
            "kind": series["kind"],
            "result_record_count": series["result_record_count"],
            "scored_task_count": series["scored_task_count"],
            "scoreless_task_count": series["scoreless_task_count"],
            "proxy_scored_task_count": series["proxy_scored_task_count"],
            "status_counts": series["status_counts"],
        }
        for series in matrix["series"]
    }

    return {
        "title": "Task Method 20-Result Gap Audit",
        "status": "pass",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_matrix": "docs/data/task_method_20_result_matrix.json",
        "score_summary": {
            "task_count": matrix["task_count"],
            "method_count": matrix["method_count"],
            "method_task_record_count": matrix["method_task_record_count"],
            "scored_method_task_count": matrix["scored_method_task_count"],
            "scoreless_method_task_count": matrix["method_task_record_count"]
            - matrix["scored_method_task_count"],
            "proxy_scored_method_task_count": len(proxy_records),
        },
        "target_policy": {
            "numeric_score_gate": (
                "A method-task cell is numeric only when a runner or verified package "
                "emits that exact task target and metric."
            ),
            "scoreless_cell_policy": (
                "Unsupported and not-evaluated cells stay explicit in the public matrix "
                "instead of being hidden or backfilled with proxy model claims."
            ),
            "proxy_policy": (
                "Proxy scores are allowed only when the matrix marks them as proxy_scored "
                "and keeps the reason/source attached."
            ),
        },
        "methods": methods,
        "missing_by_status": dict(sorted(missing_by_status.items())),
        "missing_by_method": dict(sorted(missing_by_method.items())),
        "missing_by_task": {
            task: sorted(series_ids) for task, series_ids in sorted(missing_by_task.items())
        },
        "missing_records": missing_records,
        "proxy_records": proxy_records,
        "immediate_actions": [
            {
                "id": "gap_audit",
                "artifact": "docs/data/task_method_20_gap_audit.json",
                "purpose": "Keep the 69 scoreless cells visible and reproducible.",
            },
            {
                "id": "model_output_probe",
                "artifact": "scripts/omni/score_model_output_probes.py",
                "purpose": (
                    "Check whether train/validation/test model outputs exist before "
                    "attempting all-task Qwen3/Cosmos scoring."
                ),
            },
            {
                "id": "guarded_gpu_launcher",
                "artifact": "scripts/omni/launch_all_task_model_scoring_when_free.sh",
                "purpose": (
                    "Start a user-provided all-task scoring command only after enough "
                    "private GPU capacity is idle."
                ),
            },
        ],
    }


def write_markdown(payload: dict) -> None:
    summary = payload["score_summary"]
    method_rows = []
    for method_id, method in payload["methods"].items():
        method_rows.append(
            [
                method["label"],
                method_id,
                f"{method['scored_task_count']}/20",
                str(method["scoreless_task_count"]),
                str(method["proxy_scored_task_count"]),
                ", ".join(f"{key}: {value}" for key, value in method["status_counts"].items()),
            ]
        )

    status_rows = [
        [status, str(count), STATUS_NEXT_STEPS.get(status, "Review matrix status.")]
        for status, count in payload["missing_by_status"].items()
    ]
    missing_rows = [
        [
            f"{row['task_number']:02d}",
            row["task_label"],
            row["method"],
            row["status_label"] or row["status"],
            row["recommended_next_step"],
        ]
        for row in payload["missing_records"]
    ]
    proxy_rows = [
        [
            f"{row['task_number']:02d}",
            row["task_label"],
            row["method"],
            row["metric_key"],
            row["reason"],
        ]
        for row in payload["proxy_records"]
    ]

    text = f"""# Task Method 20-Result Gap Audit

Generated: `{payload['generated_at_utc']}`

This audit is the explicit gap ledger for the 9-method x 20-task result matrix.
It keeps missing cells visible while preserving the rule that a numeric score
requires a real task target and source artifact.

## Score Summary

- Method-task records: `{summary['method_task_record_count']}`
- Numeric scored records: `{summary['scored_method_task_count']}`
- Scoreless records: `{summary['scoreless_method_task_count']}`
- Proxy-scored records: `{summary['proxy_scored_method_task_count']}`
- Source matrix: [`docs/data/task_method_20_result_matrix.json`](docs/data/task_method_20_result_matrix.json)

## Method Coverage

{markdown_table(['Method', 'ID', 'Scored', 'Scoreless', 'Proxy', 'Status counts'], method_rows)}

## Gap Classes

{markdown_table(['Status', 'Count', 'Next step'], status_rows)}

## Scoreless Records

{markdown_table(['Task', 'Task label', 'Method', 'Status', 'Required evidence'], missing_rows)}

## Proxy Records

{markdown_table(['Task', 'Task label', 'Method', 'Metric', 'Proxy note'], proxy_rows)}

## Immediate Actions

- Keep [`docs/data/task_method_20_gap_audit.json`](docs/data/task_method_20_gap_audit.json) next to the radar and matrix so readers can distinguish scored, proxy-scored, and scoreless cells.
- Use [`scripts/omni/score_model_output_probes.py`](scripts/omni/score_model_output_probes.py) to check whether train/validation/test model outputs are present before trying to extend Qwen3/Cosmos to all 20 task contracts.
- Use [`scripts/omni/launch_all_task_model_scoring_when_free.sh`](scripts/omni/launch_all_task_model_scoring_when_free.sh) as the guarded waiter for a real all-task scoring command when private GPU capacity is available.
"""
    OUTPUT_MD.write_text(text, encoding="utf-8")


def main() -> None:
    matrix = read_json(MATRIX_JSON)
    payload = build_payload(matrix)
    write_json(OUTPUT_JSON, payload)
    write_markdown(payload)
    print(f"wrote {OUTPUT_JSON.relative_to(ROOT)}")
    print(f"wrote {OUTPUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

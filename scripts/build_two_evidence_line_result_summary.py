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


def build_method_blocks(lines_out: list[dict]) -> list[dict]:
    methods_by_id = {
        method["id"]: {**method, "line_label": line["label"], "line_id": line["id"]}
        for line in lines_out
        for method in line["methods"]
    }

    def summarize(method_ids: list[str]) -> dict:
        methods = [methods_by_id[method_id] for method_id in method_ids]
        return {
            "methods": [method["label"] for method in methods],
            "scored_method_task_count": sum(method["scored_task_count"] for method in methods),
            "method_task_record_count": sum(method["result_record_count"] for method in methods),
            "direct_scored_method_task_count": sum(method["direct_scored_task_count"] for method in methods),
            "proxy_scored_method_task_count": sum(method["proxy_scored_task_count"] for method in methods),
        }

    blocks = [
        {
            "line_id": "single_public_sample_episode",
            "line_label": "1 sample episode",
            "block": "Task-head baselines",
            "method_ids": ["minimal", "neural_mlp"],
            "evidence_type": "Direct target metrics on the public sample windows.",
            "read_as": "Task construction, local reproducibility, and Minimal-vs-Neural behavior.",
        },
        {
            "line_id": "selected_128_episode_surface",
            "line_label": "128 selected episodes",
            "block": "Aligned baseline heads",
            "method_ids": [
                "metadata128_simple",
                "metadata128_neural_mlp",
                "raw128_simple",
                "raw128_neural_mlp",
            ],
            "evidence_type": "Direct processed-target metrics where available; compact proxies for documented raw-target gaps.",
            "read_as": "Same-split metadata/raw-feature baseline comparison.",
        },
        {
            "line_id": "selected_128_episode_surface",
            "line_label": "128 selected episodes",
            "block": "Qwen3-Omni series",
            "method_ids": ["qwen3_omni_v6_lora"],
            "evidence_type": "Verified selected-128 Qwen3-Omni v6 LoRA plus source-linked task-specific probes.",
            "read_as": "Trainable Qwen3-Omni diagnostic baseline on the selected-128 surface.",
        },
        {
            "line_id": "selected_128_episode_surface",
            "line_label": "128 selected episodes",
            "block": "Cosmos3 series",
            "method_ids": [
                "cosmos3_super_reasoner",
                "cosmos3_nano_future_window",
            ],
            "evidence_type": "Verified Cosmos3-Super Reasoner and Cosmos3-Nano Future Window public-safe artifacts.",
            "read_as": "Cosmos3 reasoner and future-window diagnostics on the selected-128 surface.",
        },
    ]
    for block in blocks:
        block.update(summarize(block["method_ids"]))
    return blocks


def build_payload(matrix: dict, lines: dict) -> dict:
    line_meta = {line["id"]: line for line in lines["lines"]}
    line_rows: dict[str, dict] = {
        line_id: {
            "id": line_id,
            "label": meta["label"],
            "short_label": meta.get("short_label"),
            "data_unit": meta["data_unit"],
            "result_statement": meta.get("result_statement"),
            "claim_boundary": meta.get("claim_boundary"),
            "not_for": meta.get("not_for"),
            "primary_use": meta["best_use"],
            "task_count": matrix["task_count"],
            "method_count": 0,
            "method_task_record_count": 0,
            "scored_method_task_count": 0,
            "direct_scored_method_task_count": 0,
            "proxy_scored_method_task_count": 0,
            "methods": [],
            "primary_visuals": meta.get("primary_visuals", []),
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
        "reader_summary": lines.get("reader_summary"),
        "score_formula": lines.get("score_formula"),
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
        "method_blocks": build_method_blocks(lines_out),
        "related_model_artifacts": lines.get("related_model_artifacts", []),
        "proxy_records": proxy_records,
        "reading_order": [
            {
                "step": "Choose the evidence line",
                "reason": "Line 1 answers task-lab and reproducibility questions; line 2 answers selected-128 comparison questions.",
            },
            {
                "step": "Open the matching radar",
                "reason": "Use the 1-episode radar for Minimal-vs-Neural behavior and the 128-episode radar for metadata/raw baselines, Qwen3-Omni v6, Cosmos3-Super, and Cosmos3-Nano.",
            },
            {
                "step": "Inspect the matrix row",
                "reason": "Every numeric score is tied to a method, task, metric key, source artifact, and proxy flag.",
            },
            {
                "step": "Check proxy cells before interpreting totals",
                "reason": "The six compact-proxy cells are numeric but are not direct raw-target measurements.",
            },
        ],
        "reader_policy": {
            "single_public_sample_episode": (
                "Use for task construction, raw-file inspection, local reproducibility, "
                "and controlled Minimal-vs-Neural baseline behavior."
            ),
            "selected_128_episode_surface": (
                "Use for held-out comparison, metadata/raw-feature baselines, Qwen3-Omni v6 LoRA, "
                "Cosmos3-Super Reasoner, Cosmos3-Nano Future Window, and scale-up decisions."
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
    entry_rows = []
    method_rows = []
    for line in payload["lines"]:
        method_labels = ", ".join(method["label"] for method in line["methods"])
        line_rows.append(
            [
                line["label"],
                line.get("result_statement") or "",
                line.get("claim_boundary") or line["primary_use"],
                line.get("not_for") or "",
            ]
        )
        entry_rows.append(
            [
                line["label"],
                str(line["method_count"]),
                str(line["task_count"]),
                f"{line['scored_method_task_count']}/{line['method_task_record_count']}",
                str(line["direct_scored_method_task_count"]),
                str(line["proxy_scored_method_task_count"]),
                "<br>".join(line.get("primary_visuals", [])),
                "<br>".join(line["artifact_entry_points"]),
            ]
        )
        for method in line["methods"]:
            method_rows.append(
                [
                    line["label"],
                    method["label"],
                    method.get("method_detail") or "",
                    f"{method['scored_task_count']}/{method['result_record_count']}",
                    str(method["direct_scored_task_count"]),
                    str(method["proxy_scored_task_count"]),
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
    method_block_rows = [
        [
            block["line_label"],
            block["block"],
            ", ".join(block["methods"]),
            f"{block['scored_method_task_count']}/{block['method_task_record_count']}",
            str(block["direct_scored_method_task_count"]),
            str(block["proxy_scored_method_task_count"]),
            block["evidence_type"],
            block["read_as"],
        ]
        for block in payload["method_blocks"]
    ]
    related_artifact_rows = [
        [row.get("name", ""), row.get("role", ""), row.get("repo", "")]
        for row in payload.get("related_model_artifacts", [])
    ]

    text = f"""# Two Evidence-Line Result Summary

Generated: `{payload['generated_at_utc']}`.

Source matrix: [`{payload['source_matrix']}`]({payload['source_matrix']})

Interpretation rule: {payload['interpretation_rule']}

## Read This First

{payload.get('reader_summary') or ''}

Score formula: {payload.get('score_formula') or ''}

| Line | What the scores mean | Valid claim | Do not claim |
| --- | --- | --- | --- |
""" + "\n".join(
        "| " + " | ".join(str(cell).replace("|", "\\|") for cell in row) + " |"
        for row in line_rows
    ) + f"""

## Public Score Totals

- Lines: {summary['line_count']}
- Tasks per method: {summary['task_count']}
- Methods: {summary['method_count']}
- Scored records: {summary['scored_method_task_count']}/{summary['method_task_record_count']}
- Direct scores: {summary['direct_scored_method_task_count']}
- Compact-proxy scores: {summary['proxy_scored_method_task_count']} documented cells

## Line Ledger And Entry Points

{markdown_table(['Line', 'Methods', 'Tasks', 'Scored records', 'Direct scores', 'Proxy scores', 'Primary visuals', 'Source artifacts'], entry_rows)}

## Method Blocks By Evidence Line

{markdown_table(['Line', 'Method block', 'Methods', 'Scored records', 'Direct scores', 'Proxy scores', 'Evidence type', 'Read as'], method_block_rows)}

## Method Detail By Line

{markdown_table(['Line', 'Method', 'Method detail', 'Scored records', 'Direct scores', 'Proxy scores'], method_rows)}

## Related Model Artifacts

{markdown_table(['Artifact', 'Role', 'Link or path'], related_artifact_rows)}

## Proxy-Scored Cells

{markdown_table(['Task', 'Task label', 'Method', 'Metric', 'Reason'], proxy_rows)}

## Reading Order

{markdown_table(['Step', 'Reason'], [[row['step'], row['reason']] for row in payload['reading_order']])}

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

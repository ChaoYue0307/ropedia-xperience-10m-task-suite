#!/usr/bin/env python3
"""Add reader-facing task names to generated public artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from task_display import TASK_DISPLAY_NAMES, task_display_name


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent


JSON_GLOBS = [
    "results/episode_task_suite/**/*.json",
    "results/omni_finetune/multi_episode_128_task_baselines/**/*.json",
    "docs/data/summary_metrics.json",
    "docs/data/evaluation_protocol.json",
    "docs/data/audio_ablation_summary.json",
    "docs/data/single_episode_explorer.json",
]

HF_JSON_GLOBS = [
    "hf_publish/space/data/summary_metrics.json",
    "hf_publish/space/data/evaluation_protocol.json",
    "hf_publish/space/data/audio_ablation_summary.json",
    "hf_publish/space/data/single_episode_explorer.json",
    "hf_publish/artifacts/data/summary_metrics.json",
    "hf_publish/artifacts/data/evaluation_protocol.json",
    "hf_publish/artifacts/data/audio_ablation_summary.json",
    "hf_publish/artifacts/data/single_episode_explorer.json",
    "hf_publish/model/results/omni_finetune/multi_episode_128_task_baselines/**/*.json",
]

TASK_METRICS_CSV = [
    ROOT / "results/omni_finetune/multi_episode_128_task_baselines/task_metrics.csv",
    PARENT / "hf_publish/model/results/omni_finetune/multi_episode_128_task_baselines/task_metrics.csv",
]

BASELINE_REPORTS = [
    ROOT / "results/omni_finetune/multi_episode_128_task_baselines/BASELINE_ALIGNMENT_REPORT.md",
    PARENT / "hf_publish/model/results/omni_finetune/multi_episode_128_task_baselines/BASELINE_ALIGNMENT_REPORT.md",
]


def update_json_obj(value: Any) -> bool:
    changed = False
    if isinstance(value, dict):
        task = value.get("task")
        if isinstance(task, str) and task in TASK_DISPLAY_NAMES:
            display = task_display_name(task)
            if value.get("task_display_name") != display:
                value["task_display_name"] = display
                changed = True
        tasks = value.get("tasks")
        if isinstance(tasks, (dict, list)):
            display_map = {task_id: task_display_name(task_id) for task_id in TASK_DISPLAY_NAMES}
            if value.get("task_display_names") != display_map:
                value["task_display_names"] = display_map
                changed = True
        for item in value.values():
            changed = update_json_obj(item) or changed
    elif isinstance(value, list):
        for item in value:
            changed = update_json_obj(item) or changed
    return changed


def update_json_file(path: Path) -> bool:
    if not path.exists() or path.name in {"object_vocab.json", "history.json"}:
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if not update_json_obj(payload):
        return False
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return True


def update_task_metrics_csv(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if not rows or "task" not in fieldnames:
        return False
    if "task_display_name" not in fieldnames:
        fieldnames.insert(fieldnames.index("task") + 1, "task_display_name")
    changed = False
    for row in rows:
        task = row.get("task", "")
        if task in TASK_DISPLAY_NAMES:
            display = task_display_name(task)
            if row.get("task_display_name") != display:
                row["task_display_name"] = display
                changed = True
    if not changed and "task_display_name" in fieldnames:
        return False
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return True


def update_baseline_report(path: Path) -> bool:
    if not path.exists():
        return False
    lines = path.read_text(encoding="utf-8").splitlines()
    changed = False
    out: list[str] = []
    for line in lines:
        if line == "| task | simple status | simple primary | neural status | neural primary |":
            out.append("| task | artifact id | simple status | simple primary | neural status | neural primary |")
            changed = True
            continue
        if line == "| --- | --- | ---: | --- | ---: |":
            out.append("| --- | --- | --- | ---: | --- | ---: |")
            changed = True
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) >= 7 and parts[1] in TASK_DISPLAY_NAMES:
            task_id = parts[1]
            out.append(
                f"| {task_display_name(task_id)} | `{task_id}` | {parts[2]} | {parts[3]} | {parts[4]} | {parts[5]} |"
            )
            changed = True
            continue
        out.append(line)
    if changed:
        path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return changed


def iter_paths(patterns: list[str], base: Path) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(base.glob(pattern))
    return sorted(set(paths))


def main() -> int:
    changed: list[str] = []
    for path in iter_paths(JSON_GLOBS, ROOT):
        if update_json_file(path):
            changed.append(str(path.relative_to(ROOT)))
    for path in iter_paths(HF_JSON_GLOBS, PARENT):
        if update_json_file(path):
            changed.append(str(path.relative_to(PARENT)))
    for path in TASK_METRICS_CSV:
        if update_task_metrics_csv(path):
            changed.append(str(path))
    for path in BASELINE_REPORTS:
        if update_baseline_report(path):
            changed.append(str(path))
    print(json.dumps({"status": "pass", "changed_files": changed}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

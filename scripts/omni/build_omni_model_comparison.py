#!/usr/bin/env python3
"""Build a compact comparison of the current single-episode and 128-episode runs."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_JSON = ROOT / "docs/data/omni_model_comparison.json"
OUTPUT_MD = ROOT / "results/omni_finetune/OMNI_MODEL_COMPARISON.md"
VERIFIED_PUBLIC = ROOT / "results/omni_finetune/verified_public"

PRIMARY_METRICS = {
    "timeline_action": "macro_f1",
    "timeline_subtask": "macro_f1",
    "transition_detection": "macro_f1",
    "next_action": "macro_f1",
    "hand_trajectory_forecast": "mpjpe",
    "contact_prediction": "macro_f1",
    "object_relevance": "micro_f1",
    "caption_grounding": "mrr",
    "cross_modal_retrieval": "mrr",
    "modality_reconstruction": "r2",
    "temporal_order": "accuracy",
    "misalignment_detection": "f1",
}

TASK_DISPLAY_NAMES = {
    "timeline_action": "Action Recognition",
    "timeline_subtask": "Procedure Step Recognition",
    "transition_detection": "Action Boundary Detection",
    "next_action": "Next-Action Prediction",
    "hand_trajectory_forecast": "Hand Trajectory Forecasting",
    "contact_prediction": "Contact State Prediction",
    "object_relevance": "Object Relevance Prediction",
    "caption_grounding": "Language Grounding",
    "cross_modal_retrieval": "Cross-Modal Retrieval",
    "modality_reconstruction": "Cross-Modal Reconstruction",
    "temporal_order": "Temporal Order Verification",
    "misalignment_detection": "Multimodal Synchronization Detection",
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def scalar(value: Any) -> float | int | str | None:
    if isinstance(value, (float, int, str)) or value is None:
        return value
    return None


def metric_from_task(task_id: str, metrics: dict[str, Any]) -> tuple[str, float | int | str | None]:
    metric_name = PRIMARY_METRICS.get(task_id, "primary_score")
    if metric_name in metrics:
        return metric_name, scalar(metrics.get(metric_name))
    if "primary_metric" in metrics:
        return str(metrics.get("primary_metric")), scalar(metrics.get("primary_score"))
    return metric_name, None


def single_episode_summary() -> dict[str, Any]:
    path = ROOT / "results/episode_task_suite/summary_report.json"
    summary = load_json(path)
    tasks = summary.get("tasks", {}) if isinstance(summary.get("tasks"), dict) else {}
    neural = summary.get("neural_tasks", {}) if isinstance(summary.get("neural_tasks"), dict) else {}
    task_rows = []
    for task_id in sorted(TASK_DISPLAY_NAMES):
        simple_metric, simple_score = metric_from_task(task_id, tasks.get(task_id, {}))
        neural_metric, neural_score = metric_from_task(task_id, neural.get(task_id, {}))
        task_rows.append(
            {
                "task": task_id,
                "task_display_name": TASK_DISPLAY_NAMES[task_id],
                "simple_status": "pass" if task_id in tasks else "missing",
                "simple_primary_metric": simple_metric,
                "simple_primary_score": simple_score,
                "neural_status": "pass" if task_id in neural else "missing",
                "neural_primary_metric": neural_metric,
                "neural_primary_score": neural_score,
            }
        )
    return {
        "id": "v1_single_episode_public_sample",
        "title": "Single-Episode Public-Sample Task Suite",
        "status": "verified",
        "scope": "one public Xperience-10M sample episode",
        "source": rel(path),
        "split": "chronological 70/30 within one episode",
        "counts": {
            "episodes": 1,
            "windows": summary.get("num_windows"),
            "frames": summary.get("num_frames"),
            "feature_dim": summary.get("feature_dim"),
            "task_count": len(tasks),
            "neural_task_count": len(neural),
        },
        "models": ["minimal task heads", "compact neural MLP task heads"],
        "task_metrics": task_rows,
        "interpretation": (
            "This layer verifies the 12 task contracts and raw multimodal feature "
            "pipeline on the public sample. It is not a cross-episode benchmark."
        ),
    }


def read_baseline_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            item: dict[str, Any] = dict(row)
            for key in ("simple_primary_score", "neural_primary_score"):
                if item.get(key) in ("", None):
                    item[key] = None
                else:
                    item[key] = float(item[key])
            task_id = str(item.get("task", ""))
            item["task_display_name"] = TASK_DISPLAY_NAMES.get(task_id, task_id.replace("_", " ").title())
            rows.append(item)
    return rows


def aligned_baseline_summary() -> dict[str, Any]:
    summary_path = ROOT / "results/omni_finetune/multi_episode_128_task_baselines/summary_report.json"
    csv_path = ROOT / "results/omni_finetune/multi_episode_128_task_baselines/task_metrics.csv"
    report_path = ROOT / "results/omni_finetune/multi_episode_128_task_baselines/BASELINE_ALIGNMENT_REPORT.md"
    summary = load_json(summary_path)
    task_rows = read_baseline_csv(csv_path)
    supported_simple = sum(1 for row in task_rows if row.get("simple_status") == "pass")
    supported_neural = sum(1 for row in task_rows if row.get("neural_status") == "pass")
    return {
        "id": "v2_multi_episode_128_aligned_metadata_baselines",
        "title": "128-Episode Aligned Simple/NN Baselines",
        "status": summary.get("status", "unknown"),
        "scope": "selected 128-episode 96/16/16 split",
        "source": rel(report_path),
        "split": "train/val/test by selected episode/session",
        "counts": {
            "rows": summary.get("num_rows"),
            "split_counts": summary.get("split_counts"),
            "episode_counts": summary.get("episode_counts"),
            "task_count": len(task_rows),
            "simple_supported_task_count": supported_simple,
            "neural_supported_task_count": supported_neural,
        },
        "models": ["metadata/text simple baselines", "metadata/text neural MLP baselines"],
        "task_metrics": task_rows,
        "interpretation": (
            "This layer aligns the previous simple and neural baseline framing to "
            "the same selected 96/16/16 split used by the model branches. It uses "
            "public-safe JSONL metadata/text features, so raw-feature-only tasks "
            "remain explicitly unsupported until 128-run sensor feature blocks exist."
        ),
    }


def verified_summaries() -> list[dict[str, Any]]:
    out = []
    for path in sorted(VERIFIED_PUBLIC.glob("*/verified_result_summary.json")):
        payload = load_json(path)
        if not payload:
            continue
        payload["_summary_path"] = rel(path)
        out.append(payload)
    return out


def model_branch_entry(payload: dict[str, Any]) -> dict[str, Any]:
    eval_payload = payload.get("eval", {})
    training = payload.get("training", {})
    dataset = payload.get("dataset", {})
    return {
        "id": payload.get("eval_run_id"),
        "title": payload.get("backbone_display_name", payload.get("backbone")),
        "status": payload.get("status"),
        "backbone": payload.get("backbone"),
        "dataset_contract": payload.get("dataset_contract"),
        "training_objective": payload.get("training_objective"),
        "source": payload.get("_summary_path"),
        "dataset_run_id": payload.get("dataset_run_id"),
        "train_run_id": payload.get("train_run_id"),
        "eval_run_id": payload.get("eval_run_id"),
        "counts": {
            "dataset_samples": dataset.get("num_samples"),
            "dataset_episodes": dataset.get("num_episodes"),
            "split_counts": dataset.get("split_counts"),
            "train_samples": training.get("num_train_samples"),
            "val_samples": training.get("num_val_samples"),
            "eval_samples": eval_payload.get("num_samples"),
            "held_out_episode_count": eval_payload.get("held_out_episode_count"),
            "num_processes": training.get("num_processes"),
        },
        "primary_metrics": eval_payload.get("primary_metrics", {}),
        "history": training.get("history", []),
    }


def model_branch_summary() -> dict[str, Any]:
    branches = [model_branch_entry(payload) for payload in verified_summaries()]
    qwen = [item for item in branches if item.get("backbone") == "qwen3_omni_lora"]
    cosmos = [item for item in branches if item.get("backbone") == "cosmos_world_model"]
    return {
        "id": "v3_multi_episode_foundation_model_branches",
        "title": "128-Episode Foundation-Model Branches",
        "status": "partial_verified",
        "scope": "selected 128-episode split and compatible derived windows",
        "source": "results/omni_finetune/verified_public/",
        "split": "episode/session held-out split; exact task target depends on backbone contract",
        "counts": {
            "verified_branch_count": len(branches),
            "qwen3_verified_package_count": len(qwen),
            "cosmos3_verified_package_count": len(cosmos),
        },
        "models": ["Qwen3-Omni LoRA", "Cosmos3-Nano future-window compatibility branch"],
        "branches": branches,
        "interpretation": (
            "This layer contains the held-out foundation-model packages. Qwen3-Omni "
            "packages evaluate structured JSON task prediction; Cosmos3-Nano currently "
            "evaluates a future-window world-model compatibility adapter, not a full "
            "diffusion-weight fine-tune."
        ),
    }


def build_report() -> dict[str, Any]:
    versions = [single_episode_summary(), aligned_baseline_summary(), model_branch_summary()]
    return {
        "title": "Ropedia Xperience-10M Current Result Versions",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "pass",
        "version_count": len(versions),
        "comparison_rule": (
            "Compare only rows with the same scope and target. Single-episode raw-feature "
            "metrics, 128-episode metadata baselines, Qwen3 structured JSON metrics, and "
            "Cosmos3 future-window metrics answer different questions."
        ),
        "version_reading_notes": [
            "Version 1 is the public-sample 12-task harness with minimal and neural heads.",
            "Version 2 is the selected 128-episode same-split simple/NN baseline alignment.",
            "Version 3 is the verified model-branch layer: the current final Qwen3-Omni LoRA package is the JSON-task diagnostic result, while Cosmos3-Nano is a future-window compatibility result rather than a full Cosmos diffusion fine-tune.",
        ],
        "versions": versions,
        "pending": [
            "Use the final Qwen3 full-eval package as the current Qwen result; older Qwen package rows remain historical diagnostics for comparison.",
            "Promote Cosmos3 from compatibility adapter to full Cosmos3 fine-tuning only after a separate environment with matching Diffusers/Cosmos dependencies is prepared.",
        ],
    }


def fmt_score(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Omni Model Comparison",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        report["comparison_rule"],
        "",
        "## Current Result Versions",
        "",
        "| version | status | scope | source |",
        "| --- | --- | --- | --- |",
    ]
    for version in report["versions"]:
        lines.append(
            "| {title} | {status} | {scope} | `{source}` |".format(
                title=version["title"],
                status=version.get("status"),
                scope=version.get("scope"),
                source=version.get("source"),
            )
        )
    lines.extend(["", "Read the three rows this way:", ""])
    lines.extend(f"- {item}" for item in report.get("version_reading_notes", []))
    lines.extend(["", "## 128-Episode Task Baselines", "", "| task | simple | neural |", "| --- | ---: | ---: |"])
    baseline = report["versions"][1]
    for row in baseline.get("task_metrics", []):
        simple = f"{row.get('simple_primary_metric') or ''} {fmt_score(row.get('simple_primary_score'))}".strip()
        neural = f"{row.get('neural_primary_metric') or ''} {fmt_score(row.get('neural_primary_score'))}".strip()
        lines.append(f"| {row.get('task_display_name')} | {simple} | {neural} |")
    lines.extend(["", "## Verified Model Branches", "", "| branch | backbone | eval samples | held-out episodes | key metrics |", "| --- | --- | ---: | ---: | --- |"])
    for branch in report["versions"][2].get("branches", []):
        metrics = branch.get("primary_metrics", {})
        key_metrics = ", ".join(
            f"{key}={fmt_score(value)}"
            for key, value in metrics.items()
            if key in {"json_validity_rate", "action_macro_f1", "future_retrieval_mrr", "temporal_consistency", "transition_accuracy", "contact_accuracy"}
        )
        counts = branch.get("counts", {})
        lines.append(
            "| {title} | `{backbone}` | {samples} | {episodes} | {metrics} |".format(
                title=branch.get("title"),
                backbone=branch.get("backbone"),
                samples=counts.get("eval_samples", ""),
                episodes=counts.get("held_out_episode_count", ""),
                metrics=key_metrics,
            )
        )
    lines.extend(["", "## Pending", ""])
    lines.extend(f"- {item}" for item in report.get("pending", []))
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    report = build_report()
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    OUTPUT_MD.write_text(markdown(report), encoding="utf-8")
    print(f"PASS: wrote {OUTPUT_JSON}")
    print(f"PASS: wrote {OUTPUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

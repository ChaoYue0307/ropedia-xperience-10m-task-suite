#!/usr/bin/env python3
"""Build the evaluation protocol docs from committed metric artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "docs/data/summary_metrics.json"
OUTPUT_JSON = ROOT / "docs/data/evaluation_protocol.json"
OUTPUT_MD = ROOT / "EVALUATION_PROTOCOL.md"


TASK_PROTOCOL = {
    "timeline_action": {
        "family": "supervised classification",
        "unit": "single window",
        "input": "current 20-frame all-feature window",
        "target": "current action label",
        "primary_metric": "macro_f1",
        "higher_is_better": True,
        "leakage_rule": "No future labels enter the input. Chronological split exposes unseen later action labels.",
    },
    "timeline_subtask": {
        "family": "supervised classification",
        "unit": "single window",
        "input": "current 20-frame all-feature window",
        "target": "current subtask label",
        "primary_metric": "macro_f1",
        "higher_is_better": True,
        "leakage_rule": "No future labels enter the input. Chronological split exposes unseen later subtask labels.",
    },
    "transition_detection": {
        "family": "temporal diagnostic",
        "unit": "single window",
        "input": "current 20-frame all-feature window",
        "target": "action boundary versus steady",
        "primary_metric": "macro_f1",
        "higher_is_better": True,
        "leakage_rule": "Boundary labels are targets only. Boundary timing is evaluated after prediction.",
    },
    "next_action": {
        "family": "short-horizon prediction",
        "unit": "single window",
        "input": "current 20-frame all-feature window at time t",
        "target": "action label at t + 20 frames",
        "primary_metric": "macro_f1",
        "higher_is_better": True,
        "leakage_rule": "Future labels are shifted into targets only; model inputs remain current-window features.",
    },
    "hand_trajectory_forecast": {
        "family": "trajectory regression",
        "unit": "single window",
        "input": "current all-feature window",
        "target": "future left/right hand 3D joints for 10 frames",
        "primary_metric": "mpjpe",
        "higher_is_better": False,
        "leakage_rule": "Future mocap coordinates are targets only, not inputs.",
    },
    "contact_prediction": {
        "family": "binary classification",
        "unit": "single window",
        "input": "non-contact and non-caption feature blocks",
        "target": "any body contact",
        "primary_metric": "macro_f1",
        "higher_is_better": True,
        "leakage_rule": "Contact-derived fields and caption labels are excluded from inputs.",
    },
    "object_relevance": {
        "family": "multi-label classification",
        "unit": "single window",
        "input": "non-caption feature blocks",
        "target": "current relevant object set",
        "primary_metric": "micro_f1",
        "higher_is_better": True,
        "leakage_rule": "Caption/object-label fields are excluded from inputs.",
    },
    "caption_grounding": {
        "family": "retrieval",
        "unit": "caption query",
        "input": "caption object/interaction query plus candidate sensor windows",
        "target": "matching time window",
        "primary_metric": "mrr",
        "higher_is_better": True,
        "leakage_rule": "Queries are ranked against held-out candidate windows; reported ranks are computed after model scoring.",
    },
    "cross_modal_retrieval": {
        "family": "retrieval",
        "unit": "sensor query",
        "input": "motion, IMU, and camera query features",
        "target": "matching depth/video window",
        "primary_metric": "top5_accuracy",
        "higher_is_better": True,
        "leakage_rule": "Query-side and candidate-side feature blocks are split before projection/ranking.",
    },
    "modality_reconstruction": {
        "family": "cross-modal regression",
        "unit": "single window",
        "input": "motion, IMU, and camera features",
        "target": "depth/video feature vector",
        "primary_metric": "r2",
        "higher_is_better": True,
        "leakage_rule": "Target feature blocks are excluded from the input side.",
    },
    "temporal_order": {
        "family": "pairwise diagnostic",
        "unit": "adjacent window pair",
        "input": "two adjacent windows",
        "target": "correct versus reversed order",
        "primary_metric": "f1",
        "higher_is_better": True,
        "leakage_rule": "Pairs are built after windowing; labels are synthetic order labels, not input features.",
    },
    "misalignment_detection": {
        "family": "pairwise diagnostic",
        "unit": "paired modality window",
        "input": "motion side plus visual/depth side",
        "target": "aligned versus shifted by 8 windows",
        "primary_metric": "f1",
        "higher_is_better": True,
        "leakage_rule": "Shift labels are synthetic targets; shifted visual/depth blocks are generated after feature splitting.",
    },
}


def metric_value(metrics: dict, metric_name: str) -> float | None:
    if metric_name == "top5_accuracy":
        return metrics.get("top5_accuracy")
    return metrics.get(metric_name)


def count_record(metrics: dict) -> dict:
    keys = [
        "num_windows",
        "num_samples",
        "num_queries",
        "num_train_windows",
        "num_test_windows",
        "num_train_samples",
        "num_test_samples",
    ]
    return {key: metrics[key] for key in keys if key in metrics}


def build_payload() -> dict:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    suite = summary["suite"]
    minimal_tasks = suite["tasks"]
    neural_tasks = suite.get("neural_tasks", {})
    task_rows = []
    for task_name, protocol in TASK_PROTOCOL.items():
        minimal = minimal_tasks.get(task_name, {})
        neural = neural_tasks.get(task_name, {})
        primary = protocol["primary_metric"]
        task_rows.append(
            {
                "task": task_name,
                **protocol,
                "counts": count_record(minimal),
                "minimal_primary_metric": metric_value(minimal, primary),
                "neural_primary_metric": metric_value(neural, primary) if neural else None,
                "minimal_metric_source": f"results/episode_task_suite/{task_name}/metrics.json",
                "neural_metric_source": f"results/episode_task_suite/neural_mlp/{task_name}/metrics.json",
            }
        )

    return {
        "title": "Ropedia Xperience-10M Task Suite Evaluation Protocol",
        "status": "pass",
        "version": "2026-06-01",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_files": [
            "docs/data/summary_metrics.json",
            "results/episode_task_suite/summary_report.json",
            "results/episode_task_suite/windows.csv",
            "results/episode_task_suite/feature_manifest.json",
        ],
        "scope": {
            "validated_episode_count": 1,
            "annotation": suite["annotation"],
            "num_frames": suite["num_frames"],
            "num_windows": suite["num_windows"],
            "feature_dim": suite["feature_dim"],
            "window_frames": suite["window_frames"],
            "stride_frames": suite["stride_frames"],
            "audio_featurized": False,
            "raw_data_redistributed": False,
        },
        "split_policy": {
            "name": "single_episode_chronological",
            "train_fraction": 0.7,
            "test_fraction": 0.3,
            "why": "The split preserves time order so future episode segments are not mixed randomly into the train set.",
            "limitation": "It is still one episode, so it cannot prove cross-episode generalization.",
        },
        "feature_policy": {
            "input_contract": "8,378-dimensional current feature vector",
            "source_manifest": "results/episode_task_suite/feature_manifest.json",
            "normalization": "Scalers are fit on train windows only for the baseline heads.",
            "audio_status": "Audio is present in sample MP4 streams and visualized in the atlas, but not extracted into the current 8,378-d feature vector.",
        },
        "baselines": [
            {
                "name": "minimal",
                "heads": ["softmax", "binary logistic", "multi-label logistic", "ridge regression", "ridge projection plus cosine ranking"],
                "purpose": "Keep each task contract interpretable and easy to debug.",
            },
            {
                "name": "neural_mlp",
                "heads": ["PyTorch MLP classifier", "PyTorch MLP regressor", "PyTorch MLP multi-label head"],
                "purpose": "Check nonlinear gains before larger omni-model fine-tuning.",
                "config": suite.get("neural_model", {}),
            },
        ],
        "task_protocols": task_rows,
        "global_leakage_controls": [
            "Use chronological train/test splits instead of random window shuffling.",
            "Fit scalers and learned projections on train windows only.",
            "Keep future labels, future mocap, contact labels, object labels, and caption labels on the target side unless a task explicitly treats language as the query.",
            "For cross-modal tasks, split query-side and candidate-side feature blocks before training and ranking.",
            "Report unseen test classes when the chronological split exposes labels absent from the train segment.",
        ],
        "unsupported_interpretations": [
            "Do not infer cross-episode generalization from this single public sample.",
            "Do not treat feature-vector reconstruction as pixel depth, mesh, NeRF, or Gaussian reconstruction.",
            "Do not treat Qwen3-Omni readiness artifacts as a real 32-episode fine-tune.",
            "Do not infer audio-visual learning from the current baseline vector because audio is not featurized.",
        ],
        "scale_up_gate": {
            "required_before_real_omni_claim": [
                "at least 32 valid Xperience-10M episodes",
                "held-out episode split with no train/test episode leakage",
                "manifest, training metadata, progress logs, metrics, predictions, and run report",
                "held-out evaluation on test episodes rather than train windows",
            ],
            "current_status": "prepared but data-gated",
            "evidence": [
                "results/omni_finetune/DATA_BLOCKER_REPORT.md",
                "results/omni_finetune/MULTI_EPISODE_ACCESS_STATUS.md",
            ],
        },
    }


def markdown_table(rows: list[dict]) -> list[str]:
    lines = [
        "| Task | Family | Unit | Input -> target | Primary metric | Minimal | Neural |",
        "| --- | --- | --- | --- | --- | ---: | ---: |",
    ]
    for row in rows:
        metric = row["primary_metric"]
        minimal = row["minimal_primary_metric"]
        neural = row["neural_primary_metric"]
        minimal_text = "n/a" if minimal is None else f"{minimal:.4f}"
        neural_text = "n/a" if neural is None else f"{neural:.4f}"
        direction = "higher better" if row["higher_is_better"] else "lower better"
        lines.append(
            "| {task} | {family} | {unit} | {input} -> {target} | {metric} ({direction}) | {minimal} | {neural} |".format(
                task=row["task"],
                family=row["family"],
                unit=row["unit"],
                input=row["input"],
                target=row["target"],
                metric=metric,
                direction=direction,
                minimal=minimal_text,
                neural=neural_text,
            )
        )
    return lines


def render_markdown(payload: dict) -> str:
    scope = payload["scope"]
    split = payload["split_policy"]
    feature = payload["feature_policy"]
    lines = [
        "# Evaluation Protocol",
        "",
        "This file defines how the public Xperience-10M sample episode is turned",
        "into benchmark-style tasks, how the baselines are evaluated, and what the",
        "reported metrics are allowed to mean.",
        "",
        "## Protocol At A Glance",
        "",
        "| Item | Current protocol |",
        "| --- | --- |",
        f"| Source scope | {scope['validated_episode_count']} public Xperience-10M sample episode |",
        f"| Frames | {scope['num_frames']:,} |",
        f"| Sliding windows | {scope['num_windows']:,} windows, {scope['window_frames']} frames each, stride {scope['stride_frames']} frames |",
        f"| Current feature vector | {scope['feature_dim']:,} dimensions |",
        f"| Split | chronological {int(split['train_fraction'] * 100)}/{int(split['test_fraction'] * 100)} train/test by time |",
        "| Baselines | minimal interpretable heads plus compact neural MLP heads |",
        "| Audio | present in MP4 streams and visualized, but not featurized in the current baseline vector |",
        "| Raw data | not redistributed |",
        "",
        "## Data Unit",
        "",
        "The basic unit is a 20-frame aligned window built from one synchronized",
        "public episode. Feature blocks are documented in",
        "`results/episode_task_suite/feature_manifest.json`; the committed window",
        "table is `results/episode_task_suite/windows.csv`.",
        "",
        "## Split Policy",
        "",
        f"The current suite uses `{split['name']}`: {split['why']} {split['limitation']}",
        "",
        "This makes some classification metrics intentionally harsh: later test",
        "segments can contain action or subtask labels not present in the train",
        "segment. Those cases are recorded in the task metrics as `unseen_test_classes`.",
        "",
        "## Feature And Head Policy",
        "",
        f"- Input contract: {feature['input_contract']}.",
        f"- Source manifest: `{feature['source_manifest']}`.",
        f"- Normalization: {feature['normalization']}",
        f"- Audio status: {feature['audio_status']}",
        "",
        "Minimal heads are used first because they make task contracts debuggable.",
        "Neural MLP heads reuse the same windows, splits, and feature tensors; they",
        "are not foundation models.",
        "",
        "## Task Contracts",
        "",
        *markdown_table(payload["task_protocols"]),
        "",
        "## Leakage Controls",
        "",
    ]
    lines.extend(f"- {item}" for item in payload["global_leakage_controls"])
    lines.extend([
        "",
        "## Unsupported Interpretations",
        "",
    ])
    lines.extend(f"- {item}" for item in payload["unsupported_interpretations"])
    lines.extend([
        "",
        "## Scale-Up Gate",
        "",
        "A real Qwen3-Omni fine-tuning claim requires all of the following before",
        "being presented as model quality:",
        "",
    ])
    lines.extend(f"- {item}" for item in payload["scale_up_gate"]["required_before_real_omni_claim"])
    lines.extend([
        "",
        "Current status: prepared but data-gated. Read",
        "`results/omni_finetune/DATA_BLOCKER_REPORT.md` and",
        "`results/omni_finetune/MULTI_EPISODE_ACCESS_STATUS.md` before interpreting any",
        "Qwen3-Omni artifact.",
        "",
        "## Machine-Readable Copy",
        "",
        "The JSON mirror is `docs/data/evaluation_protocol.json`.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    payload = build_payload()
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    OUTPUT_MD.write_text(render_markdown(payload), encoding="utf-8")
    print(f"PASS: wrote {OUTPUT_JSON}")
    print(f"PASS: wrote {OUTPUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

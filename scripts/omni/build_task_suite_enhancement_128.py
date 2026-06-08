#!/usr/bin/env python3
"""Build a public-safe 128-episode task-suite enhancement pack.

This does not train a new model and does not overwrite prior result packages.
It turns the current verified 128-episode evidence into concrete next-run
contracts: dense-window sizing, hierarchical action targets, bottleneck
priorities, and experiment cards for pushing the task suite harder without
adding more raw episodes.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ID = "task_suite_enhancement_128_v1_20260608"
DEFAULT_OUTPUT_ROOT = ROOT / "results/omni_finetune"
PUBLIC_JSON = ROOT / "docs/data/task_suite_enhancement_128.json"
PUBLIC_MD = ROOT / "TASK_SUITE_ENHANCEMENT_128.md"

WINDOWS_CSV = ROOT / "results/omni_finetune/multi_episode_128_task_baselines/windows.csv"
BASELINE_SUMMARY = ROOT / "results/omni_finetune/multi_episode_128_task_baselines/summary_report.json"
QWEN_V4_METRICS = (
    ROOT
    / "results/omni_finetune/verified_public/"
    / "xperience10m_qwen3_omni_128ep_structured_json_v4_4epoch_full8gpu_lora_eval_test_full/eval/metrics.json"
)
QWEN_V4_PREDICTIONS = (
    ROOT
    / "results/omni_finetune/verified_public/"
    / "xperience10m_qwen3_omni_128ep_structured_json_v4_4epoch_full8gpu_lora_eval_test_full/eval/predictions.jsonl"
)
COSMOS_FD_SUMMARY = (
    ROOT
    / "results/omni_finetune/verified_public/"
    / "xperience10m_cosmos3_super_forward_dynamics_lora_128ep_train1epoch_256_attn_full8gpu_20260608_eval_test_full_fsdp/"
    / "verified_result_summary.json"
)


ACTION_FAMILY_PATTERNS = [
    ("locomotion", ["walk", "enter", "approach", "move through", "move towards", "arrive"]),
    ("reach_grasp_release", ["reach", "grasp", "pick up", "hold", "release"]),
    ("place_arrange_align", ["place", "arrange", "align", "position", "set"]),
    ("manipulate_adjust", ["manipulate", "adjust", "bend", "fold", "open", "close", "secure"]),
    ("tool_cut_mark_write", ["cut", "mark", "draw", "write", "scissor", "knife", "pen", "marker"]),
    ("sort_count_organize", ["sort", "count", "organize", "bundle", "gather"]),
    ("inspect_observe_use", ["inspect", "observe", "browse", "use", "operate", "check", "examine"]),
    ("clean_cook", ["clean", "wipe", "rinse", "wash", "stir", "pot", "stove"]),
]

TASK_PRIORITIES = {
    "timeline_action": "highest",
    "timeline_subtask": "highest",
    "next_action": "highest",
    "hand_trajectory_forecast": "high",
    "cross_modal_retrieval": "high",
    "modality_reconstruction": "high",
    "misalignment_detection": "high",
    "object_relevance": "medium",
    "caption_grounding": "medium",
    "temporal_order": "medium",
    "contact_prediction": "medium",
    "transition_detection": "medium",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_windows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row["start_frame"] = int(row["start_frame"])
            row["end_frame"] = int(row["end_frame"])
            rows.append(row)
    return rows


def action_family(label: str | None) -> str:
    text = (label or "unknown").strip().lower()
    if not text or text == "unknown":
        return "unknown"
    for family, needles in ACTION_FAMILY_PATTERNS:
        if any(needle in text for needle in needles):
            return family
    return "other_fine_action"


def split_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(row["split"] for row in rows).items()))


def episode_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    by_split: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_split[row["split"]].add(row["episode_id"])
    return {split: len(episodes) for split, episodes in sorted(by_split.items())}


def dense_window_estimates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_episode[row["episode_id"]].append(row)
    current_total = len(rows)
    scenarios = [
        {
            "id": "current_export",
            "window_frames": 20,
            "stride_frames": "selected_sparse_windows",
            "role": "current public 128-episode JSON-task export",
        },
        {
            "id": "dense_20f_stride20",
            "window_frames": 20,
            "stride_frames": 20,
            "role": "non-overlap dense coverage over each observed episode frame span",
        },
        {
            "id": "dense_20f_stride10",
            "window_frames": 20,
            "stride_frames": 10,
            "role": "2x overlap action/subtask densification",
        },
        {
            "id": "dense_20f_stride5",
            "window_frames": 20,
            "stride_frames": 5,
            "role": "high-overlap action boundary and transition stress setting",
        },
        {
            "id": "medium_40f_stride20",
            "window_frames": 40,
            "stride_frames": 20,
            "role": "subtask/procedure context window",
        },
        {
            "id": "long_80f_stride40",
            "window_frames": 80,
            "stride_frames": 40,
            "role": "procedure and world-model context window",
        },
    ]
    records: list[dict[str, Any]] = []
    for scenario in scenarios:
        if scenario["id"] == "current_export":
            counts = Counter(row["split"] for row in rows)
            total = current_total
        else:
            counts: Counter[str] = Counter()
            for episode_rows in by_episode.values():
                max_frame = max(row["end_frame"] for row in episode_rows) + 1
                window = int(scenario["window_frames"])
                stride = int(scenario["stride_frames"])
                count = max(0, ((max_frame - window) // stride) + 1) if max_frame >= window else 0
                counts[episode_rows[0]["split"]] += count
            total = sum(counts.values())
        records.append(
            {
                **scenario,
                "estimated_windows": total,
                "estimated_split_windows": dict(sorted(counts.items())),
                "multiplier_vs_current_export": round(total / current_total, 2) if current_total else None,
                "source_note": "Estimated from current public-safe window frame spans; the real exporter must still validate raw-stream availability and label coverage.",
            }
        )

    multiscale = {
        "id": "multiscale_20s10_40s20_80s40",
        "role": "recommended no-new-episode v5 export: short action windows plus medium/long procedure context",
        "components": ["dense_20f_stride10", "medium_40f_stride20", "long_80f_stride40"],
    }
    component_records = {record["id"]: record for record in records}
    total = sum(component_records[item]["estimated_windows"] for item in multiscale["components"])
    split_total: Counter[str] = Counter()
    for item in multiscale["components"]:
        split_total.update(component_records[item]["estimated_split_windows"])
    records.append(
        {
            **multiscale,
            "estimated_windows": total,
            "estimated_split_windows": dict(sorted(split_total.items())),
            "multiplier_vs_current_export": round(total / current_total, 2) if current_total else None,
            "source_note": "Composite planning estimate; store as a new export run rather than replacing existing 128-episode packages.",
        }
    )
    return records


def task_bottlenecks(summary: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for task in summary.get("tasks", []):
        simple = task.get("simple") or {}
        neural = task.get("neural") or {}
        simple_status = simple.get("status", "missing")
        score = simple.get("primary_score")
        unseen = simple.get("unseen_test_class_count")
        classes = simple.get("num_classes")
        if simple_status.startswith("unsupported"):
            bottleneck = "missing raw 128-episode feature blocks"
            next_action = "export compact raw-feature shards for this task before model comparison"
        elif unseen:
            bottleneck = "fine-grained label explosion and held-out unseen labels"
            next_action = "add hierarchical action/subtask families plus label-normalized scoring"
        elif score is not None and float(score) < 0.05:
            bottleneck = "weak public-safe metadata/text baseline"
            next_action = "add dense windows and stronger fusion baselines before interpreting model quality"
        elif task.get("task") in {"contact_prediction", "transition_detection", "temporal_order"}:
            bottleneck = "usable control task"
            next_action = "keep as sanity/control metric for future dense-window and model runs"
        else:
            bottleneck = "moderate task signal, still needs robustness split"
            next_action = "add session/task-family slices and bootstrap confidence intervals"
        records.append(
            {
                "task": task.get("task"),
                "display_name": summary.get("task_display_names", {}).get(task.get("task"), task.get("task")),
                "priority": TASK_PRIORITIES.get(task.get("task"), "medium"),
                "simple_status": simple_status,
                "simple_primary_metric": simple.get("primary_metric"),
                "simple_primary_score": score,
                "neural_status": neural.get("status", "not_run") if neural else "not_run",
                "neural_primary_score": neural.get("primary_score") if neural else None,
                "num_classes": classes,
                "unseen_test_class_count": unseen,
                "bottleneck": bottleneck,
                "next_action": next_action,
            }
        )
    priority_order = {"highest": 0, "high": 1, "medium": 2, "low": 3}
    return sorted(records, key=lambda item: (priority_order.get(item["priority"], 9), item["task"] or ""))


def qwen_error_summary(metrics: dict[str, Any], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    families: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "samples": 0,
        "action_exact": 0,
        "seen_samples": 0,
        "unseen_samples": 0,
        "contact_exact": 0,
        "transition_exact": 0,
    })
    object_counts: Counter[str] = Counter()
    for row in predictions:
        true_json = row.get("true_json") or {}
        pred_json = row.get("pred_json") or {}
        family = action_family(true_json.get("action") or row.get("true_label"))
        bucket = families[family]
        bucket["samples"] += 1
        bucket["action_exact"] += int((true_json.get("action") or row.get("true_label")) == pred_json.get("action"))
        bucket["seen_samples"] += int(bool(row.get("true_label_seen_in_train")))
        bucket["unseen_samples"] += int(not bool(row.get("true_label_seen_in_train")))
        bucket["contact_exact"] += int(true_json.get("contact") == pred_json.get("contact"))
        bucket["transition_exact"] += int(true_json.get("transition") == pred_json.get("transition"))
        object_counts.update(str(item).lower() for item in true_json.get("objects") or [])

    family_records = []
    for family, bucket in families.items():
        samples = bucket["samples"]
        family_records.append(
            {
                "family": family,
                "samples": samples,
                "action_exact_rate": bucket["action_exact"] / samples if samples else 0,
                "seen_share": bucket["seen_samples"] / samples if samples else 0,
                "contact_exact_rate": bucket["contact_exact"] / samples if samples else 0,
                "transition_exact_rate": bucket["transition_exact"] / samples if samples else 0,
            }
        )
    family_records.sort(key=lambda item: (-item["samples"], item["family"]))

    eval_label_counts = metrics.get("eval_label_counts") or {}
    singleton_labels = sum(1 for value in eval_label_counts.values() if value == 1)
    return {
        "run_id": metrics.get("run_id"),
        "samples": metrics.get("num_samples"),
        "json_validity_rate": metrics.get("json_validity_rate"),
        "action_macro_f1": metrics.get("action_macro_f1"),
        "subtask_accuracy": metrics.get("subtask_accuracy"),
        "next_action_accuracy": metrics.get("next_action_accuracy"),
        "contact_accuracy": metrics.get("contact_accuracy"),
        "transition_accuracy": metrics.get("transition_accuracy"),
        "object_micro_f1": metrics.get("object_micro_f1"),
        "num_unseen_label_samples": metrics.get("num_unseen_label_samples"),
        "unseen_label_sample_share": (
            metrics.get("num_unseen_label_samples") / metrics.get("num_samples")
            if metrics.get("num_samples")
            else None
        ),
        "seen_label_accuracy": metrics.get("seen_label_accuracy"),
        "unseen_label_accuracy": metrics.get("unseen_label_accuracy"),
        "eval_unique_labels": len(eval_label_counts),
        "eval_singleton_label_count": singleton_labels,
        "eval_singleton_label_share": singleton_labels / len(eval_label_counts) if eval_label_counts else None,
        "action_family_error_summary": family_records,
        "top_true_objects": [{"object": item, "count": count} for item, count in object_counts.most_common(20)],
    }


def hierarchical_contract() -> dict[str, Any]:
    return {
        "id": "xperience10m_128_hierarchical_action_targets_v1",
        "status": "ready_for_export",
        "purpose": "Reduce fine-grained label sparsity without changing the sealed 96/16/16 episode split.",
        "target_fields": [
            {
                "field": "action_family",
                "source": "normalized true action string",
                "values": [family for family, _ in ACTION_FAMILY_PATTERNS] + ["other_fine_action", "unknown"],
                "metric": "macro_f1",
            },
            {
                "field": "action_verb",
                "source": "first normalized verb phrase from action label",
                "metric": "macro_f1 with train-seen and unseen slices",
            },
            {
                "field": "fine_action",
                "source": "existing action label",
                "metric": "exact match and label-normalized semantic family match",
            },
            {
                "field": "subtask_family",
                "source": "normalized subtask phrase or main task fallback",
                "metric": "accuracy and macro_f1",
            },
            {
                "field": "contact_transition",
                "source": "existing contact and transition fields",
                "metric": "accuracy, balanced accuracy, calibration",
            },
            {
                "field": "object_set",
                "source": "existing objects list",
                "metric": "micro_f1 and object-category recall",
            },
        ],
        "public_safety": [
            "No raw MP4/HDF5/RRD files are written.",
            "No full Qwen/Cosmos weights are mirrored.",
            "Generated labels and aggregate metrics remain public-safe derived metadata.",
        ],
    }


def experiment_backlog(public_result_dir: str) -> list[dict[str, Any]]:
    return [
        {
            "id": "dense_window_export_v1",
            "priority": 1,
            "status": "ready_to_implement",
            "goal": "Create a new dense-window export over the same 128 episodes without replacing existing JSONL packages.",
            "expected_artifacts": [
                "dataset_dense_20f_stride10.jsonl",
                "dataset_dense_multiscale_manifest.json",
                "label_family_distribution.json",
            ],
            "gate": "episode ids and split assignment must exactly match the current 96/16/16 split",
        },
        {
            "id": "hierarchical_qwen3_v5",
            "priority": 2,
            "status": "ready_after_dense_export",
            "goal": "Train/evaluate Qwen3 with hierarchical action/subtask targets, constrained label options, and no-public-overwrite packaging.",
            "suggested_setup": "high-rank LoRA or partial projector/last-layer unfreeze before full-parameter tuning",
            "primary_comparison": "Qwen3 v4 action/subtask/next-action plus seen/unseen-label slices",
        },
        {
            "id": "raw_feature_unblocker_128",
            "priority": 3,
            "status": "ready_to_implement_on_training_host",
            "goal": "Export compact 128-episode raw feature shards for tasks currently marked unsupported_without_raw_128_feature_blocks.",
            "target_tasks": [
                "hand_trajectory_forecast",
                "cross_modal_retrieval",
                "modality_reconstruction",
                "misalignment_detection",
            ],
        },
        {
            "id": "cosmos3_fd_v2_multiscale",
            "priority": 4,
            "status": "ready_after_dense_export",
            "goal": "Continue Cosmos3-Super forward-dynamics with multiscale horizons and temporal consistency metrics.",
            "primary_comparison": "Cosmos3-Super Forward-Dynamics v1 validation/test MSE and rank-level loss records",
        },
        {
            "id": "robustness_and_confidence_pack",
            "priority": 5,
            "status": "ready_from_existing_outputs",
            "goal": "Add bootstrap confidence intervals, task-family slices, session slices, and random-time/random-label sanity checks.",
            "public_output": f"{public_result_dir}/robustness_pack_v1.json",
        },
    ]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fieldnames} for row in rows)


def markdown(payload: dict[str, Any]) -> str:
    dense = payload["dense_window_scenarios"]
    bottlenecks = payload["task_bottlenecks"]
    qwen = payload["qwen_v4_error_pressure"]
    selected_counts = payload["current_128_split"]["selected_episode_counts"]
    windowed_counts = payload["current_128_split"]["windowed_episode_counts"]
    split_windows = payload["current_128_split"]["split_windows"]
    selected_text = f"train {selected_counts['train']} / val {selected_counts['val']} / test {selected_counts['test']}"
    windowed_text = f"train {windowed_counts['train']} / val {windowed_counts['val']} / test {windowed_counts['test']}"
    windows_text = f"train {split_windows['train']} / val {split_windows['val']} / test {split_windows['test']}"
    lines = [
        "# 128-Episode Task Suite Enhancement Pack",
        "",
        f"Run id: `{payload['run_id']}`",
        "",
        "This non-overwriting enhancement pack records how to push the current 128-episode task suite harder without adding more raw episodes.",
        "",
        "## Current Evidence",
        "",
        f"- Current public export windows: `{payload['current_128_split']['total_windows']}`",
        f"- Window split counts: `{windows_text}`",
        f"- Selected episode split: `{selected_text}`",
        f"- Windowed episode ids in baseline CSV: `{windowed_text}`",
        f"- Qwen3 v4 JSON validity: `{qwen['json_validity_rate']:.4f}`",
        f"- Qwen3 v4 action macro-F1: `{qwen['action_macro_f1']:.6f}`",
        f"- Qwen3 v4 subtask accuracy: `{qwen['subtask_accuracy']:.6f}`",
        f"- Qwen3 v4 unseen-label sample share: `{qwen['unseen_label_sample_share']:.4f}`",
        "",
        "## Dense-Window Scenarios",
        "",
        "| scenario | estimated windows | multiplier | role |",
        "| --- | ---: | ---: | --- |",
    ]
    for item in dense:
        lines.append(
            f"| `{item['id']}` | {item['estimated_windows']} | {item['multiplier_vs_current_export']} | {item['role']} |"
        )
    lines.extend([
        "",
        "## Highest-Priority Bottlenecks",
        "",
        "| task | priority | simple score | bottleneck | next action |",
        "| --- | --- | ---: | --- | --- |",
    ])
    for item in bottlenecks[:8]:
        score = item["simple_primary_score"]
        score_text = "" if score is None else f"{score:.6f}"
        lines.append(
            f"| {item['display_name']} | {item['priority']} | {score_text} | {item['bottleneck']} | {item['next_action']} |"
        )
    lines.extend([
        "",
        "## Recommended Next Run",
        "",
        "Use `multiscale_20s10_40s20_80s40` as the next export target, then train a Qwen3 v5 hierarchical-target LoRA/partial-unfreeze run against the unchanged 96/16/16 episode split.",
        "",
        "In parallel, export compact raw 128-episode feature shards for trajectory, retrieval, reconstruction, and synchronization tasks so the simple and neural baselines can be fully aligned beyond the JSON-supported labels.",
        "",
        "The current artifacts remain the baseline; future runs should write new run ids and publish separate verified packages.",
        "",
    ])
    return "\n".join(lines)


def build_payload(run_id: str, output_dir: Path) -> dict[str, Any]:
    windows = load_windows(WINDOWS_CSV)
    baseline = read_json(BASELINE_SUMMARY)
    qwen_metrics = read_json(QWEN_V4_METRICS)
    qwen_predictions = load_jsonl(QWEN_V4_PREDICTIONS)
    cosmos_summary = read_json(COSMOS_FD_SUMMARY)
    per_episode_counts = Counter(row["episode_id"] for row in windows)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    current_split = {
        "total_windows": len(windows),
        "split_windows": split_counts(windows),
        "selected_episode_counts": baseline.get("episode_counts"),
        "windowed_episode_counts": episode_counts(windows),
        "unique_main_tasks": len({row["main_task"] for row in windows}),
        "windows_per_episode": {
            "min": min(per_episode_counts.values()),
            "median": statistics.median(per_episode_counts.values()),
            "max": max(per_episode_counts.values()),
        },
    }
    public_result_dir = output_dir.relative_to(ROOT).as_posix()
    return {
        "title": "Ropedia Xperience-10M 128-Episode Task Suite Enhancement Pack",
        "status": "pass",
        "run_id": run_id,
        "generated_at_utc": now,
        "scope": "No-new-episode enhancement plan over the current selected 128-episode 96/16/16 split.",
        "current_128_split": current_split,
        "dense_window_scenarios": dense_window_estimates(windows),
        "hierarchical_target_contract": hierarchical_contract(),
        "task_bottlenecks": task_bottlenecks(baseline),
        "qwen_v4_error_pressure": qwen_error_summary(qwen_metrics, qwen_predictions),
        "cosmos3_super_forward_dynamics_reference": {
            "status": cosmos_summary.get("status"),
            "run_id": cosmos_summary.get("run_id"),
            "train_rows": cosmos_summary.get("train_rows"),
            "val_rows": cosmos_summary.get("val_rows"),
            "test_rows": cosmos_summary.get("test_rows"),
            "test_mse": cosmos_summary.get("test_mse"),
            "adapter_parameter_numel": cosmos_summary.get("adapter_parameter_numel"),
        },
        "experiment_backlog": experiment_backlog(public_result_dir),
        "public_artifacts": {
            "result_dir": public_result_dir,
            "public_json": "docs/data/task_suite_enhancement_128.json",
            "public_markdown": "TASK_SUITE_ENHANCEMENT_128.md",
        },
        "non_overwrite_policy": {
            "result_directory_created_once": True,
            "stable_public_summaries_update_to_latest_enhancement_pack": True,
            "prior_model_result_packages_overwritten": False,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--force", action="store_true", help="allow rewriting the run directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_root / args.run_id
    if output_dir.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite existing enhancement run: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=args.force)

    payload = build_payload(args.run_id, output_dir)
    report_md = markdown(payload)
    write_json(output_dir / "enhancement_plan.json", payload)
    write_json(output_dir / "hierarchical_target_contract.json", payload["hierarchical_target_contract"])
    write_json(output_dir / "experiment_backlog.json", {"status": "pass", "experiments": payload["experiment_backlog"]})
    write_json(PUBLIC_JSON, payload)
    (output_dir / "ENHANCEMENT_REPORT.md").write_text(report_md, encoding="utf-8")
    PUBLIC_MD.write_text(report_md, encoding="utf-8")

    write_csv(
        output_dir / "dense_window_scenarios.csv",
        payload["dense_window_scenarios"],
        ["id", "estimated_windows", "multiplier_vs_current_export", "window_frames", "stride_frames", "role", "source_note"],
    )
    write_csv(
        output_dir / "task_bottlenecks.csv",
        payload["task_bottlenecks"],
        [
            "task",
            "display_name",
            "priority",
            "simple_status",
            "simple_primary_metric",
            "simple_primary_score",
            "neural_status",
            "neural_primary_score",
            "num_classes",
            "unseen_test_class_count",
            "bottleneck",
            "next_action",
        ],
    )
    write_csv(
        output_dir / "qwen_action_family_error_summary.csv",
        payload["qwen_v4_error_pressure"]["action_family_error_summary"],
        ["family", "samples", "action_exact_rate", "seen_share", "contact_exact_rate", "transition_exact_rate"],
    )
    print(f"PASS: wrote {output_dir / 'enhancement_plan.json'}")
    print(f"PASS: wrote {PUBLIC_JSON}")
    print(f"PASS: wrote {PUBLIC_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

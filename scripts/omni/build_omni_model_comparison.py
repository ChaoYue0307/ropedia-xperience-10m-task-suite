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

QWEN_RUN_PRIORITY = {
    "xperience10m_qwen3_omni_128ep_multiscale_cap96_v5_full8gpu_lora_eval_test_full": 500,
    "xperience10m_qwen3_omni_128ep_structured_json_v4_4epoch_full8gpu_lora_eval_test_full": 400,
    "xperience10m_qwen3_omni_128ep_structured_json_v3_strict_label_prompt_reuse_lora_eval_test_full": 300,
    "xperience10m_qwen3_omni_128ep_structured_json_v2_reuse_full8gpu_lora_eval_test_full": 200,
    "xperience10m_qwen3_omni_128ep_fullsplit_fast8gpu_lora_fsdp_full_train_noval_tail_logits_fullstatesave_v6_eval_test_full": 100,
    "xperience10m_qwen3_omni_128ep_96train_16val_16test_valmon_20260605_eval": 50,
}
QWEN_V5_EVAL_RUN_ID = "xperience10m_qwen3_omni_128ep_multiscale_cap96_v5_full8gpu_lora_eval_test_full"

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
    cosmos_nano = [item for item in branches if item.get("backbone") == "cosmos_world_model"]
    cosmos_super = [
        item
        for item in branches
        if item.get("backbone") in {"cosmos3_super_reasoner", "cosmos3_super_forward_dynamics"}
    ]
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
            "cosmos3_verified_package_count": len(cosmos_nano) + len(cosmos_super),
            "cosmos3_nano_verified_package_count": len(cosmos_nano),
            "cosmos3_super_verified_package_count": len(cosmos_super),
        },
        "models": [
            "Qwen3-Omni LoRA",
            "Cosmos3-Nano future-window compatibility branch",
            "Cosmos3-Super Reasoner base-weight evaluation",
            "Cosmos3-Super forward-dynamics LoRA",
        ],
        "branches": branches,
        "interpretation": (
            "This layer contains the held-out foundation-model packages. Qwen3-Omni "
            "packages evaluate structured JSON task prediction; Cosmos3-Nano evaluates "
            "a future-window world-model compatibility adapter; Cosmos3-Super Reasoner "
            "evaluates staged base weights through vLLM on the JSON task; Cosmos3-Super "
            "Forward-Dynamics LoRA is the first Super adapter branch and evaluates "
            "camera-pose-conditioned future vision velocity loss."
        ),
    }


def qwen_current_rank(branch: dict[str, Any]) -> tuple[int, float, str]:
    branch_id = str(branch.get("id") or "")
    metrics = branch.get("primary_metrics", {}) if isinstance(branch.get("primary_metrics"), dict) else {}
    json_validity = metrics.get("json_validity_rate")
    return (
        QWEN_RUN_PRIORITY.get(branch_id, 0),
        float(json_validity) if isinstance(json_validity, (int, float)) else -1.0,
        branch_id,
    )


def qwen3_smoke_entry() -> dict[str, Any]:
    path = ROOT / "results/omni_exploration/qwen3_adapter_smoke/metrics.json"
    metrics = load_json(path)
    if not metrics:
        return {
            "id": "qwen3_omni_sensor_adapter_smoke_1ep",
            "title": "Qwen3-Omni Sensor-Adapter Smoke",
            "scope": "one public Xperience-10M sample episode",
            "status": "missing",
            "source": rel(path),
            "weights": "none",
            "interpretation": "Expected readiness entry, but the local metrics file is missing.",
        }
    return {
        "id": "qwen3_omni_sensor_adapter_smoke_1ep",
        "title": "Qwen3-Omni Sensor-Adapter Smoke",
        "scope": "one public Xperience-10M sample episode",
        "status": "verified_smoke",
        "source": rel(path),
        "split": metrics.get("split"),
        "counts": {
            "episodes": metrics.get("num_episodes"),
            "windows": metrics.get("num_windows"),
            "train_windows": metrics.get("num_train_windows"),
            "test_windows": metrics.get("num_test_windows"),
            "feature_dim": metrics.get("feature_dim"),
            "adapter_tokens": metrics.get("num_adapter_tokens"),
        },
        "primary_metrics": {
            "accuracy": metrics.get("accuracy"),
            "macro_f1": metrics.get("macro_f1"),
            "train_final_loss": metrics.get("train_final_loss"),
        },
        "base_model_target": metrics.get("base_model_target"),
        "qwen3_loaded": metrics.get("qwen3_loaded"),
        "weights": "no Qwen3 base weights or LoRA adapter weights; adapter-token readiness smoke only",
        "interpretation": (
            "This validates the sensor-adapter token path on one real episode before "
            "loading or LoRA-tuning Qwen3-Omni. It is not comparable to the 128-episode "
            "held-out LoRA result."
        ),
    }


def qwen_full_parameter_gate_entries() -> list[dict[str, Any]]:
    path = ROOT / "docs/data/qwen3_full_parameter_gates.json"
    payload = load_json(path)
    rows = payload.get("runs", []) if isinstance(payload.get("runs"), list) else []
    entries = []
    for row in rows:
        status = row.get("status", "unknown")
        entries.append(
            {
                "id": row.get("run_id") or row.get("id"),
                "title": row.get("title"),
                "scope_label": "full-param gate",
                "scope": row.get("scope"),
                "status": status,
                "source": row.get("summary_path") or rel(path),
                "split": "selected 128-episode train split",
                "counts": {
                    "samples": row.get("num_train_samples"),
                    "steps": row.get("observed_train_steps"),
                    "num_processes": row.get("num_processes"),
                },
                "primary_metrics": {
                    "full_parameter_gate": status,
                    "observed_train_steps": row.get("observed_train_steps"),
                    "final_step_loss": row.get("final_step_loss"),
                    "epoch_train_loss": row.get("epoch_train_loss"),
                    "checkpoint_saved": row.get("checkpoint_saved"),
                },
                "weights": row.get("checkpoint_policy"),
                "interpretation": (
                    "Full-parameter FSDP feasibility evidence only. This gate is not a "
                    "held-out model result, full fine-tune, checkpoint release, or public "
                    "weight package."
                ),
            }
        )
    return entries


def cosmos3_super_readiness_entry() -> dict[str, Any] | None:
    paths = [
        path
        for path in sorted(
            (ROOT / "results/omni_finetune").glob(
                "xperience10m_cosmos3_super_training_readiness_*/training_readiness.json"
            )
        )
        if "metadata_a100" not in path.parent.name
    ]
    if not paths:
        return None
    payloads = [(path, load_json(path)) for path in paths]
    path, payload = max(payloads, key=lambda item: item[1].get("finished_at_unix") or 0)
    decision = payload.get("decision", {}) if isinstance(payload.get("decision"), dict) else {}
    dataset = payload.get("dataset", {}) if isinstance(payload.get("dataset"), dict) else {}
    return {
        "id": payload.get("run_id", path.parent.name),
        "title": "Cosmos3-Super Training Readiness Probe",
        "scope": "selected 128-episode 96/16/16 JSON-task dataset and staged Cosmos3-Super runtime",
        "status": decision.get("status", "unknown"),
        "source": rel(path),
        "split": "train/val/test by selected episode/session",
        "counts": {
            "dataset_samples": dataset.get("total_samples"),
            "split_counts": dataset.get("split_summary"),
        },
        "primary_metrics": {
            "diffusers_runtime_supported": decision.get("diffusers_runtime_supported"),
            "chat_sft_supported": decision.get("chat_sft_supported"),
            "weights_updated": decision.get("weights_updated"),
        },
        "weights": "none; readiness audit only, no adapter checkpoint",
        "interpretation": (
            "This probe confirms the staged Cosmos3-Super Diffusers/GPU runtime and "
            "the same JSON QA dataset are visible. It predates the camera-pose action-target "
            "export, so use the 20260608 contract audit for the current trainer-readiness status."
        ),
    }


def cosmos3_super_staging_readiness_entry() -> dict[str, Any] | None:
    paths = sorted(
        (ROOT / "results/omni_finetune").glob(
            "xperience10m_cosmos3_super_training_readiness_metadata_a100_*/training_readiness.json"
        )
    )
    if not paths:
        return None
    payloads = [(path, load_json(path)) for path in paths]
    path, payload = max(payloads, key=lambda item: item[1].get("finished_at_unix") or 0)
    decision = payload.get("decision", {}) if isinstance(payload.get("decision"), dict) else {}
    dataset = payload.get("dataset", {}) if isinstance(payload.get("dataset"), dict) else {}
    model = payload.get("model", {}) if isinstance(payload.get("model"), dict) else {}
    runtime = payload.get("runtime", {}) if isinstance(payload.get("runtime"), dict) else {}
    return {
        "id": payload.get("run_id", path.parent.name),
        "title": "Cosmos3-Super Remote Staging Readiness Probe",
        "scope_label": "staging readiness",
        "scope": "secondary 4-GPU staging tree, JSON-task dataset visibility, and metadata-only Cosmos3-Super runtime probe",
        "status": decision.get("status", "unknown"),
        "source": rel(path),
        "split": "train/val/test by selected episode/session",
        "counts": {
            "dataset_samples": dataset.get("total_samples"),
            "split_counts": dataset.get("split_summary"),
        },
        "primary_metrics": {
            "model_files_visible": model.get("exists"),
            "diffusers_runtime_supported": decision.get("diffusers_runtime_supported"),
            "cuda_device_count": runtime.get("cuda_device_count"),
            "weights_updated": decision.get("weights_updated"),
        },
        "weights": "none; staging readiness audit only, no adapter checkpoint",
        "interpretation": (
            "This metadata-only probe checks the secondary 4-GPU staging tree without "
            "loading the model pipeline or updating weights. It confirms the JSON task "
            "dataset is present, but the Cosmos3-Super model files and Diffusers runtime "
            "are not staged there yet, so real Super training should wait for model/runtime "
            "staging or run on the already prepared main host."
        ),
    }


def cosmos3_super_action_contract_entry() -> dict[str, Any] | None:
    paths = sorted(
        (ROOT / "results/omni_finetune").glob(
            "xperience10m_cosmos3_super_training_contract_audit_*/training_contract_audit.json"
        )
    )
    if not paths:
        return None
    payloads = [(path, load_json(path)) for path in paths]
    path, payload = max(payloads, key=lambda item: item[1].get("finished_at_unix") or 0)
    decision = payload.get("decision", {}) if isinstance(payload.get("decision"), dict) else {}
    dataset = payload.get("dataset", {}) if isinstance(payload.get("dataset"), dict) else {}
    target_modes = dataset.get("target_mode_counts", {}) if isinstance(dataset.get("target_mode_counts"), dict) else {}
    only_forward_dynamics = set(target_modes) == {"forward_dynamics"}
    return {
        "id": payload.get("run_id", path.parent.name),
        "title": "Cosmos3-Super Camera-Pose Target Audit",
        "scope_label": "action target contract",
        "scope": "selected 128-episode 96/16/16 dataset augmented with camera_pose proxy cosmos_action_target records",
        "status": "ready_for_forward_dynamics_trainer" if only_forward_dynamics else "ready_for_action_lora_trainer" if decision.get("status") == "ready_for_cosmos3_super_action_lora" else decision.get("status", "unknown"),
        "source": rel(path),
        "split": "train/val/test by selected episode/session",
        "counts": {
            "dataset_samples": dataset.get("num_rows"),
            "rows_with_action_target": dataset.get("rows_with_action_target"),
            "valid_action_targets": dataset.get("valid_action_targets"),
            "split_counts": dataset.get("split_counts"),
            "episode_split_counts": dataset.get("episode_split_counts"),
        },
        "primary_metrics": {
            "domain_name": "camera_pose",
            "raw_action_dim": 9,
            "mode": next(iter(target_modes), "forward_dynamics"),
            "valid_action_targets": dataset.get("valid_action_targets"),
            "weights_updated": decision.get("weights_updated"),
        },
        "weights": "none; action-target contract audit only, no adapter checkpoint",
        "interpretation": (
            "The selected dataset now has valid Cosmos3 camera_pose forward_dynamics targets "
            "for an egocentric camera-motion proxy. These remove the target-schema blocker "
            "for action-conditioned world-model training, but they supervise noisy vision "
            "tokens rather than preds_action. The remaining work is a trainable "
            "Cosmos3-Super implementation that can backpropagate through this loss "
            "surface at the required memory scale; action-token prediction needs a "
            "separate policy or inverse-dynamics target export."
        ),
    }


def cosmos3_super_packer_entry() -> dict[str, Any] | None:
    paths = sorted(
        (ROOT / "results/omni_finetune").glob("xperience10m_cosmos3_super_action_packer_*/packer_summary.json")
    )
    if not paths:
        return None
    payloads = [(path, load_json(path)) for path in paths]
    path, payload = max(payloads, key=lambda item: item[1].get("finished_at_unix") or 0)
    row_contract = payload.get("row_contract", {}) if isinstance(payload.get("row_contract"), dict) else {}
    pack_result = payload.get("pack_result", {}) if isinstance(payload.get("pack_result"), dict) else {}
    return {
        "id": payload.get("run_id", path.parent.name),
        "title": "Cosmos3-Super Action Batch Packer Smoke",
        "scope_label": "batch packer",
        "scope": "one selected train row from the camera_pose forward_dynamics augmented JSONL",
        "status": payload.get("status", "unknown"),
        "source": rel(path),
        "split": row_contract.get("split"),
        "counts": {
            "samples": 1,
            "raw_action_rows": (row_contract.get("raw_actions_shape") or [None, None])[0],
            "raw_action_dim": row_contract.get("raw_action_dim"),
        },
        "primary_metrics": {
            "mode": row_contract.get("mode"),
            "loss_surface": row_contract.get("loss_surface"),
            "pipeline_loaded": pack_result.get("pipeline_loaded"),
            "weights_updated": payload.get("weights_updated"),
        },
        "weights": "none; schema-only packer smoke, no adapter checkpoint",
        "interpretation": (
            "The selected row maps to a camera_pose forward_dynamics contract. In the installed Cosmos3 pipeline this "
            "uses raw actions as conditioning and supervises noisy vision tokens; it does not supervise preds_action."
        ),
    }


def run_entry_from_version(version: dict[str, Any], *, run_id: str, weights: str, interpretation: str) -> dict[str, Any]:
    return {
        "id": run_id,
        "title": version.get("title"),
        "scope": version.get("scope"),
        "status": version.get("status"),
        "source": version.get("source"),
        "split": version.get("split"),
        "counts": version.get("counts", {}),
        "weights": weights,
        "interpretation": interpretation,
    }


def model_grouped_view(versions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    single_episode = versions[0]
    aligned_128 = versions[1]
    branch_version = versions[2]
    branches = branch_version.get("branches", [])
    qwen_branches = [branch for branch in branches if branch.get("backbone") == "qwen3_omni_lora"]
    cosmos_nano_branches = [branch for branch in branches if branch.get("backbone") == "cosmos_world_model"]
    cosmos_super_branches = [branch for branch in branches if branch.get("backbone") == "cosmos3_super_reasoner"]
    cosmos_super_fd_branches = [branch for branch in branches if branch.get("backbone") == "cosmos3_super_forward_dynamics"]
    qwen_full_parameter_gates = qwen_full_parameter_gate_entries()
    cosmos_super_readiness = cosmos3_super_readiness_entry()
    cosmos_super_staging_readiness = cosmos3_super_staging_readiness_entry()
    cosmos_super_action_contract = cosmos3_super_action_contract_entry()
    cosmos_super_packer = cosmos3_super_packer_entry()
    if qwen_branches:
        current_qwen = max(qwen_branches, key=qwen_current_rank)
        for branch in qwen_branches:
            branch["is_current"] = branch.get("id") == current_qwen.get("id")
            branch["weights_repository"] = (
                "https://huggingface.co/cy0307/ropedia-qwen3-omni-lora-128ep"
                if branch["is_current"]
                else "historical diagnostic package; keep separate from the final 128-episode adapter repo"
            )
    for branch in cosmos_nano_branches:
        branch["is_current"] = True
        branch["weights_repository"] = (
            "planned separate Cosmos3 model repo after a real Cosmos diffusion/LoRA "
            "fine-tune exists; current result remains artifacts-only"
        )
    for branch in cosmos_super_branches:
        branch["is_current"] = True
        branch["weights_repository"] = (
            "none for this run: staged base nv-community/Cosmos3-Super weights were "
            "evaluated through vLLM; create a separate repo only after new adapter or "
            "fine-tuned weights exist"
        )
    for branch in cosmos_super_fd_branches:
        branch["is_current"] = True
        branch["weights_repository"] = "https://huggingface.co/cy0307/ropedia-cosmos3-super-forward-dynamics-lora-128ep"
    return [
        {
            "id": "task_head_baselines",
            "model_family": "Minimal and Neural Task Heads",
            "model_type": "lightweight supervised/self-supervised task heads",
            "weight_repository": "https://huggingface.co/cy0307/ropedia-xperience-10m-task-baselines",
            "one_episode_runs": [
                run_entry_from_version(
                    single_episode,
                    run_id="task_heads_single_episode_public_sample",
                    weights="baseline model files in the baseline model repo; no foundation-model weights",
                    interpretation="Raw multimodal feature task harness on the public sample.",
                )
            ],
            "multi_episode_128_runs": [
                run_entry_from_version(
                    aligned_128,
                    run_id="task_heads_128_episode_metadata_baselines",
                    weights="metadata/text baseline artifacts; raw 128 sensor-feature model weights not yet complete",
                    interpretation="Same selected 96/16/16 split and task ids as the model branches, but metadata/text features only.",
                )
            ],
            "comparison_note": (
                "This is the cleanest 1-episode versus 128-episode grouping for the "
                "same simple/NN task-head family, but the feature surface changes from "
                "raw public-sample features to public-safe 128-episode metadata/text features."
            ),
        },
        {
            "id": "qwen3_omni_lora",
            "model_family": "Qwen3-Omni LoRA",
            "model_type": "PEFT LoRA adapter over Qwen/Qwen3-Omni-30B-A3B-Instruct",
            "weight_repository": "https://huggingface.co/cy0307/ropedia-qwen3-omni-lora-128ep",
            "one_episode_runs": [qwen3_smoke_entry()],
            "readiness_runs": qwen_full_parameter_gates,
            "multi_episode_128_runs": qwen_branches,
            "comparison_note": (
                "The one-episode Qwen entry is only a sensor-adapter smoke test with "
                "Qwen3 weights unloaded. The 128-episode entries are real held-out LoRA "
                "diagnostics; the current final adapter belongs in the separate Qwen model repo. "
                "The full-parameter rows are feasibility gates only and intentionally publish "
                "no checkpoints or full-parameter weights."
            ),
        },
        {
            "id": "cosmos3_nano_world_model",
            "model_family": "Cosmos3-Nano Future-Window World Model",
            "model_type": "world-model/future-window branch",
            "weight_repository": "planned: cy0307/ropedia-cosmos3-nano-future-window-lora-128ep after real adapter weights exist",
            "one_episode_runs": [
                {
                    "id": "cosmos3_nano_one_episode",
                    "title": "Cosmos3-Nano One-Episode Fine-Tune",
                    "scope": "one public Xperience-10M sample episode",
                    "status": "not_run",
                    "source": None,
                    "weights": "none",
                    "interpretation": (
                        "No Cosmos3 one-episode adapter or diffusion-weight fine-tune is currently published. "
                        "Use the public-sample task suite only as model-agnostic evidence."
                    ),
                }
            ],
            "multi_episode_128_runs": cosmos_nano_branches,
            "comparison_note": (
                "The current 128-episode Cosmos result is a public-safe future-window "
                "compatibility adapter. It is not yet a full Cosmos diffusion/LoRA weight release."
            ),
        },
        {
            "id": "cosmos3_super_reasoner",
            "model_family": "Cosmos3-Super Reasoner",
            "model_type": "base-weight vLLM Reasoner evaluation over nv-community/Cosmos3-Super",
            "weight_repository": "none for this run; staged base weights only, no new fine-tuned weights",
            "one_episode_runs": [
                {
                    "id": "cosmos3_super_one_episode",
                    "title": "Cosmos3-Super One-Episode Fine-Tune",
                    "scope": "one public Xperience-10M sample episode",
                    "status": "not_run",
                    "source": None,
                    "weights": "none",
                    "interpretation": (
                        "No one-episode Cosmos3-Super adapter or fine-tuned weight run is published. "
                        "The available Super result is the 128-episode held-out base-weight evaluation."
                    ),
                }
            ],
            "readiness_runs": [
                entry
                for entry in (
                    cosmos_super_readiness,
                    cosmos_super_staging_readiness,
                    cosmos_super_action_contract,
                    cosmos_super_packer,
                )
                if entry
            ],
            "multi_episode_128_runs": cosmos_super_branches,
            "comparison_note": (
                "Cosmos3-Super is now represented by a verified 448-window held-out "
                "Reasoner evaluation on the same JSON task as Qwen3. It uses staged base "
                "weights through vLLM, so it is a model-branch diagnostic, not a weight release. "
                "A camera-pose proxy forward-dynamics target export now passes the contract audit "
                "and schema-only packer smoke; the separate Forward-Dynamics LoRA group records "
                "the trainable adapter run and loss-based held-out evaluation."
            ),
        },
        {
            "id": "cosmos3_super_forward_dynamics",
            "model_family": "Cosmos3-Super Forward-Dynamics LoRA",
            "model_type": "PEFT LoRA over nv-community/Cosmos3-Super for camera-pose-conditioned future vision velocity",
            "weight_repository": "https://huggingface.co/cy0307/ropedia-cosmos3-super-forward-dynamics-lora-128ep",
            "one_episode_runs": [
                {
                    "id": "cosmos3_super_forward_dynamics_overfit_smoke",
                    "title": "Cosmos3-Super Forward-Dynamics Overfit Smoke",
                    "scope": "small overfit smoke before 128-episode scale-up",
                    "status": "verified_smoke",
                    "source": "results/omni_finetune/xperience10m_cosmos3_super_forward_dynamics_lora_overfit_after_qwen_v4_20260608_fsdp8_attn256_gradfix_savefix2/",
                    "weights": "local repaired LoRA smoke adapter, not public packaged as final",
                    "interpretation": (
                        "Validated the trainable adapter path, FSDP save repair, and Diffusers load before the full 128-episode run."
                    ),
                }
            ],
            "multi_episode_128_runs": cosmos_super_fd_branches,
            "comparison_note": (
                "This is the first verified Cosmos3-Super fine-tuned adapter branch. "
                "Its metric is forward-dynamics MSE, so compare it to world-model loss "
                "or future-prediction targets, not to Qwen JSON classification accuracy."
            ),
        },
    ]


def build_report() -> dict[str, Any]:
    versions = [single_episode_summary(), aligned_baseline_summary(), model_branch_summary()]
    model_groups = model_grouped_view(versions)
    qwen_branch_ids = {
        str(branch.get("id"))
        for branch in versions[2].get("branches", [])
        if branch.get("backbone") == "qwen3_omni_lora"
    }
    if QWEN_V5_EVAL_RUN_ID in qwen_branch_ids:
        pending = [
            "Use the verified Qwen3 v5 dense multiscale full-eval package as the current Qwen row; older Qwen package rows remain historical diagnostics for comparison.",
        ]
    else:
        pending = [
            "Use the verified Qwen3 v4 4-epoch full-eval package as the current Qwen row; older Qwen package rows remain historical diagnostics for comparison.",
        ]
        pending.append(
            "Complete the Qwen3-Omni v5 dense multiscale raw-media export, all-GPU LoRA train, held-out eval, and public package before promoting it over the current Qwen v4 row."
        )
    return {
        "title": "Ropedia Xperience-10M Current Result Versions and Model Groups",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "pass",
        "version_count": len(versions),
        "model_group_count": len(model_groups),
        "comparison_rule": (
            "Compare only rows with the same scope and target. Single-episode raw-feature "
            "metrics, 128-episode metadata baselines, Qwen3 structured JSON metrics, and "
            "the two Cosmos3 targets answer different questions: Nano future-window retrieval "
            "versus Super structured JSON Reasoner evaluation."
        ),
        "version_reading_notes": [
            "Version 1 is the public-sample 12-task harness with minimal and neural heads.",
            "Version 2 is the selected 128-episode same-split simple/NN baseline alignment.",
            "Version 3 is the verified model-branch layer: the current final Qwen3-Omni LoRA package is the JSON-task diagnostic result, Cosmos3-Nano is a future-window compatibility result, Cosmos3-Super Reasoner is a base-weight JSON-task evaluation, and Cosmos3-Super Forward-Dynamics LoRA is the first Super fine-tuned adapter branch.",
        ],
        "versions": versions,
        "model_groups": model_groups,
        "model_group_reading_notes": [
            "Use model_groups when comparing one-episode and 128-episode artifacts within the same model family.",
            "Task-head baselines have both a one-episode public-sample run and a 128-episode same-split metadata/text run.",
            "Qwen3-Omni has a one-episode sensor-adapter smoke test, full-parameter feasibility gates, and separate 128-episode LoRA diagnostic packages; the newest verified full-eval 128-episode adapter belongs in the Qwen LoRA model repo.",
            "Cosmos3-Nano has a 128-episode future-window compatibility package.",
            "Cosmos3-Super now has both a 128-episode base-weight Reasoner evaluation on the JSON task and a fine-tuned forward-dynamics LoRA branch over camera-pose proxy targets.",
        ],
        "pending": pending,
    }


def fmt_score(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def entry_count_text(entry: dict[str, Any]) -> str:
    counts = entry.get("counts", {}) if isinstance(entry.get("counts"), dict) else {}
    pieces = []
    for label, keys in (
        ("episodes", ("episodes", "dataset_episodes", "held_out_episode_count")),
        ("windows/samples", ("windows", "rows", "dataset_samples", "eval_samples", "samples")),
        ("eval", ("eval_samples",)),
    ):
        value = next((counts.get(key) for key in keys if counts.get(key) is not None), None)
        if value is not None:
            pieces.append(f"{value} {label}")
    return ", ".join(pieces)


def entry_metric_text(entry: dict[str, Any]) -> str:
    metrics = entry.get("primary_metrics", {}) if isinstance(entry.get("primary_metrics"), dict) else {}
    if not metrics:
        return ""
    keep = [
        "json_validity_rate",
        "action_macro_f1",
        "future_retrieval_mrr",
        "test_forward_dynamics_mse",
        "val_forward_dynamics_mse",
        "train_final_loss",
        "adapter_parameter_numel",
        "temporal_consistency",
        "transition_accuracy",
        "contact_accuracy",
        "accuracy",
        "macro_f1",
        "domain_name",
        "raw_action_dim",
        "mode",
        "valid_action_targets",
        "loss_surface",
        "pipeline_loaded",
        "diffusers_runtime_supported",
        "chat_sft_supported",
        "weights_updated",
        "full_parameter_gate",
        "observed_train_steps",
        "final_step_loss",
        "epoch_train_loss",
        "checkpoint_saved",
    ]
    return ", ".join(f"{key}={fmt_score(metrics[key])}" for key in keep if key in metrics)


def append_model_group(lines: list[str], group: dict[str, Any]) -> None:
    lines.extend(
        [
            "",
            f"### {group['model_family']}",
            "",
            group.get("comparison_note", ""),
            "",
            f"- Weight repo policy: {group.get('weight_repository')}",
            "",
            "| scope | status | run | counts | metrics | source |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    rows = []
    for entry in group.get("one_episode_runs", []):
        rows.append(("1 episode", entry))
    for entry in group.get("readiness_runs", []):
        rows.append((entry.get("scope_label", "readiness"), entry))
    for entry in group.get("multi_episode_128_runs", []):
        rows.append(("128 episode", entry))
    for scope, entry in rows:
        source = entry.get("source")
        source_text = "" if source in (None, "") else f"`{source}`"
        current = " current" if entry.get("is_current") else ""
        lines.append(
            "| {scope} | {status}{current} | {title} | {counts} | {metrics} | {source} |".format(
                scope=scope,
                status=entry.get("status", ""),
                current=current,
                title=entry.get("title") or entry.get("id"),
                counts=entry_count_text(entry),
                metrics=entry_metric_text(entry),
                source=source_text,
            )
        )


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
    lines.extend(["", "## Model-Family Grouped View", ""])
    lines.extend(f"- {item}" for item in report.get("model_group_reading_notes", []))
    for group in report.get("model_groups", []):
        append_model_group(lines, group)
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
            if key
            in {
                "json_validity_rate",
                "action_macro_f1",
                "future_retrieval_mrr",
                "test_forward_dynamics_mse",
                "val_forward_dynamics_mse",
                "train_final_loss",
                "adapter_parameter_numel",
                "temporal_consistency",
                "transition_accuracy",
                "contact_accuracy",
            }
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

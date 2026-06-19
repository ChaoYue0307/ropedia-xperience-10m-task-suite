#!/usr/bin/env python3
"""Build a unified 20-task radar chart for baseline and model-branch metrics."""

from __future__ import annotations

import html
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TASK_SUITE_PATH = ROOT / "docs/data/task_suite_20.json"
QWEN_V6_METRICS_PATH = (
    ROOT
    / "results/omni_finetune/verified_public"
    / "xperience10m_qwen3_omni_128ep_multiscale_cap96_v6_rank64_lr5e5_full8gpu_lora_eval_test_full"
    / "eval/metrics.json"
)
COSMOS_SUPER_REASONER_METRICS_PATH = (
    ROOT
    / "results/omni_finetune/verified_public"
    / "xperience10m_cosmos3_super_reasoner_128ep_test_full_20260607"
    / "eval/metrics.json"
)
COSMOS_NANO_METRICS_PATH = (
    ROOT
    / "results/omni_finetune/verified_public"
    / "xperience10m_cosmos3_nano_128ep_future_window_h5_compat_adapter_eval_test_full"
    / "eval/metrics.json"
)
COSMOS_SUPER_FD_METRICS_PATH = (
    ROOT
    / "results/omni_finetune/verified_public"
    / "xperience10m_cosmos3_super_forward_dynamics_lora_128ep_train1epoch_256_attn_full8gpu_20260608_eval_test_full_fsdp"
    / "eval/metrics.json"
)
METADATA128_BASELINE_DIR = ROOT / "results/omni_finetune/a100_128_metadata_task_baselines_20260616_v2"
RAW128_BASELINE_DIR = ROOT / "results/omni_finetune/a100_128_raw20_task_baselines_complete20_proxy_20260616T091500Z"
MODEL_OUTPUT_TASK_PROBE_DIR = ROOT / "results/omni_finetune/model_output_task_probes_20260616"
QWEN_FUTURE_TASK_PROBE_DIR = (
    ROOT
    / "results/omni_finetune"
    / "xperience10m_qwen3_omni_v6_future_task_probes_a100_20260616T143608Z"
)
QWEN_ORDER_SYNC_TIME_PROBE_DIR = (
    ROOT
    / "results/omni_finetune"
    / "xperience10m_qwen3_omni_v6_order_sync_time_probes_a100_20260617T132500Z"
)
QWEN_RETRIEVAL_TASK_PROBE_DIR = (
    ROOT
    / "results/omni_finetune"
    / "xperience10m_qwen3_omni_v6_retrieval_task_probes_a100_20260617T175919Z"
)
QWEN_CROSS_MODAL_RETRIEVAL_PROBE_DIR = (
    ROOT
    / "results/omni_finetune"
    / "xperience10m_qwen3_omni_v6_cross_modal_retrieval_probe_a100_20260618T000000Z"
)
QWEN_CAMERA_VIEW_SYNC_PROBE_DIR = (
    ROOT
    / "results/omni_finetune"
    / "xperience10m_qwen3_omni_v6_camera_view_sync_mosaic_tile_a100_20260619T0305Z"
)
QWEN_SENSOR_TARGET_PROBE_DIR = (
    ROOT
    / "results/omni_finetune"
    / "xperience10m_qwen3_omni_v6_sensor_target_probes_a100_20260619T000000Z"
)
COSMOS_SUPER_RETRIEVAL_TASK_PROBE_DIR = (
    ROOT
    / "results/omni_finetune"
    / "xperience10m_cosmos3_super_retrieval_task_probes_a100_20260619T000000Z"
)
QWEN_ACTION_OBJECT_METRICS_PATH = (
    MODEL_OUTPUT_TASK_PROBE_DIR / "action_object_relation/qwen3_omni_v6_lora/metrics.json"
)
COSMOS_SUPER_ACTION_OBJECT_METRICS_PATH = (
    MODEL_OUTPUT_TASK_PROBE_DIR / "action_object_relation/cosmos3_super_reasoner/metrics.json"
)
COSMOS_SUPER_CAPTION_GROUNDING_METRICS_PATH = (
    MODEL_OUTPUT_TASK_PROBE_DIR / "caption_grounding/cosmos3_super_reasoner/metrics.json"
)
COSMOS_SUPER_TIME_TO_TRANSITION_METRICS_PATH = (
    MODEL_OUTPUT_TASK_PROBE_DIR / "time_to_transition/cosmos3_super_reasoner/metrics.json"
)
COSMOS_SUPER_LONG_HORIZON_METRICS_PATH = (
    MODEL_OUTPUT_TASK_PROBE_DIR / "long_horizon_next_action/cosmos3_super_reasoner/metrics.json"
)
COSMOS_NANO_LONG_HORIZON_METRICS_PATH = (
    MODEL_OUTPUT_TASK_PROBE_DIR / "long_horizon_next_action/cosmos3_nano_future_window/metrics.json"
)
COSMOS_NANO_NEXT_SUBTASK_METRICS_PATH = (
    MODEL_OUTPUT_TASK_PROBE_DIR / "next_subtask_forecast/cosmos3_nano_future_window/metrics.json"
)
COSMOS_NANO_MODALITY_RECONSTRUCTION_METRICS_PATH = (
    MODEL_OUTPUT_TASK_PROBE_DIR / "modality_reconstruction/cosmos3_nano_future_window/metrics.json"
)
COSMOS_NANO_OBJECT_SET_METRICS_PATH = (
    MODEL_OUTPUT_TASK_PROBE_DIR / "object_set_forecast/cosmos3_nano_future_window/metrics.json"
)
COSMOS_NANO_ACTION_OBJECT_METRICS_PATH = (
    MODEL_OUTPUT_TASK_PROBE_DIR / "action_object_relation/cosmos3_nano_future_window/metrics.json"
)
COSMOS_NANO_TIME_TO_TRANSITION_METRICS_PATH = (
    MODEL_OUTPUT_TASK_PROBE_DIR / "time_to_transition/cosmos3_nano_future_window/metrics.json"
)
QWEN_FUTURE_TASK_METRIC_PATHS = {
    "caption_grounding": QWEN_RETRIEVAL_TASK_PROBE_DIR / "caption_grounding/metrics.json",
    "cross_modal_retrieval": QWEN_CROSS_MODAL_RETRIEVAL_PROBE_DIR / "cross_modal_retrieval/metrics.json",
    "temporal_order": QWEN_ORDER_SYNC_TIME_PROBE_DIR / "temporal_order/metrics.json",
    "misalignment_detection": QWEN_ORDER_SYNC_TIME_PROBE_DIR / "misalignment_detection/metrics.json",
    "long_horizon_next_action": QWEN_FUTURE_TASK_PROBE_DIR / "long_horizon_next_action/metrics.json",
    "next_subtask_forecast": QWEN_FUTURE_TASK_PROBE_DIR / "next_subtask_forecast/metrics.json",
    "object_set_forecast": QWEN_FUTURE_TASK_PROBE_DIR / "object_set_forecast/metrics.json",
    "time_to_transition": QWEN_ORDER_SYNC_TIME_PROBE_DIR / "time_to_transition/metrics.json",
    "camera_view_sync_retrieval": QWEN_CAMERA_VIEW_SYNC_PROBE_DIR / "camera_view_sync_retrieval/metrics.json",
    "hand_trajectory_forecast": QWEN_SENSOR_TARGET_PROBE_DIR / "hand_trajectory_forecast/metrics.json",
    "modality_reconstruction": QWEN_SENSOR_TARGET_PROBE_DIR / "modality_reconstruction/metrics.json",
    "imu_to_hand_pose": QWEN_SENSOR_TARGET_PROBE_DIR / "imu_to_hand_pose/metrics.json",
}
QWEN_FUTURE_TASK_METRIC_KEYS = {
    "caption_grounding": "caption_grounding_mrr",
    "cross_modal_retrieval": "cross_modal_retrieval_mrr",
    "temporal_order": "temporal_order_f1",
    "misalignment_detection": "misalignment_detection_f1",
    "long_horizon_next_action": "long_horizon_next_action_macro_f1",
    "next_subtask_forecast": "next_subtask_forecast_macro_f1",
    "object_set_forecast": "object_set_forecast_micro_f1",
    "time_to_transition": "time_to_transition_mae",
    "camera_view_sync_retrieval": "camera_view_sync_retrieval_mrr",
    "hand_trajectory_forecast": "hand_trajectory_forecast_mrr",
    "modality_reconstruction": "modality_reconstruction_mrr",
    "imu_to_hand_pose": "imu_to_hand_pose_mrr",
}
COSMOS_SUPER_RETRIEVAL_TASK_METRIC_PATHS = {
    "hand_trajectory_forecast": COSMOS_SUPER_RETRIEVAL_TASK_PROBE_DIR / "hand_trajectory_forecast/metrics.json",
    "cross_modal_retrieval": COSMOS_SUPER_RETRIEVAL_TASK_PROBE_DIR / "cross_modal_retrieval/metrics.json",
    "modality_reconstruction": COSMOS_SUPER_RETRIEVAL_TASK_PROBE_DIR / "modality_reconstruction/metrics.json",
    "imu_to_hand_pose": COSMOS_SUPER_RETRIEVAL_TASK_PROBE_DIR / "imu_to_hand_pose/metrics.json",
    "camera_view_sync_retrieval": COSMOS_SUPER_RETRIEVAL_TASK_PROBE_DIR / "camera_view_sync_retrieval/metrics.json",
}
COSMOS_SUPER_RETRIEVAL_TASK_METRIC_KEYS = {
    "hand_trajectory_forecast": "hand_trajectory_forecast_mrr",
    "cross_modal_retrieval": "cross_modal_retrieval_mrr",
    "modality_reconstruction": "modality_reconstruction_mrr",
    "imu_to_hand_pose": "imu_to_hand_pose_mrr",
    "camera_view_sync_retrieval": "camera_view_sync_retrieval_mrr",
}
OUTPUT_JSON = ROOT / "docs/data/unified_task_model_radar.json"
OUTPUT_SINGLE_JSON = ROOT / "docs/data/single_episode_task_model_radar.json"
OUTPUT_128_JSON = ROOT / "docs/data/episode128_task_model_radar.json"
OUTPUT_MATRIX_JSON = ROOT / "docs/data/task_method_20_result_matrix.json"
OUTPUT_MATRIX_MD = ROOT / "TASK_METHOD_20_RESULT_MATRIX.md"
OUTPUT_SVG = ROOT / "docs/assets/charts/unified_task_model_radar.svg"
OUTPUT_SINGLE_SVG = ROOT / "docs/assets/charts/single_episode_task_model_radar.svg"
OUTPUT_128_SVG = ROOT / "docs/assets/charts/episode128_task_model_radar.svg"


SERIES = {
    "minimal": {
        "label": "Minimal",
        "short_label": "Min",
        "color": "#ccffa0",
        "kind": "full_20_task_baseline",
        "scope": "1 public sample episode",
        "stroke_dasharray": None,
    },
    "neural_mlp": {
        "label": "Neural MLP",
        "short_label": "NN",
        "color": "#67e8d1",
        "kind": "full_20_task_baseline",
        "scope": "1 public sample episode",
        "stroke_dasharray": None,
    },
    "metadata128_simple": {
        "label": "128ep Aligned Simple",
        "short_label": "128-S",
        "color": "#ffd166",
        "kind": "partial_128_episode_aligned_baseline",
        "scope": "128 selected episodes, JSONL metadata/text plus staged sensor-block targets where available",
        "stroke_dasharray": "9 6",
    },
    "metadata128_neural_mlp": {
        "label": "128ep Aligned NN",
        "short_label": "128-NN",
        "color": "#f472b6",
        "kind": "partial_128_episode_aligned_baseline",
        "scope": "128 selected episodes, JSONL metadata/text plus staged sensor-block targets where available",
        "stroke_dasharray": "3 6",
    },
    "raw128_simple": {
        "label": "128ep Raw Simple",
        "short_label": "128-RS",
        "color": "#f59e0b",
        "kind": "complete_128_episode_raw_feature_baseline",
        "scope": "128 selected episodes, staged 4430-dim sensor NPZ features; 2 compact proxy axes",
        "stroke_dasharray": "8 4",
    },
    "raw128_neural_mlp": {
        "label": "128ep Raw NN",
        "short_label": "128-RN",
        "color": "#22d3ee",
        "kind": "complete_128_episode_raw_feature_baseline",
        "scope": "128 selected episodes, staged 4430-dim sensor NPZ features; 2 compact proxy axes",
        "stroke_dasharray": "2 5",
    },
    "qwen3_omni_v6_lora": {
        "label": "Qwen3-Omni v6 LoRA",
        "short_label": "Qwen3",
        "color": "#9bb8ff",
        "kind": "partial_128_episode_foundation_model_overlay",
        "scope": "128 selected episodes, held-out test",
        "stroke_dasharray": "7 7",
    },
    "cosmos3_super_reasoner": {
        "label": "Cosmos3-Super Reasoner",
        "short_label": "C3-S",
        "color": "#ff9c7a",
        "kind": "partial_128_episode_foundation_model_overlay",
        "scope": "128 selected episodes, held-out test",
        "stroke_dasharray": "4 7",
    },
    "cosmos3_nano_future_window": {
        "label": "Cosmos3-Nano Future Window",
        "short_label": "C3-N",
        "color": "#d9c7ff",
        "kind": "partial_128_episode_world_model_overlay",
        "scope": "128 selected episodes, held-out test",
        "stroke_dasharray": "2 7",
    },
}

FOUNDATION_TASK_METRICS = {
    "timeline_action": {
        "qwen3_omni_v6_lora": "action_macro_f1",
        "cosmos3_super_reasoner": "action_macro_f1",
        "cosmos3_nano_future_window": "action_accuracy_from_retrieved_future",
    },
    "timeline_subtask": {
        "qwen3_omni_v6_lora": "subtask_accuracy",
        "cosmos3_super_reasoner": "subtask_accuracy",
    },
    "transition_detection": {
        "qwen3_omni_v6_lora": "transition_accuracy",
        "cosmos3_super_reasoner": "transition_accuracy",
        "cosmos3_nano_future_window": "transition_accuracy",
    },
    "next_action": {
        "qwen3_omni_v6_lora": "next_action_accuracy",
        "cosmos3_super_reasoner": "next_action_accuracy",
        "cosmos3_nano_future_window": "action_accuracy_from_retrieved_future",
    },
    "contact_prediction": {
        "qwen3_omni_v6_lora": "contact_accuracy",
        "cosmos3_super_reasoner": "contact_accuracy",
        "cosmos3_nano_future_window": "contact_accuracy",
    },
    "object_relevance": {
        "qwen3_omni_v6_lora": "object_micro_f1",
        "cosmos3_super_reasoner": "object_micro_f1",
    },
    "action_object_relation": {
        "qwen3_omni_v6_lora": "action_object_relation_macro_f1",
        "cosmos3_super_reasoner": "action_object_relation_macro_f1",
        "cosmos3_nano_future_window": "action_object_relation_macro_f1",
    },
    "caption_grounding": {
        "cosmos3_super_reasoner": "caption_grounding_iou",
    },
    "long_horizon_next_action": {
        "cosmos3_super_reasoner": "long_horizon_next_action_macro_f1",
        "cosmos3_nano_future_window": "long_horizon_next_action_macro_f1",
    },
    "next_subtask_forecast": {
        "cosmos3_nano_future_window": "next_subtask_forecast_macro_f1",
    },
    "modality_reconstruction": {
        "cosmos3_nano_future_window": "feature_reconstruction_quality",
    },
    "object_set_forecast": {
        "cosmos3_nano_future_window": "object_set_forecast_micro_f1",
    },
    "cross_modal_retrieval": {
        "cosmos3_nano_future_window": "future_retrieval_mrr",
    },
    "time_to_transition": {
        "cosmos3_super_reasoner": "time_to_transition_mae",
        "cosmos3_nano_future_window": "time_to_transition_mae",
    },
}

FOUNDATION_METRIC_PATHS = {
    "qwen3_omni_v6_lora": QWEN_V6_METRICS_PATH,
    "cosmos3_super_reasoner": COSMOS_SUPER_REASONER_METRICS_PATH,
    "cosmos3_nano_future_window": COSMOS_NANO_METRICS_PATH,
}

FOUNDATION_METRIC_SOURCE_OVERRIDES = {
    ("qwen3_omni_v6_lora", "action_object_relation"): QWEN_ACTION_OBJECT_METRICS_PATH,
    ("cosmos3_super_reasoner", "action_object_relation"): COSMOS_SUPER_ACTION_OBJECT_METRICS_PATH,
    ("cosmos3_super_reasoner", "caption_grounding"): COSMOS_SUPER_CAPTION_GROUNDING_METRICS_PATH,
    ("qwen3_omni_v6_lora", "caption_grounding"): QWEN_FUTURE_TASK_METRIC_PATHS["caption_grounding"],
    ("qwen3_omni_v6_lora", "cross_modal_retrieval"): QWEN_FUTURE_TASK_METRIC_PATHS["cross_modal_retrieval"],
    ("qwen3_omni_v6_lora", "temporal_order"): QWEN_FUTURE_TASK_METRIC_PATHS["temporal_order"],
    ("qwen3_omni_v6_lora", "misalignment_detection"): QWEN_FUTURE_TASK_METRIC_PATHS["misalignment_detection"],
    ("qwen3_omni_v6_lora", "long_horizon_next_action"): QWEN_FUTURE_TASK_METRIC_PATHS["long_horizon_next_action"],
    ("qwen3_omni_v6_lora", "next_subtask_forecast"): QWEN_FUTURE_TASK_METRIC_PATHS["next_subtask_forecast"],
    ("qwen3_omni_v6_lora", "object_set_forecast"): QWEN_FUTURE_TASK_METRIC_PATHS["object_set_forecast"],
    ("qwen3_omni_v6_lora", "time_to_transition"): QWEN_FUTURE_TASK_METRIC_PATHS["time_to_transition"],
    ("qwen3_omni_v6_lora", "camera_view_sync_retrieval"): QWEN_FUTURE_TASK_METRIC_PATHS["camera_view_sync_retrieval"],
    ("qwen3_omni_v6_lora", "hand_trajectory_forecast"): QWEN_FUTURE_TASK_METRIC_PATHS["hand_trajectory_forecast"],
    ("qwen3_omni_v6_lora", "modality_reconstruction"): QWEN_FUTURE_TASK_METRIC_PATHS["modality_reconstruction"],
    ("qwen3_omni_v6_lora", "imu_to_hand_pose"): QWEN_FUTURE_TASK_METRIC_PATHS["imu_to_hand_pose"],
    ("cosmos3_nano_future_window", "long_horizon_next_action"): COSMOS_NANO_LONG_HORIZON_METRICS_PATH,
    ("cosmos3_nano_future_window", "next_subtask_forecast"): COSMOS_NANO_NEXT_SUBTASK_METRICS_PATH,
    ("cosmos3_nano_future_window", "modality_reconstruction"): COSMOS_NANO_MODALITY_RECONSTRUCTION_METRICS_PATH,
    ("cosmos3_nano_future_window", "action_object_relation"): COSMOS_NANO_ACTION_OBJECT_METRICS_PATH,
    ("cosmos3_nano_future_window", "object_set_forecast"): COSMOS_NANO_OBJECT_SET_METRICS_PATH,
    ("cosmos3_nano_future_window", "time_to_transition"): COSMOS_NANO_TIME_TO_TRANSITION_METRICS_PATH,
    ("cosmos3_super_reasoner", "long_horizon_next_action"): COSMOS_SUPER_LONG_HORIZON_METRICS_PATH,
    ("cosmos3_super_reasoner", "time_to_transition"): COSMOS_SUPER_TIME_TO_TRANSITION_METRICS_PATH,
    ("cosmos3_super_reasoner", "hand_trajectory_forecast"): COSMOS_SUPER_RETRIEVAL_TASK_METRIC_PATHS["hand_trajectory_forecast"],
    ("cosmos3_super_reasoner", "cross_modal_retrieval"): COSMOS_SUPER_RETRIEVAL_TASK_METRIC_PATHS["cross_modal_retrieval"],
    ("cosmos3_super_reasoner", "modality_reconstruction"): COSMOS_SUPER_RETRIEVAL_TASK_METRIC_PATHS["modality_reconstruction"],
    ("cosmos3_super_reasoner", "imu_to_hand_pose"): COSMOS_SUPER_RETRIEVAL_TASK_METRIC_PATHS["imu_to_hand_pose"],
    ("cosmos3_super_reasoner", "camera_view_sync_retrieval"): COSMOS_SUPER_RETRIEVAL_TASK_METRIC_PATHS["camera_view_sync_retrieval"],
}

SHORT_TASK_LABELS = {
    "timeline_action": "Action",
    "timeline_subtask": "Step",
    "transition_detection": "Boundary",
    "next_action": "Next act",
    "hand_trajectory_forecast": "Hand traj",
    "contact_prediction": "Contact",
    "object_relevance": "Objects",
    "caption_grounding": "Language",
    "cross_modal_retrieval": "X-modal",
    "modality_reconstruction": "Recon",
    "temporal_order": "Order",
    "misalignment_detection": "Sync",
    "long_horizon_next_action": "Long act",
    "next_subtask_forecast": "Long step",
    "interaction_text_prediction": "Interact txt",
    "action_object_relation": "Act+obj",
    "object_set_forecast": "Future obj",
    "imu_to_hand_pose": "IMU->hand",
    "camera_view_sync_retrieval": "Cam sync",
    "time_to_transition": "Time2bdry",
}

METHOD_DETAILS = {
    "minimal": "Single-episode simple heads over the public sample split.",
    "neural_mlp": "Single-episode compact PyTorch MLP heads on the same 20 task contracts.",
    "metadata128_simple": "128-episode aligned simple baselines: JSONL metadata/text tasks plus staged sensor-block tasks where the processed target exists.",
    "metadata128_neural_mlp": "128-episode aligned MLP baselines: JSONL metadata/text tasks plus staged sensor-block tasks where the processed target exists.",
    "raw128_simple": "128-episode 4430-dim sensor NPZ simple heads; tasks 15/19 use compact proxies.",
    "raw128_neural_mlp": "128-episode 4430-dim sensor NPZ MLP heads; tasks 15/19 use compact proxies.",
    "qwen3_omni_v6_lora": "Verified held-out Qwen3-Omni v6 LoRA metrics, plus task 16 and any completed private-GPU future/retrieval/sensor-target probes scored from task-specific JSON.",
    "cosmos3_super_reasoner": "Verified Cosmos3-Super base-weight Reasoner JSON-task evaluation, plus task 8/16 and a derived task-20 action-boundary timing probe scored from existing verified JSON.",
    "cosmos3_nano_future_window": "Verified Cosmos3-Nano future-window compatibility metrics, plus tasks 10/13/14/16/17 and a derived task-20 boundary timing probe scored from existing held-out future-window artifacts.",
}

PROXY_TASK_IDS = {"interaction_text_prediction", "camera_view_sync_retrieval"}
SINGLE_EPISODE_SERIES = ("minimal", "neural_mlp")
EPISODE128_SERIES = (
    "metadata128_simple",
    "metadata128_neural_mlp",
    "raw128_simple",
    "raw128_neural_mlp",
    "qwen3_omni_v6_lora",
    "cosmos3_super_reasoner",
    "cosmos3_nano_future_window",
)

STATUS_LABELS = {
    "scored": "scored",
    "proxy_scored": "proxy scored",
    "unsupported_without_required_target": "unsupported",
    "not_supported_by_metadata_only_package": "not supported",
    "not_evaluated_in_verified_package": "not evaluated",
    "missing_public_metric": "missing metric",
}

STATUS_SHORT = {
    "scored": "score",
    "proxy_scored": "proxy",
    "unsupported_without_required_target": "unsupported",
    "not_supported_by_metadata_only_package": "not supported",
    "not_evaluated_in_verified_package": "not evaluated",
    "missing_public_metric": "missing",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def foundation_task_metric_mapping(
    qwen_metrics: dict[str, Any],
    cosmos_super_metrics: dict[str, Any],
) -> dict[str, dict[str, str]]:
    mapping = {task_id: dict(series_metrics) for task_id, series_metrics in FOUNDATION_TASK_METRICS.items()}
    for task_id, path in QWEN_FUTURE_TASK_METRIC_PATHS.items():
        payload = read_json(path)
        metric_key = QWEN_FUTURE_TASK_METRIC_KEYS[task_id]
        metric_value = payload.get(metric_key)
        if payload.get("status") != "pass" or not isinstance(metric_value, (int, float)):
            continue
        qwen_metrics[metric_key] = metric_value
        mapping.setdefault(task_id, {})["qwen3_omni_v6_lora"] = metric_key
    for task_id, path in COSMOS_SUPER_RETRIEVAL_TASK_METRIC_PATHS.items():
        payload = read_json(path)
        metric_key = COSMOS_SUPER_RETRIEVAL_TASK_METRIC_KEYS[task_id]
        metric_value = payload.get(metric_key)
        if payload.get("status") != "pass" or not isinstance(metric_value, (int, float)):
            continue
        cosmos_super_metrics[metric_key] = metric_value
        mapping.setdefault(task_id, {})["cosmos3_super_reasoner"] = metric_key
    return mapping


def read_a100_metadata_record(task_id: str, *, neural: bool = False) -> dict[str, Any] | None:
    path = METADATA128_BASELINE_DIR / ("neural_mlp" if neural else "") / task_id / "metrics.json"
    if not path.exists():
        return None
    payload = read_json(path)
    status = payload.get("status", "missing_public_metric")
    score = payload.get("primary_score") if status == "pass" else None
    return {
        "raw": score,
        "metric_key": payload.get("primary_metric"),
        "source": str(path.relative_to(ROOT)),
        "scope": payload.get("scope") or "multi_episode_128_aligned_baseline",
        "status": "scored" if status == "pass" and score is not None else "unsupported_without_required_target",
        "reason": payload.get("reason")
        or payload.get("error")
        or (
            "the 128-episode aligned artifact for this task does not contain a numeric public score"
            if status != "pass"
            else None
        ),
    }


def read_a100_raw_metric(task_id: str, *, neural: bool = False) -> dict[str, Any] | None:
    candidates = (
        [RAW128_BASELINE_DIR / "neural_mlp_raw128" / task_id / "metrics.json"]
        if neural
        else [
            RAW128_BASELINE_DIR / "simple_raw128" / task_id / "metrics.json",
            RAW128_BASELINE_DIR / "simple_raw128_centroid" / task_id / "metrics.json",
            RAW128_BASELINE_DIR / "simple_raw128_ridge" / task_id / "metrics.json",
        ]
    )
    for path in candidates:
        if not path.exists():
            continue
        payload = read_json(path)
        if payload.get("status") != "pass":
            continue
        score = payload.get("primary_score")
        if score is None:
            continue
        return {
            "raw": score,
            "metric_key": payload.get("primary_metric"),
            "source": str(path.relative_to(ROOT)),
            "scope": "multi_episode_128_raw_sensor_feature_baseline",
            "status": "proxy_scored" if task_id in PROXY_TASK_IDS else "scored",
            "reason": "documented compact proxy completion for this raw128 task axis" if task_id in PROXY_TASK_IDS else None,
        }
    return None


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def score_from_raw(value: float | None, direction: str, best_lower: float | None = None) -> float | None:
    if value is None:
        return None
    if direction == "lower":
        if value <= 0:
            return 1.0
        if best_lower is None or best_lower <= 0:
            return None
        return clamp01(best_lower / value)
    return clamp01(value)


def format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    if abs(value) >= 10:
        return f"{value:.2f}"
    if abs(value) >= 1:
        return f"{value:.3f}"
    return f"{value:.4f}"


def status_label(status: str | None) -> str:
    return STATUS_LABELS.get(status or "", status or "unknown")


def make_missing_record(series_id: str, task_id: str, metric_key: str | None) -> dict[str, Any]:
    if series_id.startswith("metadata128"):
        status = "not_supported_by_metadata_only_package"
        reason = (
            "the 128-episode aligned rerun did not produce this task target; "
            "raw interaction text, paired camera-view embeddings, or a task-specific target builder is required"
        )
        scope = "multi_episode_128_aligned_baseline"
    elif series_id in {"qwen3_omni_v6_lora", "cosmos3_super_reasoner", "cosmos3_nano_future_window"}:
        status = "not_evaluated_in_verified_package"
        reason = (
            "the verified public model package did not ask this branch to emit that task target; "
            "a new task-specific evaluation package is required for a numeric score"
        )
        scope = "multi_episode_128_partial_model_overlay"
    else:
        status = "missing_public_metric"
        reason = "no public metric artifact was found for this method-task pair"
        scope = SERIES.get(series_id, {}).get("scope")
    return {
        "raw": None,
        "metric_key": metric_key,
        "source": None,
        "scope": scope,
        "status": status,
        "reason": reason,
        "normalized_score": None,
        "raw_text": "n/a",
    }


def finalize_value_record(item: dict[str, Any], direction: str, best_lower: float | None) -> None:
    raw = item.get("raw")
    item.setdefault("status", "scored" if isinstance(raw, (int, float)) else "missing_public_metric")
    item["normalized_score"] = score_from_raw(raw if isinstance(raw, (int, float)) else None, direction, best_lower)
    if item["normalized_score"] is None and item.get("status") in {"scored", "proxy_scored"}:
        item["status"] = "missing_public_metric"
        item.setdefault("reason", "numeric raw score could not be normalized for this task")
    item["raw_text"] = format_metric(raw if isinstance(raw, (int, float)) else None)
    item["status_label"] = status_label(item.get("status"))


def matrix_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in payload["tasks"]:
        for series_id, series_spec in SERIES.items():
            value = task["values"][series_id]
            rows.append(
                {
                    "task_number": task["task_number"],
                    "task_id": task["task_id"],
                    "task_label": task["label"],
                    "series_id": series_id,
                    "method": series_spec["label"],
                    "status": value.get("status"),
                    "status_label": value.get("status_label", status_label(value.get("status"))),
                    "scored": value.get("normalized_score") is not None,
                    "proxy_scored": value.get("status") == "proxy_scored",
                    "raw": value.get("raw"),
                    "raw_text": value.get("raw_text", "n/a"),
                    "normalized_score": value.get("normalized_score"),
                    "metric_key": value.get("metric_key"),
                    "source": value.get("source"),
                    "scope": value.get("scope"),
                    "reason": value.get("reason"),
                }
            )
    return rows


def render_matrix_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Task Method 20-Result Matrix",
        "",
        "Every method has one record for each of the 20 unified task contracts. Numeric scores appear only where a committed runner or verified package produced that task target.",
        "",
        "Legend: `score` = numeric task score, `proxy` = documented raw128 compact proxy score, `unsupported` = artifact exists but required target is not present, `not supported` = metadata-only package cannot form that target, `not evaluated` = verified model package did not request that target.",
        "",
        "| Method | Records | Scored | Proxy scored | Scoreless | Status counts |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for record in payload["series"]:
        counts = record["status_counts"]
        count_text = ", ".join(f"{status_label(key)} {value}" for key, value in sorted(counts.items()))
        lines.append(
            f"| {record['label']} | {record['result_record_count']} | {record['scored_task_count']} | "
            f"{record['proxy_scored_task_count']} | {record['scoreless_task_count']} | {count_text} |"
        )
    lines.extend(
        [
            "",
            "| # | Task | " + " | ".join(spec["short_label"] for spec in SERIES.values()) + " |",
            "| ---: | --- | " + " | ".join("---" for _ in SERIES) + " |",
        ]
    )
    for task in payload["tasks"]:
        cells = [STATUS_SHORT.get(task["values"][series_id].get("status"), "unknown") for series_id in SERIES]
        lines.append(f"| {task['task_number']:02d} | {task['label']} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "Sources and raw values are in `docs/data/task_method_20_result_matrix.json` and `docs/data/unified_task_model_radar.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def filtered_radar_payload(
    payload: dict[str, Any],
    series_ids: tuple[str, ...],
    *,
    title: str,
    description: str,
) -> dict[str, Any]:
    selected = set(series_ids)
    series = [json.loads(json.dumps(record)) for record in payload["series"] if record["id"] in selected]
    tasks = []
    for task in payload["tasks"]:
        task_copy = {key: json.loads(json.dumps(value)) for key, value in task.items() if key != "values"}
        task_copy["values"] = {
            series_id: json.loads(json.dumps(task["values"][series_id]))
            for series_id in series_ids
            if series_id in task["values"]
        }
        tasks.append(task_copy)
    rows = [
        json.loads(json.dumps(row))
        for row in payload["task_method_result_matrix"]
        if row.get("series_id") in selected
    ]
    return {
        "title": title,
        "status": payload["status"],
        "generated_at_utc": payload["generated_at_utc"],
        "description": description,
        "task_count": payload["task_count"],
        "method_count": len(series),
        "method_task_record_count": sum(record.get("result_record_count", 0) for record in series),
        "scored_method_task_count": sum(record.get("scored_task_count", 0) for record in series),
        "normalization_policy": payload["normalization_policy"],
        "source_unified_radar": "docs/data/unified_task_model_radar.json",
        "source_result_matrix": "docs/data/task_method_20_result_matrix.json",
        "series": series,
        "tasks": tasks,
        "task_method_result_matrix": rows,
    }


def point(cx: float, cy: float, radius: float, angle: float) -> tuple[float, float]:
    return cx + math.cos(angle) * radius, cy + math.sin(angle) * radius


def svg_text(
    x: float,
    y: float,
    text: str,
    *,
    size: int = 16,
    fill: str = "#f4f8ef",
    anchor: str = "start",
    weight: int | str = 600,
    opacity: float = 1.0,
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="Space Grotesk, Arial, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" opacity="{opacity:.3f}">{html.escape(text)}</text>'
    )


def split_text(text: str, max_chars: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        if len(current) + 1 + len(word) <= max_chars:
            current += " " + word
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def svg_text_lines(
    x: float,
    y: float,
    lines: list[str],
    *,
    size: int = 14,
    fill: str = "#f4f8ef",
    anchor: str = "start",
    weight: int | str = 600,
    line_height: float = 18,
    opacity: float = 1.0,
) -> list[str]:
    return [
        svg_text(x, y + idx * line_height, line, size=size, fill=fill, anchor=anchor, weight=weight, opacity=opacity)
        for idx, line in enumerate(lines)
    ]


def polyline(points: list[tuple[float, float]], *, fill: str, stroke: str, opacity: float, stroke_width: float, dash: str | None = None) -> str:
    coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<polygon points="{coords}" fill="{fill}" fill-opacity="{opacity:.3f}" '
        f'stroke="{stroke}" stroke-opacity="0.92" stroke-width="{stroke_width}"{dash_attr}/>'
    )


def build_payload() -> dict[str, Any]:
    suite = read_json(TASK_SUITE_PATH)
    qwen = read_json(QWEN_V6_METRICS_PATH)
    cosmos_super = read_json(COSMOS_SUPER_REASONER_METRICS_PATH)
    cosmos_nano = read_json(COSMOS_NANO_METRICS_PATH)
    cosmos_fd = read_json(COSMOS_SUPER_FD_METRICS_PATH)
    qwen.update(read_json(QWEN_ACTION_OBJECT_METRICS_PATH))
    cosmos_super.update(read_json(COSMOS_SUPER_ACTION_OBJECT_METRICS_PATH))
    cosmos_super.update(read_json(COSMOS_SUPER_CAPTION_GROUNDING_METRICS_PATH))
    cosmos_super.update(read_json(COSMOS_SUPER_LONG_HORIZON_METRICS_PATH))
    cosmos_super.update(read_json(COSMOS_SUPER_TIME_TO_TRANSITION_METRICS_PATH))
    cosmos_nano.update(read_json(COSMOS_NANO_LONG_HORIZON_METRICS_PATH))
    cosmos_nano.update(read_json(COSMOS_NANO_NEXT_SUBTASK_METRICS_PATH))
    cosmos_nano.update(read_json(COSMOS_NANO_MODALITY_RECONSTRUCTION_METRICS_PATH))
    cosmos_nano.update(read_json(COSMOS_NANO_ACTION_OBJECT_METRICS_PATH))
    cosmos_nano.update(read_json(COSMOS_NANO_OBJECT_SET_METRICS_PATH))
    cosmos_nano.update(read_json(COSMOS_NANO_TIME_TO_TRANSITION_METRICS_PATH))
    foundation_task_metrics = foundation_task_metric_mapping(qwen, cosmos_super)
    foundation_metrics = {
        "qwen3_omni_v6_lora": qwen,
        "cosmos3_super_reasoner": cosmos_super,
        "cosmos3_nano_future_window": cosmos_nano,
    }

    tasks: list[dict[str, Any]] = []
    for row in suite.get("tasks", []):
        values: dict[str, dict[str, Any]] = {
            "minimal": {
                "raw": row.get("minimal_primary_metric"),
                "metric_key": row.get("metric_key"),
                "source": row.get("artifact_sources", {}).get("minimal_metrics"),
                "scope": "single_episode_public_sample",
                "status": "scored",
            },
            "neural_mlp": {
                "raw": row.get("neural_primary_metric"),
                "metric_key": row.get("metric_key"),
                "source": row.get("artifact_sources", {}).get("neural_metrics"),
                "scope": "single_episode_public_sample",
                "status": "scored",
            },
        }
        for series_id, metric_key in foundation_task_metrics.get(row["task_id"], {}).items():
            raw = foundation_metrics.get(series_id, {}).get(metric_key)
            values[series_id] = {
                "raw": raw,
                "metric_key": metric_key,
                "source": str(
                    FOUNDATION_METRIC_SOURCE_OVERRIDES.get(
                        (series_id, row["task_id"]),
                        FOUNDATION_METRIC_PATHS[series_id],
                    ).relative_to(ROOT)
                ),
                "scope": "multi_episode_128_partial_model_overlay",
                "status": "scored" if isinstance(raw, (int, float)) else "missing_public_metric",
                "reason": None if isinstance(raw, (int, float)) else f"metric {metric_key} is absent from the verified public package",
            }
        metadata_simple = read_a100_metadata_record(row["task_id"], neural=False)
        if metadata_simple:
            values["metadata128_simple"] = metadata_simple
        metadata_neural = read_a100_metadata_record(row["task_id"], neural=True)
        if metadata_neural:
            values["metadata128_neural_mlp"] = metadata_neural
        raw_simple = read_a100_raw_metric(row["task_id"], neural=False)
        if raw_simple:
            values["raw128_simple"] = raw_simple
        raw_neural = read_a100_raw_metric(row["task_id"], neural=True)
        if raw_neural:
            values["raw128_neural_mlp"] = raw_neural

        lower_values = [
            item["raw"]
            for item in values.values()
            if row.get("metric_direction") == "lower" and isinstance(item.get("raw"), (int, float)) and item["raw"] > 0
        ]
        best_lower = min(lower_values) if lower_values else None
        for series_id in SERIES:
            values.setdefault(series_id, make_missing_record(series_id, row["task_id"], row.get("metric_key")))
        for item in values.values():
            finalize_value_record(item, row.get("metric_direction", "higher"), best_lower)

        tasks.append(
            {
                "task_number": row["task_number"],
                "task_id": row["task_id"],
                "label": row.get("task_display_name", row["task_id"]),
                "axis_label": f"{row['task_number']:02d} {row.get('task_display_name', row['task_id'])}",
                "short_label": SHORT_TASK_LABELS.get(row["task_id"], row["task_id"].replace("_", " ").title()),
                "origin": row.get("origin"),
                "metric_key": row.get("metric_key"),
                "metric_name": row.get("metric_name"),
                "metric_direction": row.get("metric_direction"),
                "raw128_proxy_axis": row["task_id"] in PROXY_TASK_IDS,
                "values": values,
            }
        )

    series_records = []
    for series_id, spec in SERIES.items():
        status_counts: dict[str, int] = {}
        for task in tasks:
            status = task["values"][series_id].get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
        covered = sum(1 for task in tasks if task["values"].get(series_id, {}).get("normalized_score") is not None)
        proxy_count = status_counts.get("proxy_scored", 0)
        scoreless = len(tasks) - covered
        series_records.append(
            {
                "id": series_id,
                **spec,
                "method_detail": METHOD_DETAILS.get(series_id, spec["scope"]),
                "plotted_as": "filled polygon" if spec["kind"].startswith("full_20_task_baseline") else "colored point overlay",
                "result_record_count": len(tasks),
                "scored_task_count": covered,
                "covered_task_count": covered,
                "proxy_scored_task_count": proxy_count,
                "scoreless_task_count": scoreless,
                "unsupported_task_count": status_counts.get("unsupported_without_required_target", 0)
                + status_counts.get("not_supported_by_metadata_only_package", 0),
                "not_evaluated_task_count": status_counts.get("not_evaluated_in_verified_package", 0),
                "status_counts": dict(sorted(status_counts.items())),
                "coverage_fraction": covered / max(len(tasks), 1),
                "result_record_fraction": len(tasks) / max(len(tasks), 1),
            }
        )

    fd_loss = (cosmos_fd.get("loss_summary") or {}).get("mean")
    payload = {
        "title": "Unified 20-Task Model Radar",
        "status": "pass",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task_count": len(tasks),
        "method_count": len(SERIES),
        "method_task_record_count": len(tasks) * len(SERIES),
        "scored_method_task_count": sum(
            1
            for task in tasks
            for series_id in SERIES
            if task["values"][series_id].get("normalized_score") is not None
        ),
        "normalization_policy": {
            "higher_is_better": "bounded metrics are plotted directly on 0-1 axes after clipping to [0, 1]",
            "lower_is_better": "lower-error metrics are converted to best_observed_value / raw_value within the same task",
            "raw_values": "raw metric values, metric keys, and sources are retained in this JSON; the SVG is an overview, not a replacement for the metric table",
            "result_record_policy": "every method has 20 task records; records without a numeric score carry explicit unsupported/not-evaluated status and reason fields",
            "foundation_model_overlay": "Qwen3/Cosmos points are plotted only on task-aligned axes. Scoreless records mean the public result does not evaluate that task contract.",
            "metadata_128_overlay": "128-episode aligned baselines have 20 records. Numeric scores come from JSONL metadata/text tasks plus staged sensor-block targets when the processed target exists; raw interaction text and paired camera-view embeddings remain explicit gaps.",
            "raw_128_overlay": "128-episode raw-feature baselines use staged sensor NPZ features. Eighteen axes use direct task targets; interaction text and camera-view sync are completed with documented compact proxies because raw interaction strings and paired video-view embeddings are absent from the 128 export.",
        },
        "series": series_records,
        "tasks": tasks,
        "model_branch_cards": [
            {
                "id": "metadata128_simple",
                "title": "128ep Aligned Simple",
                "status": "a100_rerun_pass",
                "coverage": f"20 records / {next(item for item in series_records if item['id'] == 'metadata128_simple')['scored_task_count']} scored aligned axes",
                "headline": "34,269 rows; train/val/test 25,629/4,608/4,032",
                "source": str((METADATA128_BASELINE_DIR / "summary_report.json").relative_to(ROOT)),
            },
            {
                "id": "metadata128_neural_mlp",
                "title": "128ep Aligned NN",
                "status": "a100_rerun_pass",
                "coverage": f"20 records / {next(item for item in series_records if item['id'] == 'metadata128_neural_mlp')['scored_task_count']} scored aligned axes",
                "headline": "compact MLP heads over metadata/text and staged block features",
                "source": str((METADATA128_BASELINE_DIR / "summary_report.json").relative_to(ROOT)),
            },
            {
                "id": "raw128_simple",
                "title": "128ep Raw Simple",
                "status": "a100_raw20_complete_with_documented_proxies",
                "coverage": f"20 records / {next(item for item in series_records if item['id'] == 'raw128_simple')['scored_task_count']} scored axes; 18 direct + 2 proxy",
                "headline": "34,269 windows; centroid/ridge heads over 4430-dim sensor blocks",
                "source": str((RAW128_BASELINE_DIR / "run_summary_all.json").relative_to(ROOT)),
            },
            {
                "id": "raw128_neural_mlp",
                "title": "128ep Raw NN",
                "status": "a100_raw20_complete_with_documented_proxies",
                "coverage": f"20 records / {next(item for item in series_records if item['id'] == 'raw128_neural_mlp')['scored_task_count']} scored axes; 18 direct + 2 proxy",
                "headline": "MLP heads over staged features; tasks 15/19 use compact proxies",
                "source": str((RAW128_BASELINE_DIR / "run_summary_all.json").relative_to(ROOT)),
            },
            {
                "id": "qwen3_omni_v6_lora",
                "title": "Qwen3-Omni v6 LoRA",
                "status": "verified",
                "task_aligned_axes": SERIES["qwen3_omni_v6_lora"]["short_label"],
                "coverage": f"20 records / {next(item for item in series_records if item['id'] == 'qwen3_omni_v6_lora')['scored_task_count']} scored task-aligned axes",
                "headline": f"JSON validity {format_metric(qwen.get('json_validity_rate'))}; action macro-F1 {format_metric(qwen.get('action_macro_f1'))}",
                "source": str(QWEN_V6_METRICS_PATH.relative_to(ROOT)),
            },
            {
                "id": "cosmos3_super_reasoner",
                "title": "Cosmos3-Super Reasoner",
                "status": "verified_base_weight_eval",
                "coverage": f"20 records / {next(item for item in series_records if item['id'] == 'cosmos3_super_reasoner')['scored_task_count']} scored task-aligned axes",
                "headline": f"JSON validity {format_metric(cosmos_super.get('json_validity_rate'))}; action macro-F1 {format_metric(cosmos_super.get('action_macro_f1'))}",
                "source": str(COSMOS_SUPER_REASONER_METRICS_PATH.relative_to(ROOT)),
            },
            {
                "id": "cosmos3_nano_future_window",
                "title": "Cosmos3-Nano Future Window",
                "status": "verified_compatibility_eval",
                "coverage": f"20 records / {next(item for item in series_records if item['id'] == 'cosmos3_nano_future_window')['scored_task_count']} scored task-aligned axes",
                "headline": f"future retrieval MRR {format_metric(cosmos_nano.get('future_retrieval_mrr'))}; transition accuracy {format_metric(cosmos_nano.get('transition_accuracy'))}",
                "source": str(COSMOS_NANO_METRICS_PATH.relative_to(ROOT)),
            },
            {
                "id": "cosmos3_super_forward_dynamics_lora",
                "title": "Cosmos3-Super Forward-Dynamics LoRA",
                "status": "verified_finetuned_adapter",
                "coverage": "separate camera-pose proxy target, not plotted on the 20 task axes",
                "headline": f"test MSE {format_metric(fd_loss)} over 448 held-out rows",
                "source": str(COSMOS_SUPER_FD_METRICS_PATH.relative_to(ROOT)),
            },
        ],
    }
    payload["task_method_result_matrix"] = matrix_rows(payload)
    return payload


def render_svg(
    payload: dict[str, Any],
    *,
    series_ids: tuple[str, ...] | None = None,
    polygon_series_ids: tuple[str, ...] = ("minimal", "neural_mlp"),
    title: str | None = None,
    subtitle: str | None = None,
    context_line: str | None = None,
    chip_specs: list[tuple[str, str]] | None = None,
    reading_rules: tuple[str, str, str] | None = None,
) -> str:
    width, height = 2400, 1840
    cx, cy, radius = 650, 860, 355
    tasks = payload["tasks"]
    n = len(tasks)
    angles = [-math.pi / 2 + 2 * math.pi * i / n for i in range(n)]
    if series_ids is None:
        series_ids = tuple(record["id"] for record in payload["series"])
    polygon_series_set = set(polygon_series_ids)
    series_records = [record for record in payload["series"] if record["id"] in set(series_ids)]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        '<filter id="softGlow"><feGaussianBlur stdDeviation="5" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>',
        '<pattern id="dots" width="22" height="22" patternUnits="userSpaceOnUse"><circle cx="2" cy="2" r="1.15" fill="#ccffa0" opacity="0.16"/></pattern>',
        "</defs>",
        '<rect width="100%" height="100%" fill="#020502"/>',
        '<rect width="100%" height="100%" fill="url(#dots)" opacity="0.45"/>',
        '<rect x="28" y="28" width="2344" height="1784" rx="18" fill="#061006" fill-opacity="0.88" stroke="#ccffa0" stroke-opacity="0.22"/>',
        svg_text(70, 86, title or payload.get("title", "20-Task Model Radar"), size=36, weight=800),
        svg_text(
            70,
            122,
            subtitle or "Task names, methods, coverage, and metric normalization in one comparison view.",
            size=18,
            fill="#dce8d7",
            weight=650,
        ),
        svg_text(
            70,
            150,
            context_line
            or "Filled areas show complete scored baselines; colored points show partial branches on task-aligned axes.",
            size=15,
            fill="#a5afa2",
            weight=560,
        ),
    ]

    if chip_specs is None:
        chip_specs = [
            ("20 task axes", "#ccffa0"),
            (f"{payload['method_task_record_count']} method-task records", "#67e8d1"),
            (f"{payload['scored_method_task_count']} scored axes", "#22d3ee"),
            ("40/40 raw128 pass", "#f59e0b"),
            ("2 compact proxy axes", "#f472b6"),
        ]
    chip_x = 70
    for label, color in chip_specs:
        chip_w = 168 if len(label) < 15 else 250
        parts.append(f'<rect x="{chip_x}" y="174" width="{chip_w}" height="34" rx="17" fill="{color}" fill-opacity="0.10" stroke="{color}" stroke-opacity="0.38"/>')
        parts.append(svg_text(chip_x + 16, 197, label, size=13, fill=color, weight=760))
        chip_x += chip_w + 12

    parts.append('<rect x="54" y="235" width="1190" height="1190" rx="14" fill="#020502" fill-opacity="0.42" stroke="#ccffa0" stroke-opacity="0.14"/>')
    parts.append(svg_text(84, 276, "Normalized task scores", size=23, weight=800))
    parts.append(svg_text(84, 302, "Each axis is one task. Longer radius means better after metric-direction normalization.", size=13, fill="#a5afa2", weight=560))

    for level in range(1, 6):
        r = radius * level / 5
        ring = [point(cx, cy, r, angle) for angle in angles]
        parts.append(polyline(ring, fill="none", stroke="#ccffa0", opacity=0, stroke_width=1.1))
        parts[-1] = parts[-1].replace('fill="none" fill-opacity="0.000"', 'fill="none"').replace('stroke-opacity="0.92"', 'stroke-opacity="0.15"')
        parts.append(svg_text(cx + 8, cy - r + 4, f"{level / 5:.1f}", size=11, fill="#a5afa2", weight=600, opacity=0.75))

    for task, angle in zip(tasks, angles):
        x, y = point(cx, cy, radius, angle)
        parts.append(f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{x:.1f}" y2="{y:.1f}" stroke="#ccffa0" stroke-opacity="0.12" stroke-width="1"/>')
        lx, ly = point(cx, cy, radius + 82, angle)
        parts.append(f'<circle cx="{lx:.1f}" cy="{ly:.1f}" r="15.5" fill="#ccffa0" fill-opacity="0.12" stroke="#ccffa0" stroke-opacity="0.34"/>')
        parts.append(svg_text(lx, ly + 4, f"{task['task_number']:02d}", size=11, fill="#ccffa0", anchor="middle", weight=800, opacity=0.98))

    for series_id in series_ids:
        if series_id not in polygon_series_set:
            continue
        spec = SERIES[series_id]
        points = []
        for task, angle in zip(tasks, angles):
            score = task["values"].get(series_id, {}).get("normalized_score")
            points.append(point(cx, cy, radius * float(score or 0.0), angle))
        parts.append(polyline(points, fill=spec["color"], stroke=spec["color"], opacity=0.18 if series_id in {"minimal", "raw128_simple"} else 0.16, stroke_width=4.2, dash=spec.get("stroke_dasharray")))
        for x, y in points:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.0" fill="{spec["color"]}" stroke="#020502" stroke-width="1.1"/>')

    for series_id in series_ids:
        if series_id in polygon_series_set:
            continue
        spec = SERIES[series_id]
        for task, angle in zip(tasks, angles):
            score = task["values"].get(series_id, {}).get("normalized_score")
            if score is None:
                continue
            x, y = point(cx, cy, radius * float(score), angle)
            radius_px = 6.5 if series_id.startswith(("metadata128", "raw128")) else 8.0
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius_px:.1f}" fill="{spec["color"]}" fill-opacity="0.92" '
                f'stroke="#020502" stroke-width="2.0"/>'
            )

    legend_x, legend_y = 1315, 178
    parts.append(f'<rect x="{legend_x - 30}" y="{legend_y - 38}" width="1000" height="560" rx="14" fill="#020502" fill-opacity="0.58" stroke="#ccffa0" stroke-opacity="0.20"/>')
    parts.append(svg_text(legend_x, legend_y, "Methods compared", size=25, weight=800))
    parts.append(svg_text(legend_x, legend_y + 30, "Each method has 20 records; scored axes and scoreless statuses stay in the JSON matrix.", size=13, fill="#a5afa2", weight=560))

    cursor = legend_y + 74
    for record in series_records:
        color = record["color"]
        parts.append(f'<line x1="{legend_x}" y1="{cursor - 7}" x2="{legend_x + 50}" y2="{cursor - 7}" stroke="{color}" stroke-width="7" stroke-linecap="round"/>')
        if record["id"] not in polygon_series_set:
            parts.append(f'<circle cx="{legend_x + 25}" cy="{cursor - 7}" r="7" fill="{color}" stroke="#020502" stroke-width="2"/>')
        parts.append(svg_text(legend_x + 66, cursor - 12, record["label"], size=15, weight=800))
        parts.append(svg_text(legend_x + 392, cursor - 12, f"20 records / {record['scored_task_count']} scored", size=13, fill=color, weight=800))
        detail_lines = split_text(METHOD_DETAILS.get(record["id"], record["scope"]), 78)[:2]
        parts.extend(svg_text_lines(legend_x + 66, cursor + 8, detail_lines, size=11, fill="#a5afa2", weight=560, line_height=15))
        cursor += 50

    key_x, key_y = 1315, 780
    parts.append(f'<rect x="{key_x - 30}" y="{key_y - 44}" width="1000" height="680" rx="14" fill="#020502" fill-opacity="0.58" stroke="#ccffa0" stroke-opacity="0.20"/>')
    parts.append(svg_text(key_x, key_y, "Task axis key", size=25, weight=800))
    parts.append(svg_text(key_x, key_y + 30, "Full task names are listed here so the polygon remains readable at homepage scale.", size=13, fill="#a5afa2", weight=560))
    for idx, task in enumerate(tasks):
        col = 0 if idx < 10 else 1
        row = idx if idx < 10 else idx - 10
        x0 = key_x + col * 500
        y0 = key_y + 74 + row * 54
        proxy = task["task_id"] in PROXY_TASK_IDS
        badge_fill = "#f472b6" if proxy else "#ccffa0"
        parts.append(f'<rect x="{x0}" y="{y0 - 16}" width="36" height="26" rx="6" fill="{badge_fill}" fill-opacity="0.14" stroke="{badge_fill}" stroke-opacity="0.40"/>')
        parts.append(svg_text(x0 + 18, y0 + 2, f"{task['task_number']:02d}", size=11, fill=badge_fill, anchor="middle", weight=800))
        name_lines = split_text(str(task["label"]), 42)[:2]
        parts.extend(svg_text_lines(x0 + 48, y0 - 3, name_lines, size=12, fill="#f4f8ef", weight=760, line_height=14))
        metric_label = f"{task.get('metric_name') or task.get('metric_key')} / {'lower better' if task.get('metric_direction') == 'lower' else 'higher better'}"
        if proxy:
            metric_label += " / raw128 proxy"
        parts.append(svg_text(x0 + 48, y0 + 29, metric_label, size=10, fill="#a5afa2", weight=560))

    table_y = 1680
    if reading_rules is None:
        reading_rules = (
            "Every method has 20 task records; radius appears only where a numeric task score exists.",
            "Raw128 completion: 18 direct task targets plus 2 compact proxies. Task 15 predicts the dominant caption/object/interaction hash bin; task 19 retrieves depth/audio sync from camera pose.",
            "Scoreless metadata/Qwen/Cosmos records are explicit unsupported or not-evaluated cells in docs/data/task_method_20_result_matrix.json.",
        )
    parts.append(f'<rect x="70" y="{table_y - 38}" width="2260" height="120" rx="12" fill="#020502" fill-opacity="0.58" stroke="#ccffa0" stroke-opacity="0.16"/>')
    parts.append(svg_text(100, table_y - 10, "Reading rules", size=16, fill="#ccffa0", weight=800))
    parts.append(svg_text(220, table_y - 10, reading_rules[0], size=14, fill="#dce8d7", weight=650))
    parts.append(svg_text(220, table_y + 18, reading_rules[1], size=13, fill="#a5afa2", weight=560))
    parts.append(svg_text(220, table_y + 44, reading_rules[2], size=13, fill="#a5afa2", weight=560))

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> int:
    payload = build_payload()
    single_payload = filtered_radar_payload(
        payload,
        SINGLE_EPISODE_SERIES,
        title="Single-Episode 20-Task Radar",
        description="Minimal and Neural MLP baselines on the one public sample episode, both scored on all 20 task contracts.",
    )
    episode128_payload = filtered_radar_payload(
        payload,
        EPISODE128_SERIES,
        title="128-Episode 20-Task Radar",
        description="Selected 128-episode metadata/raw baselines plus verified Qwen3/Cosmos branches. Every method has 20 records; numeric scores appear only where the public artifact produced that task target.",
    )
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_SINGLE_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_128_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MATRIX_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_SVG.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_SINGLE_SVG.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_128_SVG.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    OUTPUT_SINGLE_JSON.write_text(json.dumps(single_payload, indent=2) + "\n", encoding="utf-8")
    OUTPUT_128_JSON.write_text(json.dumps(episode128_payload, indent=2) + "\n", encoding="utf-8")
    matrix_payload = {
        "title": "Task Method 20-Result Matrix",
        "status": "pass",
        "generated_at_utc": payload["generated_at_utc"],
        "task_count": payload["task_count"],
        "method_count": payload["method_count"],
        "method_task_record_count": payload["method_task_record_count"],
        "scored_method_task_count": payload["scored_method_task_count"],
        "series": payload["series"],
        "records": payload["task_method_result_matrix"],
    }
    OUTPUT_MATRIX_JSON.write_text(json.dumps(matrix_payload, indent=2) + "\n", encoding="utf-8")
    OUTPUT_MATRIX_MD.write_text(render_matrix_markdown(payload), encoding="utf-8")
    OUTPUT_SVG.write_text(render_svg(payload), encoding="utf-8")
    OUTPUT_SINGLE_SVG.write_text(
        render_svg(
            single_payload,
            series_ids=SINGLE_EPISODE_SERIES,
            polygon_series_ids=SINGLE_EPISODE_SERIES,
            title="Single-Episode 20-Task Radar",
            subtitle="One public sample episode; both baseline heads score every task axis.",
            context_line="This view isolates the 1-episode task-head setup from the multi-episode model branches.",
            chip_specs=[
                ("20 task axes", "#ccffa0"),
                ("40 method-task records", "#67e8d1"),
                ("40 scored axes", "#22d3ee"),
                ("2 filled baseline polygons", "#f472b6"),
            ],
            reading_rules=(
                "Both single-episode methods have numeric scores on every one of the 20 task contracts.",
                "This radar is the cleanest view of public-sample Minimal vs Neural MLP behavior before any 128-episode scale-up.",
                "Raw metric values and sources remain in docs/data/single_episode_task_model_radar.json and docs/data/task_method_20_result_matrix.json.",
            ),
        ),
        encoding="utf-8",
    )
    OUTPUT_128_SVG.write_text(
        render_svg(
            episode128_payload,
            series_ids=EPISODE128_SERIES,
            polygon_series_ids=("raw128_simple", "raw128_neural_mlp"),
            title="128-Episode 20-Task Radar",
            subtitle="Selected 96/16/16 episode split; raw-feature heads score all 20 axes.",
            context_line="Raw128 baselines are filled polygons; metadata, Qwen3, and Cosmos branches plot only evaluated task targets.",
            chip_specs=[
                ("20 task axes", "#ccffa0"),
                ("140 method-task records", "#67e8d1"),
                (f"{episode128_payload['scored_method_task_count']} scored axes", "#22d3ee"),
                ("40/40 raw128 pass", "#f59e0b"),
                (
                    f"{episode128_payload['method_task_record_count'] - episode128_payload['scored_method_task_count']} explicit scoreless",
                    "#f472b6",
                ),
            ],
            reading_rules=(
                "Every 128-episode method has 20 result records; radius appears only where a numeric score exists.",
                "Raw128 Simple and Raw128 NN are complete 20/20 scored multi-episode baselines; tasks 15/19 are documented compact proxies.",
                "Qwen3/Cosmos task 16 uses existing verified action/object JSON; other scoreless cells remain explicit not-supported or not-evaluated records.",
            ),
        ),
        encoding="utf-8",
    )
    print(f"PASS: wrote {OUTPUT_JSON}")
    print(f"PASS: wrote {OUTPUT_SINGLE_JSON}")
    print(f"PASS: wrote {OUTPUT_128_JSON}")
    print(f"PASS: wrote {OUTPUT_MATRIX_JSON}")
    print(f"PASS: wrote {OUTPUT_MATRIX_MD}")
    print(f"PASS: wrote {OUTPUT_SVG}")
    print(f"PASS: wrote {OUTPUT_SINGLE_SVG}")
    print(f"PASS: wrote {OUTPUT_128_SVG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

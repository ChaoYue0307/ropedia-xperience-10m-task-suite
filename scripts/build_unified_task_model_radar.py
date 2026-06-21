#!/usr/bin/env python3
"""Build unified 20-task radar charts for baseline and model diagnostics."""

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
QWEN_INTERACTION_TEXT_PROBE_DIR = (
    ROOT
    / "results/omni_finetune"
    / "xperience10m_qwen3_omni_v6_interaction_text_task15_a100_20260620T010305Z"
)
COSMOS_SUPER_RETRIEVAL_TASK_PROBE_DIR = (
    ROOT
    / "results/omni_finetune"
    / "xperience10m_cosmos3_super_retrieval_task_probes_a100_textonly_prompatch_v2_20260620"
)
COSMOS_SUPER_FUTURE_TASK_PROBE_DIR = (
    ROOT
    / "results/omni_finetune"
    / "xperience10m_cosmos3_super_future_task_probes_a100_textonly_v1_20260620"
)
COSMOS_SUPER_INTERACTION_TEXT_PROBE_DIR = (
    ROOT
    / "results/omni_finetune"
    / "xperience10m_cosmos3_super_interaction_text_task15_textonly_v1_20260620T1558Z"
)
COSMOS_NANO_RETRIEVAL_TASK_PROBE_DIR = (
    ROOT
    / "results/omni_finetune"
    / "xperience10m_cosmos3_nano_retrieval_task_probes_a100_patched_textonly_20260621"
)
COSMOS_NANO_INTERACTION_TEXT_PROBE_DIR = (
    ROOT
    / "results/omni_finetune"
    / "xperience10m_cosmos3_nano_interaction_text_task15_patched_textonly_20260621"
)
COSMOS_NANO_FUTURE_ORDER_PROBE_DIR = (
    ROOT
    / "results/omni_finetune"
    / "xperience10m_cosmos3_nano_future_order_misalignment_patched_textonly_20260621"
)
COSMOS_NANO_CURRENT_TASK_PROBE_DIR = (
    ROOT
    / "results/omni_finetune"
    / "xperience10m_cosmos3_nano_current_subtask_object_relevance_patched_textonly_20260621"
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
    "interaction_text_prediction": QWEN_INTERACTION_TEXT_PROBE_DIR / "interaction_text_prediction/metrics.json",
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
    "interaction_text_prediction": "macro_f1",
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
COSMOS_NANO_RETRIEVAL_TASK_METRIC_PATHS = {
    "hand_trajectory_forecast": COSMOS_NANO_RETRIEVAL_TASK_PROBE_DIR / "hand_trajectory_forecast/metrics.json",
    "caption_grounding": COSMOS_NANO_RETRIEVAL_TASK_PROBE_DIR / "caption_grounding/metrics.json",
    "imu_to_hand_pose": COSMOS_NANO_RETRIEVAL_TASK_PROBE_DIR / "imu_to_hand_pose/metrics.json",
    "camera_view_sync_retrieval": COSMOS_NANO_RETRIEVAL_TASK_PROBE_DIR / "camera_view_sync_retrieval/metrics.json",
}
COSMOS_NANO_RETRIEVAL_TASK_METRIC_KEYS = {
    "hand_trajectory_forecast": "hand_trajectory_forecast_mrr",
    "caption_grounding": "caption_grounding_mrr",
    "imu_to_hand_pose": "imu_to_hand_pose_mrr",
    "camera_view_sync_retrieval": "camera_view_sync_retrieval_mrr",
}
COSMOS_SUPER_FUTURE_TASK_METRIC_PATHS = {
    "temporal_order": COSMOS_SUPER_FUTURE_TASK_PROBE_DIR / "temporal_order/metrics.json",
    "misalignment_detection": COSMOS_SUPER_FUTURE_TASK_PROBE_DIR / "misalignment_detection/metrics.json",
    "next_subtask_forecast": COSMOS_SUPER_FUTURE_TASK_PROBE_DIR / "next_subtask_forecast/metrics.json",
    "object_set_forecast": COSMOS_SUPER_FUTURE_TASK_PROBE_DIR / "object_set_forecast/metrics.json",
}
COSMOS_SUPER_FUTURE_TASK_METRIC_KEYS = {
    "temporal_order": "temporal_order_f1",
    "misalignment_detection": "misalignment_detection_f1",
    "next_subtask_forecast": "next_subtask_forecast_macro_f1",
    "object_set_forecast": "object_set_forecast_micro_f1",
}
COSMOS_NANO_FUTURE_ORDER_TASK_METRIC_PATHS = {
    "temporal_order": COSMOS_NANO_FUTURE_ORDER_PROBE_DIR / "temporal_order/metrics.json",
    "misalignment_detection": COSMOS_NANO_FUTURE_ORDER_PROBE_DIR / "misalignment_detection/metrics.json",
}
COSMOS_NANO_FUTURE_ORDER_TASK_METRIC_KEYS = {
    "temporal_order": "temporal_order_f1",
    "misalignment_detection": "misalignment_detection_f1",
}
COSMOS_NANO_CURRENT_TASK_METRIC_PATHS = {
    "timeline_subtask": COSMOS_NANO_CURRENT_TASK_PROBE_DIR / "timeline_subtask/metrics.json",
    "object_relevance": COSMOS_NANO_CURRENT_TASK_PROBE_DIR / "object_relevance/metrics.json",
}
COSMOS_NANO_CURRENT_TASK_METRIC_KEYS = {
    "timeline_subtask": "timeline_subtask_macro_f1",
    "object_relevance": "object_relevance_micro_f1",
}
COSMOS_SUPER_INTERACTION_TEXT_TASK_METRIC_PATHS = {
    "interaction_text_prediction": COSMOS_SUPER_INTERACTION_TEXT_PROBE_DIR / "interaction_text_prediction/metrics.json",
}
COSMOS_SUPER_INTERACTION_TEXT_TASK_METRIC_KEYS = {
    "interaction_text_prediction": "macro_f1",
}
COSMOS_NANO_INTERACTION_TEXT_TASK_METRIC_PATHS = {
    "interaction_text_prediction": COSMOS_NANO_INTERACTION_TEXT_PROBE_DIR / "interaction_text_prediction/metrics.json",
}
COSMOS_NANO_INTERACTION_TEXT_TASK_METRIC_KEYS = {
    "interaction_text_prediction": "macro_f1",
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
        "cosmos3_nano_future_window": "timeline_subtask_macro_f1",
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
    "hand_trajectory_forecast": {
        "cosmos3_nano_future_window": "hand_trajectory_forecast_mrr",
    },
    "object_relevance": {
        "qwen3_omni_v6_lora": "object_micro_f1",
        "cosmos3_super_reasoner": "object_micro_f1",
        "cosmos3_nano_future_window": "object_relevance_micro_f1",
    },
    "action_object_relation": {
        "qwen3_omni_v6_lora": "action_object_relation_macro_f1",
        "cosmos3_super_reasoner": "action_object_relation_macro_f1",
        "cosmos3_nano_future_window": "action_object_relation_macro_f1",
    },
    "caption_grounding": {
        "cosmos3_super_reasoner": "caption_grounding_iou",
        "cosmos3_nano_future_window": "caption_grounding_mrr",
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
    "temporal_order": {
        "cosmos3_nano_future_window": "temporal_order_f1",
    },
    "misalignment_detection": {
        "cosmos3_nano_future_window": "misalignment_detection_f1",
    },
    "imu_to_hand_pose": {
        "cosmos3_nano_future_window": "imu_to_hand_pose_mrr",
    },
    "camera_view_sync_retrieval": {
        "cosmos3_nano_future_window": "camera_view_sync_retrieval_mrr",
    },
    "interaction_text_prediction": {
        "cosmos3_nano_future_window": "macro_f1",
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
    ("cosmos3_super_reasoner", "temporal_order"): COSMOS_SUPER_FUTURE_TASK_METRIC_PATHS["temporal_order"],
    ("cosmos3_super_reasoner", "misalignment_detection"): COSMOS_SUPER_FUTURE_TASK_METRIC_PATHS["misalignment_detection"],
    ("cosmos3_super_reasoner", "next_subtask_forecast"): COSMOS_SUPER_FUTURE_TASK_METRIC_PATHS["next_subtask_forecast"],
    ("cosmos3_super_reasoner", "object_set_forecast"): COSMOS_SUPER_FUTURE_TASK_METRIC_PATHS["object_set_forecast"],
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
    ("qwen3_omni_v6_lora", "interaction_text_prediction"): QWEN_FUTURE_TASK_METRIC_PATHS["interaction_text_prediction"],
    ("cosmos3_nano_future_window", "long_horizon_next_action"): COSMOS_NANO_LONG_HORIZON_METRICS_PATH,
    ("cosmos3_nano_future_window", "next_subtask_forecast"): COSMOS_NANO_NEXT_SUBTASK_METRICS_PATH,
    ("cosmos3_nano_future_window", "modality_reconstruction"): COSMOS_NANO_MODALITY_RECONSTRUCTION_METRICS_PATH,
    ("cosmos3_nano_future_window", "action_object_relation"): COSMOS_NANO_ACTION_OBJECT_METRICS_PATH,
    ("cosmos3_nano_future_window", "object_set_forecast"): COSMOS_NANO_OBJECT_SET_METRICS_PATH,
    ("cosmos3_nano_future_window", "time_to_transition"): COSMOS_NANO_TIME_TO_TRANSITION_METRICS_PATH,
    ("cosmos3_nano_future_window", "hand_trajectory_forecast"): COSMOS_NANO_RETRIEVAL_TASK_METRIC_PATHS["hand_trajectory_forecast"],
    ("cosmos3_nano_future_window", "caption_grounding"): COSMOS_NANO_RETRIEVAL_TASK_METRIC_PATHS["caption_grounding"],
    ("cosmos3_nano_future_window", "imu_to_hand_pose"): COSMOS_NANO_RETRIEVAL_TASK_METRIC_PATHS["imu_to_hand_pose"],
    ("cosmos3_nano_future_window", "camera_view_sync_retrieval"): COSMOS_NANO_RETRIEVAL_TASK_METRIC_PATHS["camera_view_sync_retrieval"],
    ("cosmos3_nano_future_window", "interaction_text_prediction"): COSMOS_NANO_INTERACTION_TEXT_TASK_METRIC_PATHS["interaction_text_prediction"],
    ("cosmos3_nano_future_window", "temporal_order"): COSMOS_NANO_FUTURE_ORDER_TASK_METRIC_PATHS["temporal_order"],
    ("cosmos3_nano_future_window", "misalignment_detection"): COSMOS_NANO_FUTURE_ORDER_TASK_METRIC_PATHS["misalignment_detection"],
    ("cosmos3_nano_future_window", "timeline_subtask"): COSMOS_NANO_CURRENT_TASK_METRIC_PATHS["timeline_subtask"],
    ("cosmos3_nano_future_window", "object_relevance"): COSMOS_NANO_CURRENT_TASK_METRIC_PATHS["object_relevance"],
    ("cosmos3_super_reasoner", "long_horizon_next_action"): COSMOS_SUPER_LONG_HORIZON_METRICS_PATH,
    ("cosmos3_super_reasoner", "time_to_transition"): COSMOS_SUPER_TIME_TO_TRANSITION_METRICS_PATH,
    ("cosmos3_super_reasoner", "hand_trajectory_forecast"): COSMOS_SUPER_RETRIEVAL_TASK_METRIC_PATHS["hand_trajectory_forecast"],
    ("cosmos3_super_reasoner", "cross_modal_retrieval"): COSMOS_SUPER_RETRIEVAL_TASK_METRIC_PATHS["cross_modal_retrieval"],
    ("cosmos3_super_reasoner", "modality_reconstruction"): COSMOS_SUPER_RETRIEVAL_TASK_METRIC_PATHS["modality_reconstruction"],
    ("cosmos3_super_reasoner", "imu_to_hand_pose"): COSMOS_SUPER_RETRIEVAL_TASK_METRIC_PATHS["imu_to_hand_pose"],
    ("cosmos3_super_reasoner", "camera_view_sync_retrieval"): COSMOS_SUPER_RETRIEVAL_TASK_METRIC_PATHS["camera_view_sync_retrieval"],
    ("cosmos3_super_reasoner", "interaction_text_prediction"): COSMOS_SUPER_INTERACTION_TEXT_TASK_METRIC_PATHS["interaction_text_prediction"],
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
    "cosmos3_super_reasoner": "Verified Cosmos3-Super base-weight Reasoner JSON-task evaluation, plus task 5/8/9/10/11/12/13/14/16/17/18/19/20 probes where public metrics exist.",
    "cosmos3_nano_future_window": "Verified Cosmos3-Nano future-window compatibility metrics, plus model-output probes for tasks 2/5/7/8/10/11/12/13/14/15/16/17/18/19 and a derived task-20 boundary timing probe scored from held-out future-window artifacts.",
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

RADAR_GROUP_SPECS = (
    {
        "id": "single_episode",
        "title": "Single-episode sample",
        "subtitle": "Public-sample simple and neural task heads.",
        "series_ids": ("minimal", "neural_mlp"),
    },
    {
        "id": "metadata_128",
        "title": "128-episode metadata/text",
        "subtitle": "Aligned JSONL metadata/text plus staged target blocks.",
        "series_ids": ("metadata128_simple", "metadata128_neural_mlp"),
    },
    {
        "id": "raw_128",
        "title": "128-episode raw features",
        "subtitle": "4430-dim sensor-block heads; proxy axes are flagged.",
        "series_ids": ("raw128_simple", "raw128_neural_mlp"),
    },
    {
        "id": "foundation_models",
        "title": "Foundation-model probes",
        "subtitle": "Verified Qwen3 and Cosmos task-specific outputs.",
        "series_ids": ("qwen3_omni_v6_lora", "cosmos3_super_reasoner", "cosmos3_nano_future_window"),
    },
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
    for task_id, path in COSMOS_SUPER_FUTURE_TASK_METRIC_PATHS.items():
        payload = read_json(path)
        metric_key = COSMOS_SUPER_FUTURE_TASK_METRIC_KEYS[task_id]
        metric_value = payload.get(metric_key)
        if payload.get("status") != "pass" or not isinstance(metric_value, (int, float)):
            continue
        cosmos_super_metrics[metric_key] = metric_value
        mapping.setdefault(task_id, {})["cosmos3_super_reasoner"] = metric_key
    for task_id, path in COSMOS_SUPER_INTERACTION_TEXT_TASK_METRIC_PATHS.items():
        payload = read_json(path)
        metric_key = COSMOS_SUPER_INTERACTION_TEXT_TASK_METRIC_KEYS[task_id]
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
    proxy_completion = bool(payload.get("proxy_completion"))
    return {
        "raw": score,
        "metric_key": payload.get("primary_metric"),
        "source": str(path.relative_to(ROOT)),
        "scope": payload.get("scope") or "multi_episode_128_aligned_baseline",
        "status": (
            "proxy_scored"
            if status == "pass" and score is not None and proxy_completion
            else "scored"
            if status == "pass" and score is not None
            else "unsupported_without_required_target"
        ),
        "reason": payload.get("reason")
        or payload.get("error")
        or payload.get("proxy_reason")
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
    def score_cell(value: dict[str, Any]) -> str:
        if value.get("normalized_score") is None:
            return status_label(value.get("status"))
        raw_text = str(value.get("raw_text") or "n/a")
        norm = value.get("normalized_score")
        norm_text = f"{float(norm):.3f}" if norm is not None else "n/a"
        metric_key = str(value.get("metric_key") or "metric")
        status = "proxy" if value.get("status") == "proxy_scored" else "direct"
        return f"{raw_text}<br><sub>{status}; norm {norm_text}; {metric_key}</sub>"

    lines = [
        "# Task Method 20-Result Matrix",
        "",
        "Every method has one record for each of the 20 unified task contracts. Numeric scores appear only where a committed runner or verified package produced that task target.",
        "",
        "Legend: `score` = direct numeric task score and `proxy` = documented compact substitute target. The current public matrix is complete at 180/180 scored records; unsupported/not-evaluated labels are retained only for future regression audits.",
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
            "## Compact Score Matrix",
            "",
            "Cells show `raw metric value`, then `direct/proxy; normalized radar value; metric key`. The raw metric is the value to cite; the normalized value is the exact linear 0-1 score retained in JSON. The SVG radar uses sqrt(normalized score) only for visual radius, so low but real differences remain visible without changing the table values.",
            "",
            "| # | Task | " + " | ".join(spec["short_label"] for spec in SERIES.values()) + " |",
            "| ---: | --- | " + " | ".join("---" for _ in SERIES) + " |",
        ]
    )
    for task in payload["tasks"]:
        cells = [score_cell(task["values"][series_id]) for series_id in SERIES]
        lines.append(f"| {task['task_number']:02d} | {task['label']} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Status Matrix",
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
    selected_groups = radar_groups_for_series(series_ids)
    chart_design = json.loads(json.dumps(payload.get("chart_design", {})))
    chart_design["method_count"] = len(series)
    chart_design["reason"] = (
        f"This split view has {len(series)} methods and {sum(record.get('result_record_count', 0) for record in series)} "
        "method-task records; grouped radar panels keep related methods readable while retaining the unified source matrix."
    )
    chart_design["groups"] = [
        {
            "id": group["id"],
            "title": group["title"],
            "series_ids": list(group["series_ids"]),
        }
        for group in selected_groups
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
        "chart_design": chart_design,
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
    for metrics_path in COSMOS_NANO_RETRIEVAL_TASK_METRIC_PATHS.values():
        cosmos_nano.update(read_json(metrics_path))
    for metrics_path in COSMOS_NANO_CURRENT_TASK_METRIC_PATHS.values():
        cosmos_nano.update(read_json(metrics_path))
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
            source_path = FOUNDATION_METRIC_SOURCE_OVERRIDES.get(
                (series_id, row["task_id"]),
                FOUNDATION_METRIC_PATHS[series_id],
            )
            source_metrics = (
                read_json(source_path)
                if (series_id, row["task_id"]) in FOUNDATION_METRIC_SOURCE_OVERRIDES
                else foundation_metrics.get(series_id, {})
            )
            raw = source_metrics.get(metric_key)
            values[series_id] = {
                "raw": raw,
                "metric_key": metric_key,
                "source": str(source_path.relative_to(ROOT)),
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
                "provenance_source": row.get("provenance_source"),
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
                "plotted_as": "grouped small-multiple radar panel with direct legend and coverage badges",
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
            "radar_visual_radius": "SVG radar panels use sqrt(normalized_score) for radius so polygon area remains closer to the score and low-valued but real differences stay visible; the JSON and matrix retain exact linear normalized_score values",
            "result_record_policy": "every method has 20 task records; the current public release has 180/180 scored rows with proxy flags and reasons retained where compact substitute targets are used",
            "foundation_model_overlay": "Qwen3-Omni and Cosmos3 are grouped in the foundation-model radar panel. All current public model rows have 20 scored task records, with source paths retained for every metric.",
            "metadata_128_overlay": "128-episode aligned baselines are grouped in the metadata/text radar panel. Numeric scores come from JSONL metadata/text tasks plus staged sensor-block targets when the processed target exists.",
            "raw_128_overlay": "128-episode raw-feature baselines are grouped in the raw-feature radar panel. Eighteen axes use direct task targets; interaction text and camera-view sync are completed with documented compact proxies because raw interaction strings and paired video-view embeddings are absent from the 128 export.",
        },
        "chart_design": {
            "mode": "grouped_small_multiples",
            "method_count": len(SERIES),
            "reason": "The public release has nine methods and 180 scored records; small-multiple radar panels avoid a nine-polygon overlay while keeping every method visible.",
            "groups": [
                {
                    "id": group["id"],
                    "title": group["title"],
                    "series_ids": list(group["series_ids"]),
                }
                for group in RADAR_GROUP_SPECS
            ],
            "visual_radius_transform": "sqrt(normalized_score)",
            "exact_value_source": "docs/data/task_method_20_result_matrix.json",
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
                "coverage": f"20 records / {next(item for item in series_records if item['id'] == 'raw128_simple')['scored_task_count']} scored records; 18 direct + 2 proxy",
                "headline": "34,269 windows; centroid/ridge heads over 4430-dim sensor blocks",
                "source": str((RAW128_BASELINE_DIR / "run_summary_all.json").relative_to(ROOT)),
            },
            {
                "id": "raw128_neural_mlp",
                "title": "128ep Raw NN",
                "status": "a100_raw20_complete_with_documented_proxies",
                "coverage": f"20 records / {next(item for item in series_records if item['id'] == 'raw128_neural_mlp')['scored_task_count']} scored records; 18 direct + 2 proxy",
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


def svg_shape(
    tag: str,
    points: list[tuple[float, float]],
    *,
    fill: str,
    fill_opacity: float,
    stroke: str,
    stroke_opacity: float = 0.92,
    stroke_width: float = 2.0,
    dash: str | None = None,
) -> str:
    coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<{tag} points="{coords}" fill="{fill}" fill-opacity="{fill_opacity:.3f}" '
        f'stroke="{stroke}" stroke-opacity="{stroke_opacity:.3f}" stroke-width="{stroke_width:.1f}" '
        f'stroke-linejoin="round" stroke-linecap="round"{dash_attr}/>'
    )


def radar_radius(score: float | None, radius: float) -> float | None:
    if score is None:
        return None
    return radius * math.sqrt(clamp01(float(score)))


def radar_groups_for_series(series_ids: tuple[str, ...]) -> list[dict[str, Any]]:
    selected = set(series_ids)
    groups: list[dict[str, Any]] = []
    assigned: set[str] = set()
    for group in RADAR_GROUP_SPECS:
        present = tuple(series_id for series_id in group["series_ids"] if series_id in selected)
        if not present:
            continue
        groups.append({**group, "series_ids": present})
        assigned.update(present)
    remaining = tuple(series_id for series_id in series_ids if series_id not in assigned)
    if remaining:
        groups.append(
            {
                "id": "other_methods",
                "title": "Other methods",
                "subtitle": "Additional method rows retained from the matrix.",
                "series_ids": remaining,
            }
        )
    return groups


def draw_radar_grid(
    parts: list[str],
    *,
    cx: float,
    cy: float,
    radius: float,
    tasks: list[dict[str, Any]],
    angles: list[float],
    label_size: int,
) -> None:
    for value in (0.05, 0.25, 0.50, 0.75, 1.0):
        ring_radius = radius * math.sqrt(value)
        ring = [point(cx, cy, ring_radius, angle) for angle in angles]
        parts.append(
            svg_shape(
                "polygon",
                ring,
                fill="none",
                fill_opacity=0,
                stroke="#ccffa0",
                stroke_opacity=0.16,
                stroke_width=1.0,
            )
        )
        parts.append(svg_text(cx + 8, cy - ring_radius + 4, f"{value:.2g}", size=max(9, label_size - 2), fill="#a5afa2", weight=620, opacity=0.72))
    for task, angle in zip(tasks, angles):
        x, y = point(cx, cy, radius, angle)
        parts.append(f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{x:.1f}" y2="{y:.1f}" stroke="#ccffa0" stroke-opacity="0.11" stroke-width="1"/>')
        lx, ly = point(cx, cy, radius + 28, angle)
        proxy = task["task_id"] in PROXY_TASK_IDS
        color = "#f472b6" if proxy else "#ccffa0"
        parts.append(f'<circle cx="{lx:.1f}" cy="{ly:.1f}" r="{label_size + 2:.1f}" fill="{color}" fill-opacity="0.12" stroke="{color}" stroke-opacity="0.34"/>')
        parts.append(svg_text(lx, ly + label_size * 0.33, f"{task['task_number']:02d}", size=label_size, fill=color, anchor="middle", weight=850, opacity=0.98))


def draw_radar_series(
    parts: list[str],
    *,
    cx: float,
    cy: float,
    radius: float,
    tasks: list[dict[str, Any]],
    angles: list[float],
    series_id: str,
    stroke_width: float,
    fill_opacity: float,
) -> None:
    spec = SERIES[series_id]
    valid_points: list[tuple[float, float]] = []
    scored_count = 0
    for task, angle in zip(tasks, angles):
        value = task["values"].get(series_id, {})
        score = value.get("normalized_score")
        plotted_radius = radar_radius(score, radius)
        if plotted_radius is None:
            continue
        scored_count += 1
        valid_points.append(point(cx, cy, plotted_radius, angle))
    if len(valid_points) >= 3 and scored_count == len(tasks):
        parts.append(
            svg_shape(
                "polygon",
                valid_points,
                fill=spec["color"],
                fill_opacity=fill_opacity,
                stroke=spec["color"],
                stroke_width=stroke_width,
                dash=spec.get("stroke_dasharray"),
            )
        )
    elif len(valid_points) >= 2:
        parts.append(
            svg_shape(
                "polyline",
                valid_points,
                fill="none",
                fill_opacity=0,
                stroke=spec["color"],
                stroke_width=stroke_width,
                dash=spec.get("stroke_dasharray"),
            )
        )
    for task, angle in zip(tasks, angles):
        value = task["values"].get(series_id, {})
        plotted_radius = radar_radius(value.get("normalized_score"), radius)
        if plotted_radius is None:
            continue
        px, py = point(cx, cy, plotted_radius, angle)
        proxy = value.get("status") == "proxy_scored"
        parts.append(
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{5.5 if proxy else 4.4:.1f}" '
            f'fill="{spec["color"]}" fill-opacity="0.95" stroke="{"#f4f8ef" if proxy else "#020502"}" '
            f'stroke-width="{2.1 if proxy else 1.3:.1f}"/>'
        )


def draw_radar_panel(
    parts: list[str],
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    group: dict[str, Any],
    payload: dict[str, Any],
    series_record_by_id: dict[str, dict[str, Any]],
    large: bool = False,
) -> None:
    tasks = payload["tasks"]
    angles = [-math.pi / 2 + 2 * math.pi * i / len(tasks) for i in range(len(tasks))]
    panel_bg = "#071007"
    parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" rx="18" fill="{panel_bg}" fill-opacity="0.90" stroke="#ccffa0" stroke-opacity="0.22"/>')
    parts.append(svg_text(x + 28, y + 44, str(group["title"]), size=26 if large else 20, weight=850))
    parts.append(svg_text(x + 28, y + 74, str(group["subtitle"]), size=14 if large else 12, fill="#a5afa2", weight=600))

    if large:
        cx = x + width * 0.39
        cy = y + height * 0.56
        radius = min(width * 0.20, height * 0.34)
        legend_x = x + width * 0.68
        legend_y = y + 160
        label_size = 12
    else:
        cx = x + width * 0.38
        cy = y + height * 0.57
        radius = min(width * 0.18, height * 0.30)
        legend_x = x + width * 0.67
        legend_y = y + 122
        label_size = 8

    draw_radar_grid(parts, cx=cx, cy=cy, radius=radius, tasks=tasks, angles=angles, label_size=label_size)

    series_ids = tuple(group["series_ids"])
    fill_opacity = 0.065 if len(series_ids) <= 2 else 0.040
    for idx, series_id in enumerate(series_ids):
        draw_radar_series(
            parts,
            cx=cx,
            cy=cy,
            radius=radius,
            tasks=tasks,
            angles=angles,
            series_id=series_id,
            stroke_width=4.3 if large else 3.2,
            fill_opacity=max(0.026, fill_opacity - idx * 0.010),
        )

    parts.append(svg_text(legend_x, legend_y - 34, "Methods", size=17 if large else 14, fill="#ccffa0", weight=850))
    for idx, series_id in enumerate(series_ids):
        record = series_record_by_id[series_id]
        color = record["color"]
        row_y = legend_y + idx * (92 if large else 74)
        parts.append(f'<line x1="{legend_x:.1f}" y1="{row_y:.1f}" x2="{legend_x + 58:.1f}" y2="{row_y:.1f}" stroke="{color}" stroke-width="{6 if large else 5}" stroke-linecap="round" stroke-dasharray="{record.get("stroke_dasharray") or ""}"/>')
        parts.append(f'<circle cx="{legend_x + 29:.1f}" cy="{row_y:.1f}" r="{6 if large else 5}" fill="{color}" stroke="#020502" stroke-width="1.5"/>')
        parts.append(svg_text(legend_x + 74, row_y + 5, record["label"], size=15 if large else 12, weight=850))
        coverage = f"{record['scored_task_count']}/20 scored"
        proxy = record.get("proxy_scored_task_count", 0)
        if proxy:
            coverage += f" · {proxy} proxy"
        parts.append(svg_text(legend_x + 74, row_y + (28 if large else 22), coverage, size=12 if large else 10, fill=color, weight=800))
        detail = split_text(METHOD_DETAILS.get(series_id, record["scope"]), 50 if large else 44)[:2]
        parts.extend(svg_text_lines(legend_x + 74, row_y + (49 if large else 40), detail, size=10 if large else 8, fill="#a5afa2", weight=560, line_height=13 if large else 10))

    parts.append(svg_text(x + 28, y + height - 30, "Radius = sqrt(normalized score); exact raw and normalized values are in the matrix.", size=11 if large else 9, fill="#a5afa2", weight=600, opacity=0.88))


def draw_task_key(parts: list[str], *, x: float, y: float, width: float, tasks: list[dict[str, Any]], compact: bool = False) -> None:
    height = 292 if not compact else 250
    parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" rx="16" fill="#020502" fill-opacity="0.62" stroke="#ccffa0" stroke-opacity="0.18"/>')
    parts.append(svg_text(x + 28, y + 42, "20-task axis key", size=20, weight=850))
    parts.append(svg_text(x + 250, y + 42, "Task numbers stay on the radar; full names and proxy axes stay here.", size=13, fill="#a5afa2", weight=600))
    col_count = 4
    col_w = (width - 56) / col_count
    row_h = 42 if not compact else 36
    for idx, task in enumerate(tasks):
        col = idx // 5
        row = idx % 5
        x0 = x + 28 + col * col_w
        y0 = y + 84 + row * row_h
        proxy = task["task_id"] in PROXY_TASK_IDS
        color = "#f472b6" if proxy else "#ccffa0"
        parts.append(f'<rect x="{x0:.1f}" y="{y0 - 17:.1f}" width="35" height="25" rx="6" fill="{color}" fill-opacity="0.13" stroke="{color}" stroke-opacity="0.40"/>')
        parts.append(svg_text(x0 + 17.5, y0 + 1, f"{task['task_number']:02d}", size=10, fill=color, anchor="middle", weight=850))
        task_name = str(task["label"])
        if len(task_name) > 34:
            task_name = task_name[:31].rstrip() + "..."
        parts.append(svg_text(x0 + 46, y0 - 2, task_name, size=11 if not compact else 10, fill="#f4f8ef", weight=800))
        metric = str(task.get("metric_name") or task.get("metric_key") or "")
        direction = "lower" if task.get("metric_direction") == "lower" else "higher"
        metric_text = f"{metric}; {direction} better"
        if proxy:
            metric_text += "; proxy axis"
        if len(metric_text) > 43:
            metric_text = metric_text[:40].rstrip() + "..."
        parts.append(svg_text(x0 + 46, y0 + 16, metric_text, size=9, fill="#a5afa2", weight=560))


def draw_reading_rules(parts: list[str], *, y: float, reading_rules: tuple[str, str, str] | None) -> None:
    if reading_rules is None:
        reading_rules = (
            "Use the panels for shape and coverage; use docs/data/task_method_20_result_matrix.json for exact ranks, raw values, direct/proxy flags, and sources.",
            "The old nine-method overlay was replaced by grouped small multiples so each radar compares only related methods.",
            "SVG radius uses sqrt(normalized_score) for readable area; JSON normalized_score remains linear and unchanged.",
        )
    parts.append(f'<rect x="70" y="{y:.1f}" width="2260" height="118" rx="14" fill="#020502" fill-opacity="0.62" stroke="#ccffa0" stroke-opacity="0.16"/>')
    parts.append(svg_text(100, y + 33, "Reading rules", size=16, fill="#ccffa0", weight=850))
    parts.append(svg_text(230, y + 33, reading_rules[0], size=13, fill="#dce8d7", weight=650))
    parts.append(svg_text(230, y + 61, reading_rules[1], size=12, fill="#a5afa2", weight=560))
    parts.append(svg_text(230, y + 87, reading_rules[2], size=12, fill="#a5afa2", weight=560))


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
    del polygon_series_ids
    width, height = 2400, 1900
    tasks = payload["tasks"]
    if series_ids is None:
        series_ids = tuple(record["id"] for record in payload["series"])
    groups = radar_groups_for_series(series_ids)
    series_record_by_id = {record["id"]: record for record in payload["series"]}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        '<filter id="softGlow"><feGaussianBlur stdDeviation="4" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>',
        '<pattern id="dots" width="22" height="22" patternUnits="userSpaceOnUse"><circle cx="2" cy="2" r="1.15" fill="#ccffa0" opacity="0.12"/></pattern>',
        "</defs>",
        '<rect width="100%" height="100%" fill="#020502"/>',
        '<rect width="100%" height="100%" fill="url(#dots)" opacity="0.40"/>',
        '<rect x="28" y="28" width="2344" height="1844" rx="22" fill="#061006" fill-opacity="0.90" stroke="#ccffa0" stroke-opacity="0.22"/>',
        svg_text(70, 86, title or payload.get("title", "20-Task Model Radar"), size=36, weight=850),
        svg_text(
            70,
            122,
            subtitle or "Grouped small-multiple radars for the nine-method, 180-result comparison.",
            size=18,
            fill="#dce8d7",
            weight=650,
        ),
        svg_text(
            70,
            150,
            context_line
            or "Related methods are compared in separate panels to avoid the unreadable nine-polygon overlay.",
            size=15,
            fill="#a5afa2",
            weight=560,
        ),
    ]

    if chip_specs is None:
        chip_specs = [
            ("20 task axes", "#ccffa0"),
            (f"{payload['method_task_record_count']} method-task records", "#67e8d1"),
            (f"{payload['scored_method_task_count']} scored records", "#22d3ee"),
            ("grouped small multiples", "#f59e0b"),
            ("sqrt visual radius", "#f472b6"),
        ]
    chip_x = 70
    for label, color in chip_specs:
        chip_w = max(128, min(280, 18 + len(label) * 8.3))
        parts.append(f'<rect x="{chip_x:.1f}" y="174" width="{chip_w:.1f}" height="34" rx="17" fill="{color}" fill-opacity="0.10" stroke="{color}" stroke-opacity="0.38"/>')
        parts.append(svg_text(chip_x + 16, 197, label, size=13, fill=color, weight=780))
        chip_x += chip_w + 12

    if len(groups) == 1:
        draw_radar_panel(
            parts,
            x=70,
            y=242,
            width=2260,
            height=1040,
            group=groups[0],
            payload=payload,
            series_record_by_id=series_record_by_id,
            large=True,
        )
        key_y = 1322
    elif len(groups) == 3:
        panel_w, panel_h = 1100, 545
        start_x, start_y = 70, 248
        gap_x, gap_y = 30, 34
        for idx, group in enumerate(groups[:2]):
            draw_radar_panel(
                parts,
                x=start_x + idx * (panel_w + gap_x),
                y=start_y,
                width=panel_w,
                height=panel_h,
                group=group,
                payload=payload,
                series_record_by_id=series_record_by_id,
            )
        draw_radar_panel(
            parts,
            x=start_x,
            y=start_y + panel_h + gap_y,
            width=panel_w * 2 + gap_x,
            height=panel_h,
            group=groups[2],
            payload=payload,
            series_record_by_id=series_record_by_id,
            large=True,
        )
        key_y = 1438
    else:
        panel_w, panel_h = 1100, 545
        start_x, start_y = 70, 248
        gap_x, gap_y = 30, 34
        for idx, group in enumerate(groups):
            col = idx % 2
            row = idx // 2
            draw_radar_panel(
                parts,
                x=start_x + col * (panel_w + gap_x),
                y=start_y + row * (panel_h + gap_y),
                width=panel_w,
                height=panel_h,
                group=group,
                payload=payload,
                series_record_by_id=series_record_by_id,
            )
        key_y = 1438

    draw_task_key(parts, x=70, y=key_y, width=2260, tasks=tasks, compact=len(groups) == 1)
    draw_reading_rules(parts, y=1750 if len(groups) > 1 else 1632, reading_rules=reading_rules)
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
        description="Selected 128-episode metadata/raw baselines plus verified Qwen3-Omni v6, Cosmos3-Super, and Cosmos3-Nano diagnostics. Every method has 20 records; numeric scores appear only where the public artifact produced that task target.",
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
            context_line="This view isolates the 1-episode task-head setup from the selected-128 model diagnostics.",
            chip_specs=[
                ("20 task axes", "#ccffa0"),
                ("40 method-task records", "#67e8d1"),
                ("40 scored records", "#22d3ee"),
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
            subtitle="Selected 96/16/16 episode split; all seven 128-episode rows score all 20 axes.",
            context_line="Metadata, raw-feature, and foundation-model methods are separated into grouped radar panels instead of one crowded overlay.",
            chip_specs=[
                ("20 task axes", "#ccffa0"),
                ("140 method-task records", "#67e8d1"),
                (f"{episode128_payload['scored_method_task_count']} scored records", "#22d3ee"),
                ("40/40 raw128 pass", "#f59e0b"),
                ("0 scoreless", "#f472b6"),
            ],
            reading_rules=(
                "Every 128-episode method has 20 result records and all 140 rows are scored in this split radar.",
                "Raw128 Simple and Raw128 NN are complete 20/20 scored multi-episode baselines; tasks 15/19 are documented compact proxies and are marked in the task key.",
                "Qwen3-Omni and Cosmos3 rows use verified held-out outputs or derived probe artifacts; source paths stay in the matrix JSON.",
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

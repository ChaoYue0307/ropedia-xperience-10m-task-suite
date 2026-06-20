"""Canonical reader-facing names for public task-suite artifacts."""

from __future__ import annotations

from typing import Any


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
    "long_horizon_next_action": "Long-Horizon Next-Action Forecasting",
    "next_subtask_forecast": "Long-Horizon Next-Subtask Forecasting",
    "interaction_text_prediction": "Interaction Text Prediction",
    "action_object_relation": "Action-Object Relation Prediction",
    "object_set_forecast": "Future Object-Set Forecasting",
    "imu_to_hand_pose": "IMU-to-Hand Pose Reconstruction",
    "camera_view_sync_retrieval": "Camera-View Synchronization Retrieval",
    "time_to_transition": "Time-to-Next-Transition Regression",
}


def task_display_name(task_id: str) -> str:
    """Return the public label while preserving unknown ids for debugging."""
    return TASK_DISPLAY_NAMES.get(task_id, task_id.replace("_", " ").title())


def with_task_display(task_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    payload.setdefault("task", task_id)
    payload["task_display_name"] = task_display_name(task_id)
    return payload

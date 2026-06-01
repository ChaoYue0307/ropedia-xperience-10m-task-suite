#!/usr/bin/env python3
"""Generate junior-friendly walkthroughs for each Xperience-10M task."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

from research_direction_taxonomy import METRIC_SPECS, fmt_metric


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "episode_task_suite"
OUT_DIR = RESULTS / "task_walkthroughs"
DOCS_DATA = ROOT / "docs" / "data"
SUMMARY_REPORT = RESULTS / "summary_report.json"

TASK_PRESENTATION: dict[str, dict[str, Any]] = {
    "timeline_action": {
        "display_name": "Action Recognition",
        "research_name": "Egocentric Action Recognition",
        "task_family": "supervised",
        "architecture_family": "multiclass classifier",
        "primary_direction": "C. Egocentric Vision & Interaction",
        "card_blurb": "Recognize the current manipulation action from synchronized visual, motion, inertial, pose, and annotation context.",
        "input_short": "20-frame multimodal window",
        "process_short": "window features -> action label builder -> classifier",
        "output_short": "current action class",
        "modalities": ["video", "depth", "pose_slam", "motion_capture", "inertial", "language"],
        "poster_modality": "video",
    },
    "timeline_subtask": {
        "display_name": "Procedure Step Recognition",
        "research_name": "Temporal Subtask Recognition",
        "task_family": "supervised",
        "architecture_family": "multiclass classifier",
        "primary_direction": "C. Egocentric Vision & Interaction",
        "card_blurb": "Recognize the broader activity stage so fine actions become a readable procedure timeline.",
        "input_short": "20-frame multimodal window",
        "process_short": "window features -> subtask label builder -> classifier",
        "output_short": "current procedure step",
        "modalities": ["video", "depth", "pose_slam", "motion_capture", "inertial", "language"],
        "poster_modality": "language",
    },
    "transition_detection": {
        "display_name": "Action Boundary Detection",
        "research_name": "Temporal Action Segmentation",
        "task_family": "diagnostic",
        "architecture_family": "binary classifier",
        "primary_direction": "C. Egocentric Vision & Interaction",
        "card_blurb": "Detect the local moment where the episode changes from one action segment to the next.",
        "input_short": "current window with boundary target",
        "process_short": "action changes -> boundary labels -> binary classifier",
        "output_short": "boundary or steady",
        "modalities": ["video", "pose_slam", "motion_capture", "inertial", "language"],
        "poster_modality": "pose_slam",
    },
    "next_action": {
        "display_name": "Next-Action Prediction",
        "research_name": "Short-Horizon Intention Prediction",
        "task_family": "supervised",
        "architecture_family": "future-label classifier",
        "primary_direction": "C. Egocentric Vision & Interaction",
        "card_blurb": "Forecast the near-future action from the current observations only.",
        "input_short": "current window at time t",
        "process_short": "current features -> future label shift -> classifier",
        "output_short": "action at t+20 frames",
        "modalities": ["video", "depth", "pose_slam", "motion_capture", "inertial"],
        "poster_modality": "video",
    },
    "hand_trajectory_forecast": {
        "display_name": "Hand Trajectory Forecasting",
        "research_name": "3D Hand Motion Forecasting",
        "task_family": "forecast",
        "architecture_family": "continuous regressor",
        "primary_direction": "A. Human Modeling & Motion Understanding",
        "card_blurb": "Predict the future 3D left/right hand path from the current multimodal state.",
        "input_short": "current multimodal window",
        "process_short": "current features -> future mocap target -> regression head",
        "output_short": "future hand-joint trajectory",
        "modalities": ["motion_capture", "video", "depth", "pose_slam", "inertial"],
        "poster_modality": "motion_capture",
    },
    "contact_prediction": {
        "display_name": "Contact State Prediction",
        "research_name": "Human-Object Contact Prediction",
        "task_family": "supervised",
        "architecture_family": "binary classifier",
        "primary_direction": "A. Human Modeling & Motion Understanding",
        "card_blurb": "Predict whether body or hand contact with the scene is occurring without leaking contact labels.",
        "input_short": "non-contact, non-caption features",
        "process_short": "feature filter -> contact target -> binary classifier",
        "output_short": "contact or no contact",
        "modalities": ["motion_capture", "video", "depth", "inertial"],
        "poster_modality": "motion_capture",
    },
    "object_relevance": {
        "display_name": "Object Relevance Prediction",
        "research_name": "Object-Centric Interaction Recognition",
        "task_family": "supervised",
        "architecture_family": "multi-label classifier",
        "primary_direction": "C. Egocentric Vision & Interaction",
        "card_blurb": "Infer which objects are relevant to the current manipulation window from non-caption features.",
        "input_short": "non-caption multimodal features",
        "process_short": "object vocabulary -> multi-hot labels -> sigmoid heads",
        "output_short": "relevant object set",
        "modalities": ["video", "depth", "pose_slam", "motion_capture", "inertial"],
        "poster_modality": "video",
    },
    "caption_grounding": {
        "display_name": "Language Grounding",
        "research_name": "Language-to-Moment Grounding",
        "task_family": "retrieval",
        "architecture_family": "retrieval ranker",
        "primary_direction": "C. Egocentric Vision & Interaction",
        "card_blurb": "Retrieve the matching time window for an annotation-derived text query.",
        "input_short": "text-like query and candidate windows",
        "process_short": "query features -> candidate index -> cosine ranker",
        "output_short": "ranked matching moments",
        "modalities": ["language", "video", "depth", "pose_slam"],
        "poster_modality": "language",
    },
    "cross_modal_retrieval": {
        "display_name": "Cross-Modal Retrieval",
        "research_name": "Multimodal Representation Retrieval",
        "task_family": "retrieval",
        "architecture_family": "two-tower retrieval head",
        "primary_direction": "D. Scene Reconstruction & World Modeling",
        "card_blurb": "Use motion, IMU, and camera-pose signals to retrieve the matching depth/video window.",
        "input_short": "motion/IMU/pose query; depth/video candidates",
        "process_short": "modality split -> projection -> nearest-neighbor ranker",
        "output_short": "ranked visual windows",
        "modalities": ["motion_capture", "inertial", "pose_slam", "depth", "video"],
        "poster_modality": "depth",
    },
    "modality_reconstruction": {
        "display_name": "Cross-Modal Reconstruction",
        "research_name": "Modality Feature Reconstruction",
        "task_family": "forecast",
        "architecture_family": "feature regressor",
        "primary_direction": "B. 3D/4D Reconstruction & Neural Rendering",
        "card_blurb": "Predict compressed depth/video feature vectors from motion, IMU, and camera-pose features.",
        "input_short": "motion, IMU, and camera/pose features",
        "process_short": "source-target split -> scaler -> regression head",
        "output_short": "reconstructed depth/video vector",
        "modalities": ["motion_capture", "inertial", "pose_slam", "depth", "video"],
        "poster_modality": "depth",
    },
    "temporal_order": {
        "display_name": "Temporal Order Verification",
        "research_name": "Temporal Order Verification",
        "task_family": "diagnostic",
        "architecture_family": "pairwise classifier",
        "primary_direction": "D. Scene Reconstruction & World Modeling",
        "card_blurb": "Tell whether two neighboring windows are in chronological order or reversed.",
        "input_short": "two adjacent windows plus difference vector",
        "process_short": "pair builder -> feature combiner -> binary classifier",
        "output_short": "correct or reversed",
        "modalities": ["video", "pose_slam", "motion_capture", "inertial"],
        "poster_modality": "video",
    },
    "misalignment_detection": {
        "display_name": "Multimodal Synchronization Detection",
        "research_name": "Cross-Modal Misalignment Detection",
        "task_family": "diagnostic",
        "architecture_family": "pairwise classifier",
        "primary_direction": "B. 3D/4D Reconstruction & Neural Rendering",
        "card_blurb": "Detect whether motion and visual/depth streams have been artificially shifted out of sync.",
        "input_short": "motion-side and visual/depth-side feature groups",
        "process_short": "aligned/shifted pairs -> feature combiner -> binary classifier",
        "output_short": "aligned or shifted",
        "modalities": ["motion_capture", "inertial", "video", "depth", "pose_slam"],
        "poster_modality": "pose_slam",
    },
}


TASK_WALKTHROUGHS: OrderedDict[str, dict[str, Any]] = OrderedDict(
    [
        (
            "timeline_action",
            {
                "plain_goal": "Look at one short multimodal window and name what action is happening now.",
                "case_study": "In the coffee-making sample, if the 20-frame window is during a pouring moment, the task asks the model to output an action such as Pour coffee or Pour milk into coffee.",
                "input": "One 20-frame window represented by the current 8,378-d feature vector: video/depth summaries, pose, SLAM/camera pose, motion capture, IMU, calibration, and language-derived context.",
                "middle_modules": [
                    "Window builder slices the episode into short overlapping windows.",
                    "Feature assembler concatenates all current feature blocks.",
                    "Label builder reads the action annotation for the center of the window.",
                    "Classifier head maps the window vector to one action class.",
                    "Evaluator compares predicted action labels against the held-out chronological segment.",
                ],
                "output": "A single action class for the current window.",
                "junior_tip": "This is like asking: given this tiny movie clip plus sensor readings, what is the person doing right now?",
                "failure_mode": "The one-episode chronological split contains future action classes that were not present in training, so low test macro-F1 is expected.",
            },
        ),
        (
            "timeline_subtask",
            {
                "plain_goal": "Predict the higher-level task stage for the current window.",
                "case_study": "A pouring action may belong to a broader subtask such as preparing or pouring a drink. The model predicts that broader stage instead of a fine action.",
                "input": "The same all-modality 8,378-d window vector used by action recognition.",
                "middle_modules": [
                    "Window builder creates the current temporal slice.",
                    "Feature assembler keeps all available modality blocks.",
                    "Subtask label builder maps the current timestamp to a subtask annotation.",
                    "Classifier head predicts the subtask class.",
                    "Evaluator reports class-balanced scores so rare subtasks matter.",
                ],
                "output": "A single subtask label for the current window.",
                "junior_tip": "Action is the verb; subtask is the chapter of the activity.",
                "failure_mode": "Single-episode ordering means some later subtasks appear only in test, so this is a pipeline check rather than a general benchmark.",
            },
        ),
        (
            "transition_detection",
            {
                "plain_goal": "Detect whether the current window is near a boundary between actions.",
                "case_study": "When the demonstrator changes from preparing to pouring, the model should flag a boundary instead of a steady action window.",
                "input": "One all-modality window vector plus labels derived from action-change timestamps.",
                "middle_modules": [
                    "Boundary builder scans action labels over time and marks windows near a change.",
                    "Feature assembler supplies all current modality features.",
                    "Binary classifier predicts steady vs boundary.",
                    "Boundary matcher checks whether predicted boundary times are close to true boundary times.",
                    "Evaluator reports macro-F1 and timing error, not just accuracy.",
                ],
                "output": "A binary label: boundary or steady.",
                "junior_tip": "This is the model's way of saying: something just changed here.",
                "failure_mode": "Boundaries are rare, so high accuracy can be misleading if the model predicts steady too often.",
            },
        ),
        (
            "next_action",
            {
                "plain_goal": "Use the current window to guess the action that will happen shortly after it.",
                "case_study": "If a window shows the person preparing to pour, the target can be the action 20 frames later, such as the start of pouring.",
                "input": "The current all-modality window vector at time t.",
                "middle_modules": [
                    "Window builder picks a current time window.",
                    "Future label builder shifts the action target by 20 frames.",
                    "Feature assembler uses only current information, not future features.",
                    "Classifier head predicts the future action class.",
                    "Evaluator checks whether the future action label is correct.",
                ],
                "output": "A single action class for t+20 frames.",
                "junior_tip": "This is short-horizon intention prediction: what will the person do next?",
                "failure_mode": "The public sample has unseen future classes in the chronological test split, which makes this very hard with one episode.",
            },
        ),
        (
            "hand_trajectory_forecast",
            {
                "plain_goal": "Predict where the hands will move over the next few frames.",
                "case_study": "When the hand is moving toward a cup or bottle, the model predicts the future 3D hand-joint path.",
                "input": "The current all-modality window vector at time t.",
                "middle_modules": [
                    "Window builder chooses the current sensor window.",
                    "Target builder extracts future left/right hand 3D joints from motion capture.",
                    "Regression head predicts a continuous trajectory, not a class label.",
                    "Output reshaper interprets the vector as future frames and joints.",
                    "Evaluator computes MPJPE, the average 3D joint-position error.",
                ],
                "output": "A future trajectory vector for left and right hand joints.",
                "junior_tip": "Instead of naming an action, this task draws the next hand path in 3D.",
                "failure_mode": "It is still a window-level forecast, not a full policy or long-horizon motion generator.",
            },
        ),
        (
            "contact_prediction",
            {
                "plain_goal": "Predict whether the body or hand is in contact with something.",
                "case_study": "During manipulation, the hand may touch a cup, table, or bottle. The task asks whether any contact is happening.",
                "input": "Non-contact and non-caption feature blocks, so the answer is not directly leaked from the target labels.",
                "middle_modules": [
                    "Feature selector removes contact-label and caption-label blocks.",
                    "Target builder converts contact annotations into a binary label.",
                    "Binary classifier predicts contact vs no contact.",
                    "Evaluator reports macro-F1 and accuracy.",
                    "Degeneracy checker records whether only one class appears.",
                ],
                "output": "A binary contact label.",
                "junior_tip": "This is a simple physical-interaction probe: is the person touching something now?",
                "failure_mode": "The current public sample is degenerate for this task because one class dominates, so perfect score does not mean the model learned contact physics.",
            },
        ),
        (
            "object_relevance",
            {
                "plain_goal": "Predict which objects matter in the current window.",
                "case_study": "If the person is pouring milk into coffee, relevant objects may include milk, cup, coffee, or container-like items.",
                "input": "Non-caption feature blocks, so the model must infer objects from sensors rather than copying the caption words.",
                "middle_modules": [
                    "Object vocabulary builder collects object labels from annotations.",
                    "Feature selector removes caption-derived label blocks.",
                    "Multi-label target builder creates a multi-hot object vector.",
                    "Sigmoid heads predict each object's relevance independently.",
                    "Evaluator reports micro-F1 and exact-match quality.",
                ],
                "output": "A multi-label object set for the current window.",
                "junior_tip": "A window can involve more than one object, so this is not a one-class classifier.",
                "failure_mode": "Object labels are sparse and language-derived, so this is currently a weak object-centric probe.",
            },
        ),
        (
            "caption_grounding",
            {
                "plain_goal": "Given a text-like query from annotation, find the matching time window.",
                "case_study": "A query like Pour milk into coffee should rank the windows from the actual pouring moment higher than unrelated windows.",
                "input": "Caption/object/interaction query features and a set of candidate sensor-window features.",
                "middle_modules": [
                    "Query builder converts annotation words into a compact query representation.",
                    "Candidate builder gathers held-out sensor windows.",
                    "Projection head maps sensor windows into the query space.",
                    "Ranker scores candidates by cosine similarity.",
                    "Evaluator reports MRR and top-k retrieval accuracy.",
                ],
                "output": "A ranked list of windows, with the correct matching window ideally near rank 1.",
                "junior_tip": "This is search: type a description, retrieve the matching moment.",
                "failure_mode": "Bag-of-objects text features are too simple for rich language grounding.",
            },
        ),
        (
            "cross_modal_retrieval",
            {
                "plain_goal": "Use one group of modalities to retrieve the matching window from another group.",
                "case_study": "Use motion, IMU, and camera-pose signals from a pouring moment to retrieve the matching depth/video representation for that same moment.",
                "input": "Query side: motion, IMU, and camera/pose features. Candidate side: depth and video features.",
                "middle_modules": [
                    "Feature splitter separates query modalities from target modalities.",
                    "Projection head maps the query vector into target-modality space.",
                    "Candidate index stores target vectors from held-out windows.",
                    "Ranker retrieves nearest candidates by cosine similarity.",
                    "Evaluator reports MRR, top-1, top-5, and top-10 accuracy.",
                ],
                "output": "A ranked list of candidate depth/video windows.",
                "junior_tip": "This checks whether different sensors agree about the same moment in time.",
                "failure_mode": "Good retrieval means useful alignment signal, but it is not yet 3D reconstruction or rendering.",
            },
        ),
        (
            "modality_reconstruction",
            {
                "plain_goal": "Predict one modality feature block from other modality blocks.",
                "case_study": "Given motion, IMU, and camera-pose signals while the hand moves, predict the matching depth/video feature vector.",
                "input": "Motion, IMU, and camera/pose features as input; depth/video features as the regression target.",
                "middle_modules": [
                    "Feature splitter defines source and target modality blocks.",
                    "Scaler normalizes source and target vectors using train statistics.",
                    "Regression head predicts the target feature vector.",
                    "Inverse scaler returns predictions to target scale.",
                    "Evaluator reports MSE, MAE, and R2.",
                ],
                "output": "A reconstructed depth/video feature vector.",
                "junior_tip": "This is feature-level imagination: can the model infer what another sensor would see?",
                "failure_mode": "This reconstructs compressed features, not raw pixels, depth maps, meshes, NeRFs, or Gaussian splats.",
            },
        ),
        (
            "temporal_order",
            {
                "plain_goal": "Tell whether two nearby windows are in the correct time order.",
                "case_study": "If window A shows reaching and window B shows pouring, the model should distinguish A then B from B then A.",
                "input": "A pair of adjacent window vectors, plus their difference vector.",
                "middle_modules": [
                    "Pair builder creates correct-order and reversed-order examples.",
                    "Feature combiner concatenates first window, second window, and their difference.",
                    "Binary classifier predicts correct vs reversed.",
                    "Evaluator reports F1, precision, and recall.",
                    "Diagnostic reader interprets whether features encode local time direction.",
                ],
                "output": "A binary label: correct order or reversed order.",
                "junior_tip": "This asks whether the representation knows which moment came first.",
                "failure_mode": "It only tests local ordering, not long-term planning or causality.",
            },
        ),
        (
            "misalignment_detection",
            {
                "plain_goal": "Detect when modalities that should match are shifted out of sync.",
                "case_study": "Motion from a pouring moment is paired with video/depth from several windows later. The task asks the model to detect that mismatch.",
                "input": "A motion-side feature group and a visual/depth-side feature group, either aligned or artificially shifted.",
                "middle_modules": [
                    "Alignment builder creates positive pairs from the same time window.",
                    "Shift builder creates negative pairs by offsetting one modality group.",
                    "Feature combiner joins both sides into one example.",
                    "Binary classifier predicts aligned vs misaligned.",
                    "Evaluator reports F1 and accuracy.",
                ],
                "output": "A binary label: aligned or shifted.",
                "junior_tip": "This is a synchronization alarm for multimodal data.",
                "failure_mode": "Synthetic shifts are useful diagnostics but do not solve calibration, reconstruction, or mapping by themselves.",
            },
        ),
    ]
)


def load_summary() -> dict[str, Any]:
    return json.loads(SUMMARY_REPORT.read_text(encoding="utf-8"))


def metric(summary: dict[str, Any], task: str, family: str) -> float | None:
    task_metrics = summary.get(family, {}).get(task, {})
    key = METRIC_SPECS[task][0]
    value = task_metrics.get(key)
    return float(value) if value is not None else None


def build_payload(summary: dict[str, Any]) -> dict[str, Any]:
    tasks = OrderedDict()
    for task, spec in TASK_WALKTHROUGHS.items():
        metric_key, metric_name, direction = METRIC_SPECS[task]
        minimal = metric(summary, task, "tasks")
        neural = metric(summary, task, "neural_tasks")
        tasks[task] = {
            **spec,
            **TASK_PRESENTATION[task],
            "task": task,
            "artifact_id": task,
            "metric": {
                "key": metric_key,
                "name": metric_name,
                "direction": direction,
                "minimal": minimal,
                "neural_mlp": neural,
            },
            "module_summary": "input window -> feature/target builder -> baseline head -> evaluator -> artifact files",
        }
    return {
        "source": "results/episode_task_suite/summary_report.json",
        "scope": {
            "episode_count": 1,
            "num_frames": summary.get("num_frames"),
            "num_windows": summary.get("num_windows"),
            "feature_dim": summary.get("feature_dim"),
            "window_frames": summary.get("window_frames"),
            "stride_frames": summary.get("stride_frames"),
            "warning": "These walkthroughs explain task contracts on one public sample episode; they are not cross-episode performance claims.",
        },
        "shared_pipeline": [
            "Read annotation.hdf5 and synchronized video-derived features.",
            "Slice the episode into 20-frame windows with stride 5.",
            "Build an 8,378-d current feature vector from available modality blocks.",
            "Construct a task-specific target from labels, future frames, paired windows, or modality splits.",
            "Train a minimal head and, when enabled, a neural MLP head.",
            "Write metrics, predictions, and model artifacts for downstream exploration.",
        ],
        "tasks": tasks,
    }


def write_markdown(payload: dict[str, Any]) -> None:
    lines = [
        "# Junior-Friendly 12-Task Walkthroughs",
        "",
        "This file explains every task in the Xperience-10M episode suite as an input -> process -> output pipeline.",
        "It is generated by `scripts/task_walkthroughs.py` from committed metrics plus hand-curated task explanations.",
        "",
        "## Shared Pipeline",
        "",
    ]
    for step in payload["shared_pipeline"]:
        lines.append(f"- {step}")
    lines.extend(["", "## Task Walkthroughs", ""])

    for task, spec in payload["tasks"].items():
        metric = spec["metric"]
        minimal = fmt_metric(metric["minimal"])
        neural = fmt_metric(metric["neural_mlp"])
        lines.extend(
            [
                f"### {spec['display_name']} (`{task}`)",
                "",
                f"**Research name:** {spec['research_name']}",
                "",
                f"**Family:** {spec['task_family']}; {spec['architecture_family']}; {spec['primary_direction']}.",
                "",
                f"**Goal:** {spec['plain_goal']}",
                "",
                f"**Case study:** {spec['case_study']}",
                "",
                f"**Input:** {spec['input']}",
                "",
                "**Middle process modules:**",
            ]
        )
        for module in spec["middle_modules"]:
            lines.append(f"- {module}")
        lines.extend(
            [
                "",
                f"**Output:** {spec['output']}",
                "",
                f"**Metric:** {metric['name']} ({metric['direction']} is better). Minimal `{minimal}`, neural MLP `{neural}`.",
                "",
                f"**Junior mental model:** {spec['junior_tip']}",
                "",
                f"**Current limitation:** {spec['failure_mode']}",
                "",
            ]
        )

    (OUT_DIR / "TASK_WALKTHROUGHS.md").write_text(
        "\n".join(lines).rstrip() + "\n", encoding="utf-8"
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DATA.mkdir(parents=True, exist_ok=True)
    payload = build_payload(load_summary())
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    (OUT_DIR / "task_walkthroughs.json").write_text(text + "\n", encoding="utf-8")
    (DOCS_DATA / "task_walkthroughs.json").write_text(text + "\n", encoding="utf-8")
    write_markdown(payload)
    print(f"Wrote {OUT_DIR / 'task_walkthroughs.json'}")
    print(f"Wrote {OUT_DIR / 'TASK_WALKTHROUGHS.md'}")
    print(f"Wrote {DOCS_DATA / 'task_walkthroughs.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

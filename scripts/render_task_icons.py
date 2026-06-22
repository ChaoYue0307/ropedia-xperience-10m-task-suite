#!/usr/bin/env python3
"""Render assigned icons for the 20-task Xperience-10M suite."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ICON_DIR = ROOT / "docs" / "assets" / "task-icons"
MANIFEST_PATH = ROOT / "docs" / "data" / "task_icon_manifest.json"
SHEET_NAME = "task-icon-atlas.png"


TASKS = [
    ("timeline_action", "01", "Action Recognition", "person action pose"),
    ("timeline_subtask", "02", "Procedure Step Recognition", "ordered step checklist"),
    ("transition_detection", "03", "Action Boundary Detection", "timeline boundary"),
    ("next_action", "04", "Next-Action Prediction", "next action arrow"),
    ("hand_trajectory_forecast", "05", "Hand Trajectory Forecasting", "future hand path"),
    ("contact_prediction", "06", "Contact State Prediction", "touch contact point"),
    ("object_relevance", "07", "Object Relevance Prediction", "highlighted relevant object"),
    ("caption_grounding", "08", "Language Grounding", "caption grounded to frame"),
    ("cross_modal_retrieval", "09", "Cross-Modal Retrieval", "search across modalities"),
    ("modality_reconstruction", "10", "Cross-Modal Reconstruction", "missing modality rebuilt"),
    ("temporal_order", "11", "Temporal Order Verification", "ordered event nodes"),
    ("misalignment_detection", "12", "Multimodal Synchronization Detection", "sync mismatch warning"),
    ("long_horizon_next_action", "13", "Long-Horizon Next-Action Forecasting", "long future action path"),
    ("next_subtask_forecast", "14", "Long-Horizon Next-Subtask Forecasting", "future step branch"),
    ("interaction_text_prediction", "15", "Interaction Text Prediction", "hand action to text"),
    ("action_object_relation", "16", "Action-Object Relation Prediction", "hand object relation graph"),
    ("object_set_forecast", "17", "Future Object-Set Forecasting", "future object cluster"),
    ("imu_to_hand_pose", "18", "IMU-to-Hand Pose Reconstruction", "imu waveform to hand pose"),
    ("camera_view_sync_retrieval", "19", "Camera-View Synchronization Retrieval", "multi-camera sync"),
    ("time_to_transition", "20", "Time-to-Next-Transition Regression", "clock to transition"),
]


BODIES = {
    "timeline_action": """
      <circle cx="50" cy="30" r="8"/>
      <path d="M50 39 L50 58 M34 48 L50 43 L66 48 M50 58 L38 76 M50 58 L66 75"/>
      <path class="accent" d="M67 27 L76 18 L73 33 L82 31 L67 52 L71 37"/>
    """,
    "timeline_subtask": """
      <rect x="24" y="20" width="52" height="60" rx="9"/>
      <path d="M36 37 L41 42 L50 32 M36 55 L41 60 L50 50 M36 70 L41 75 L50 65"/>
      <path class="accent" d="M56 38 H68 M56 56 H68 M56 72 H66"/>
    """,
    "transition_detection": """
      <path d="M18 58 H82"/>
      <path class="accent" d="M50 22 V82"/>
      <circle cx="25" cy="58" r="7"/>
      <circle cx="41" cy="58" r="7"/>
      <circle cx="63" cy="58" r="7"/>
      <circle cx="78" cy="58" r="7"/>
    """,
    "next_action": """
      <circle cx="35" cy="32" r="7"/>
      <path d="M35 40 L35 58 M24 47 L35 43 L46 47 M35 58 L27 73 M35 58 L47 73"/>
      <path class="accent" d="M54 50 H76 M66 39 L77 50 L66 61"/>
    """,
    "hand_trajectory_forecast": """
      <path d="M24 66 C30 52 39 47 49 48"/>
      <path d="M38 69 L33 51 M48 68 L46 49 M58 70 L59 52 M68 72 L72 58"/>
      <path class="accent dashed" d="M25 36 C42 18 61 20 77 36"/>
      <circle class="accent-fill" cx="77" cy="36" r="4"/>
    """,
    "contact_prediction": """
      <path d="M42 24 V61 M53 28 V62 M64 35 V63 M31 41 V64"/>
      <path d="M31 64 C37 78 64 79 72 63"/>
      <rect class="accent" x="24" y="72" width="52" height="6" rx="3"/>
      <circle class="accent-fill" cx="54" cy="72" r="5"/>
    """,
    "object_relevance": """
      <rect x="22" y="26" width="18" height="18" rx="4"/>
      <rect class="accent" x="48" y="24" width="26" height="26" rx="6"/>
      <rect x="24" y="58" width="20" height="20" rx="5"/>
      <rect x="58" y="60" width="16" height="16" rx="4"/>
      <path class="accent dashed" d="M61 37 C52 42 48 48 45 57"/>
    """,
    "caption_grounding": """
      <rect x="21" y="22" width="58" height="38" rx="6"/>
      <path class="accent" d="M32 72 H62 L73 82 V72 H78"/>
      <path d="M31 33 H66 M31 44 H56"/>
      <path class="accent dashed" d="M52 60 L67 72"/>
    """,
    "cross_modal_retrieval": """
      <rect x="18" y="24" width="25" height="20" rx="5"/>
      <path d="M24 39 L30 33 L35 38 L39 34"/>
      <path class="accent" d="M57 25 C68 25 76 33 76 44 C76 55 68 63 57 63 C46 63 38 55 38 44 C38 33 46 25 57 25 Z"/>
      <path class="accent" d="M66 56 L80 70"/>
      <path d="M22 64 H44 M22 72 H38"/>
    """,
    "modality_reconstruction": """
      <rect x="20" y="25" width="23" height="23" rx="5"/>
      <rect x="57" y="25" width="23" height="23" rx="5"/>
      <rect class="accent dashed" x="38" y="58" width="24" height="24" rx="5"/>
      <path class="accent" d="M43 36 H57 M50 43 V58 M31 48 L42 60 M69 48 L58 60"/>
    """,
    "temporal_order": """
      <path d="M18 52 H82"/>
      <circle cx="27" cy="52" r="8"/>
      <circle class="accent" cx="50" cy="52" r="8"/>
      <circle cx="73" cy="52" r="8"/>
      <path class="accent" d="M34 36 L45 30 L56 36 M56 68 L67 74 L78 68"/>
    """,
    "misalignment_detection": """
      <path d="M18 38 C27 22 36 54 45 38 C54 22 63 54 72 38"/>
      <path class="accent" d="M18 62 C27 46 36 78 45 62 C54 46 63 78 72 62"/>
      <path class="accent" d="M80 28 L90 48 H70 Z"/>
      <circle class="accent-fill" cx="80" cy="43" r="1.8"/>
    """,
    "long_horizon_next_action": """
      <path class="accent dashed" d="M20 73 C31 23 57 23 75 45"/>
      <path class="accent" d="M66 44 L77 46 L72 36"/>
      <circle cx="25" cy="72" r="5"/>
      <circle cx="77" cy="46" r="5"/>
      <path d="M51 62 L51 76 M42 69 H60 M51 76 L43 86 M51 76 L61 86"/>
    """,
    "next_subtask_forecast": """
      <rect x="21" y="23" width="24" height="18" rx="5"/>
      <rect x="56" y="20" width="24" height="18" rx="5"/>
      <rect x="56" y="52" width="24" height="18" rx="5"/>
      <path class="accent" d="M45 32 H55 M45 32 C53 35 54 55 56 61"/>
      <path d="M26 32 H39 M61 29 H74 M61 61 H74"/>
    """,
    "interaction_text_prediction": """
      <path d="M22 55 C28 42 39 39 50 42 M34 60 L31 45 M45 62 L45 42 M56 63 L59 48"/>
      <rect class="accent" x="58" y="23" width="25" height="22" rx="5"/>
      <path class="accent" d="M64 32 H77 M64 39 H72"/>
      <path class="accent dashed" d="M52 47 L62 45"/>
    """,
    "action_object_relation": """
      <circle cx="30" cy="64" r="9"/>
      <rect x="63" y="24" width="19" height="19" rx="5"/>
      <circle class="accent" cx="66" cy="70" r="8"/>
      <path d="M38 59 C48 47 58 37 63 34"/>
      <path class="accent" d="M38 67 C49 71 58 72 66 70"/>
    """,
    "object_set_forecast": """
      <rect x="21" y="29" width="17" height="17" rx="4"/>
      <circle cx="56" cy="35" r="8"/>
      <path d="M72 27 L82 44 H62 Z"/>
      <path class="accent dashed" d="M24 68 C39 57 62 57 78 68"/>
      <circle class="accent-fill" cx="31" cy="67" r="4"/>
      <circle class="accent-fill" cx="51" cy="61" r="4"/>
      <circle class="accent-fill" cx="71" cy="67" r="4"/>
    """,
    "imu_to_hand_pose": """
      <path class="accent" d="M15 56 C22 35 30 76 37 55 C44 35 52 76 59 55"/>
      <path d="M70 23 V53 M60 35 L70 30 L81 36 M70 53 L62 72 M70 53 L82 72"/>
      <circle cx="70" cy="18" r="6"/>
      <path class="accent dashed" d="M58 55 L68 50"/>
    """,
    "camera_view_sync_retrieval": """
      <rect x="19" y="30" width="26" height="20" rx="5"/>
      <path d="M45 36 L55 31 V49 L45 44"/>
      <rect x="57" y="55" width="26" height="20" rx="5"/>
      <path d="M83 61 L91 57 V73 L83 69"/>
      <path class="accent" d="M28 66 C43 79 61 79 76 66"/>
      <path class="accent" d="M69 66 L78 66 L74 74"/>
    """,
    "time_to_transition": """
      <circle cx="39" cy="46" r="20"/>
      <path d="M39 34 V47 L51 54"/>
      <path class="accent" d="M65 24 V77"/>
      <path d="M65 52 H84"/>
      <path class="accent" d="M77 42 L87 52 L77 62"/>
    """,
}


def render_svg(task_id: str, number: str, title: str) -> str:
    body = "\n".join(line.rstrip() for line in BODIES[task_id].strip("\n").splitlines())
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" role="img" aria-labelledby="title desc">
  <title id="title">{number} {title}</title>
  <desc id="desc">Ropedia Xperience-10M task icon for {title}.</desc>
  <defs>
    <filter id="glow" x="-40%" y="-40%" width="180%" height="180%">
      <feGaussianBlur stdDeviation="2.6" result="blur"/>
      <feColorMatrix in="blur" type="matrix" values="0 0 0 0 0.61 0 0 0 0 1 0 0 0 0 0.43 0 0 0 .85 0"/>
      <feMerge><feMergeNode/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <linearGradient id="ring" x1="18" y1="12" x2="84" y2="90" gradientUnits="userSpaceOnUse">
      <stop stop-color="#ccffa0"/>
      <stop offset=".55" stop-color="#7ae5c3"/>
      <stop offset="1" stop-color="#9bdfff"/>
    </linearGradient>
  </defs>
  <rect x="4" y="4" width="92" height="92" rx="18" fill="#020602"/>
  <rect x="7" y="7" width="86" height="86" rx="16" fill="#ccffa0" fill-opacity=".035" stroke="url(#ring)" stroke-width="2"/>
  <g fill="none" stroke="#eaf5e5" stroke-width="5" stroke-linecap="round" stroke-linejoin="round" filter="url(#glow)">
{body}
  </g>
  <style>
    .accent {{ stroke: #a7ff4f; }}
    .accent-fill {{ fill: #a7ff4f; stroke: #a7ff4f; }}
    .dashed {{ stroke-dasharray: 6 6; }}
  </style>
</svg>
"""


def build_manifest() -> dict:
    return {
        "schema": "ropedia.task_icons.v1",
        "description": "Assigned icon assets for the unified 20-task Xperience-10M suite.",
        "overall_sheet": f"assets/task-icons/{SHEET_NAME}",
        "tasks": [
            {
                "task_id": task_id,
                "task_number": int(number),
                "display_name": title,
                "motif": motif,
                "icon": f"assets/task-icons/{number}_{task_id}.svg",
            }
            for task_id, number, title, motif in TASKS
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sheet-source",
        type=Path,
        default=None,
        help="Optional generated 20-icon atlas PNG to copy into docs/assets/task-icons.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ICON_DIR.mkdir(parents=True, exist_ok=True)
    for task_id, number, title, _motif in TASKS:
        target = ICON_DIR / f"{number}_{task_id}.svg"
        target.write_text(render_svg(task_id, number, title), encoding="utf-8")

    if args.sheet_source:
        if not args.sheet_source.exists():
            raise FileNotFoundError(args.sheet_source)
        shutil.copyfile(args.sheet_source, ICON_DIR / SHEET_NAME)

    MANIFEST_PATH.write_text(json.dumps(build_manifest(), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(TASKS)} task icons and {MANIFEST_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

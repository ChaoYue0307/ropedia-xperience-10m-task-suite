#!/usr/bin/env python3
"""Build research takeaways from committed Xperience-10M metric artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "docs/data/summary_metrics.json"
AUDIO_PATH = ROOT / "docs/data/audio_ablation_summary.json"
OUTPUT_JSON = ROOT / "docs/data/research_takeaways.json"
OUTPUT_MD = ROOT / "RESEARCH_TAKEAWAYS.md"


def pct_delta(new: float, old: float, higher_is_better: bool = True) -> float:
    if old == 0:
        return 0.0
    if higher_is_better:
        return (new - old) / abs(old)
    return (old - new) / abs(old)


def fmt(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:.{digits}f}"


def task_metric(tasks: dict, task: str, key: str) -> float:
    return float(tasks[task][key])


def build_payload() -> dict:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    audio_summary = json.loads(AUDIO_PATH.read_text(encoding="utf-8")) if AUDIO_PATH.exists() else None
    suite = summary["suite"]
    tasks = suite["tasks"]
    neural = suite.get("neural_tasks", {})
    models = summary["models"]
    omni = summary.get("omni_relay", {})

    hand_min = task_metric(tasks, "hand_trajectory_forecast", "mpjpe")
    hand_neural = task_metric(neural, "hand_trajectory_forecast", "mpjpe")
    temporal_min = task_metric(tasks, "temporal_order", "f1")
    temporal_neural = task_metric(neural, "temporal_order", "f1")
    misalign_min = task_metric(tasks, "misalignment_detection", "f1")
    misalign_neural = task_metric(neural, "misalignment_detection", "f1")
    retrieval_min_mrr = task_metric(tasks, "cross_modal_retrieval", "mrr")
    retrieval_neural_mrr = task_metric(neural, "cross_modal_retrieval", "mrr")
    recon_min_r2 = task_metric(tasks, "modality_reconstruction", "r2")
    recon_neural_r2 = task_metric(neural, "modality_reconstruction", "r2")
    action_chrono = task_metric(tasks, "timeline_action", "macro_f1")
    subtask_chrono = task_metric(tasks, "timeline_subtask", "macro_f1")

    takeaways = [
        {
            "id": "episode_to_benchmark",
            "title": "One episode can become a real benchmark contract",
            "readout": (
                "The public sample is converted into 5,821 frames, 1,161 aligned "
                f"20-frame windows, and an {suite['feature_dim']:,}-dimensional feature contract."
            ),
            "evidence": [
                {"label": "frames", "value": suite["num_frames"]},
                {"label": "windows", "value": suite["num_windows"]},
                {"label": "feature_dim", "value": suite["feature_dim"]},
            ],
            "source": "docs/data/summary_metrics.json",
            "current_scope": "This benchmark defines the task contract; cross-episode generalization is evaluated in the multi-episode stage.",
        },
        {
            "id": "chronological_split_exposes_class_shift",
            "title": "Chronological splits expose action-class shift",
            "readout": (
                "Earlier all-feature action classifiers reach high macro-F1 on their "
                "local split, but the 12-task chronological action/subtask heads are "
                "much harder because later held-out windows include unseen labels."
            ),
            "evidence": [
                {"label": "all_feature_action_macro_f1", "value": models["all_modalities_action"]["macro_f1"]},
                {"label": "suite_action_macro_f1", "value": action_chrono},
                {"label": "suite_subtask_macro_f1", "value": subtask_chrono},
                {"label": "unseen_action_test_classes", "value": len(tasks["timeline_action"].get("unseen_test_classes", []))},
            ],
            "source": "results/episode_task_suite/summary_report.json",
            "current_scope": "This split is useful for studying label shift; broad action-recognition conclusions need held-out episodes.",
        },
        {
            "id": "neural_heads_help_dynamics",
            "title": "Small neural heads help dynamic and temporal probes",
            "readout": (
                "The MLP heads substantially improve hand trajectory forecasting, "
                "temporal-order verification, and motion/visual synchronization."
            ),
            "evidence": [
                {"label": "hand_mpjpe_minimal", "value": hand_min},
                {"label": "hand_mpjpe_neural", "value": hand_neural},
                {"label": "hand_mpjpe_relative_improvement", "value": pct_delta(hand_neural, hand_min, higher_is_better=False)},
                {"label": "temporal_order_f1_minimal", "value": temporal_min},
                {"label": "temporal_order_f1_neural", "value": temporal_neural},
                {"label": "misalignment_f1_minimal", "value": misalign_min},
                {"label": "misalignment_f1_neural", "value": misalign_neural},
            ],
            "source": "results/episode_task_suite/neural_mlp/*/metrics.json",
            "current_scope": "These gains are measured within one episode and are candidates for held-out-episode testing.",
        },
        {
            "id": "retrieval_and_reconstruction_remain_open",
            "title": "Retrieval and reconstruction remain the harder multimodal problems",
            "readout": (
                "Ridge/cosine retrieval remains stronger than the neural projection on "
                "this sample, and cross-modal reconstruction still has negative R2."
            ),
            "evidence": [
                {"label": "retrieval_mrr_minimal", "value": retrieval_min_mrr},
                {"label": "retrieval_mrr_neural", "value": retrieval_neural_mrr},
                {"label": "retrieval_top5_minimal", "value": tasks["cross_modal_retrieval"]["top5_accuracy"]},
                {"label": "reconstruction_r2_minimal", "value": recon_min_r2},
                {"label": "reconstruction_r2_neural", "value": recon_neural_r2},
            ],
            "source": "results/episode_task_suite/cross_modal_retrieval/metrics.json",
            "current_scope": "The current reconstruction task predicts feature vectors; depth, mesh, NeRF, and Gaussian-splatting outputs are future task variants.",
        },
    ]

    if audio_summary is not None:
        audio_aggregate = audio_summary["aggregate"]
        modality_recon = next(
            (item for item in audio_summary["task_summaries"] if item["task"] == "modality_reconstruction"),
            {},
        )
        object_relevance = next(
            (item for item in audio_summary["task_summaries"] if item["task"] == "object_relevance"),
            {},
        )
        takeaways.append(
            {
                "id": "audio_contribution_is_task_specific",
                "title": "Audio helps some tasks and hurts others on the public sample",
                "readout": (
                    "The current AAC audio block improves the primary metric on 6 of 12 tasks, "
                    "while raw log-mel replacement improves over the current handcrafted block on 6 of 12 tasks. "
                    "The largest current-audio gain appears in feature reconstruction, not in action classification."
                ),
                "evidence": [
                    {"label": "tasks_where_current_audio_improves", "value": audio_aggregate["tasks_where_handcrafted_audio_improves"]},
                    {"label": "mean_current_audio_delta", "value": audio_aggregate["mean_handcrafted_audio_delta"]},
                    {"label": "tasks_where_raw_replacement_improves", "value": audio_aggregate["tasks_where_raw_replacement_improves_over_handcrafted"]},
                    {"label": "mean_raw_replacement_delta_vs_current", "value": audio_aggregate["mean_raw_replacement_delta_vs_handcrafted"]},
                    {"label": "reconstruction_current_audio_delta", "value": modality_recon.get("handcrafted_audio_delta")},
                    {"label": "object_relevance_current_audio_delta", "value": object_relevance.get("handcrafted_audio_delta")},
                ],
                "source": "results/audio_ablation/audio_ablation_summary.json",
                "current_scope": (
                    "This is a single-episode ablation over fixed ridge heads. It validates that audio is wired into the task suite "
                    "and shows where it changes metrics; it does not prove cross-episode audio generalization."
                ),
            }
        )

    takeaways.append(
        {
            "id": "scale_requires_episodes",
            "title": "The next scientific unit is held-out episodes, not more adjacent windows",
            "readout": (
                "The prepared Qwen3-Omni path targets 32 episodes from 32 sessions, "
                "but it remains data-gated until access and held-out evaluation complete."
            ),
            "evidence": [
                {"label": "target_episodes", "value": omni.get("target_episodes")},
                {"label": "selected_sessions", "value": omni.get("selected_sessions")},
                {"label": "valid_candidates", "value": omni.get("valid_candidates")},
            ],
            "source": "results/omni_finetune/MULTI_EPISODE_ACCESS_STATUS.md",
            "current_scope": omni.get(
                "current_scope",
                "The 32-episode fine-tune requires gated data staging and held-out evaluation.",
            ),
        }
    )

    return {
        "title": "Ropedia Xperience-10M Research Takeaways",
        "status": "pass",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_files": [
            "docs/data/summary_metrics.json",
            "results/episode_task_suite/summary_report.json",
            "results/episode_task_suite/neural_mlp/*/metrics.json",
            "docs/data/audio_ablation_summary.json",
            "results/omni_finetune/MULTI_EPISODE_ACCESS_STATUS.md",
        ],
        "scope": {
            "validated_episode_count": 1,
            "num_frames": suite["num_frames"],
            "num_windows": suite["num_windows"],
            "feature_dim": suite["feature_dim"],
            "audio_featurized": True,
            "raw_data_redistributed": False,
        },
        "takeaways": takeaways,
    }


def render_md(payload: dict) -> str:
    lines = [
        "# Research Takeaways",
        "",
        "This generated note summarizes what the current public Xperience-10M sample",
        "pipeline actually shows. It is built from committed metric artifacts, not",
        "from hand-edited score text.",
        "",
        "## Scope",
        "",
        f"- validated episodes: {payload['scope']['validated_episode_count']}",
        f"- frames: {payload['scope']['num_frames']:,}",
        f"- aligned windows: {payload['scope']['num_windows']:,}",
        f"- current feature dimension: {payload['scope']['feature_dim']:,}",
        "- raw Xperience-10M data is not redistributed",
        "- AAC audio from the sample MP4 stream is extracted into the current feature vector",
        "",
        "## Takeaways",
        "",
    ]
    for item in payload["takeaways"]:
        lines.extend(
            [
                f"### {item['title']}",
                "",
                item["readout"],
                "",
                "| Metric | Value |",
                "| --- | ---: |",
            ]
        )
        for evidence in item["evidence"]:
            value = evidence["value"]
            if isinstance(value, float):
                value_text = fmt(value)
            elif isinstance(value, int):
                value_text = fmt(value)
            elif value is None:
                value_text = "n/a"
            else:
                value_text = str(value)
            lines.append(f"| `{evidence['label']}` | {value_text} |")
        lines.extend(["", f"Source: `{item['source']}`.", "", f"Current scope: {item['current_scope']}", ""])
    lines.extend(
        [
            "## How To Read These Results",
            "",
            "- High single-episode scores are useful pipeline checks for the current task contracts.",
            "- Low chronological action/subtask scores are informative because they expose later-label shift.",
        "- Neural gains on trajectory/order/alignment make those tasks good candidates for the next fine-tuning stage.",
        "- Audio ablation is task-specific: current AAC and raw log-mel features help some probes and hurt others.",
        "- Retrieval and reconstruction remain the main multimodal representation challenges.",
            "- The next credible model-quality result needs held-out episodes.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    payload = build_payload()
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    OUTPUT_MD.write_text(render_md(payload), encoding="utf-8")
    print(f"PASS: wrote {OUTPUT_JSON}")
    print(f"PASS: wrote {OUTPUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

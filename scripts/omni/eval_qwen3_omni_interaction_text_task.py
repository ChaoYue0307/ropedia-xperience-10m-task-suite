#!/usr/bin/env python3
"""Evaluate Qwen3-Omni v6 on raw interaction-text prediction.

This task-15 runner waits for the full raw ``annotation.hdf5`` caption
extraction, aligns each held-out window to the nearest raw interaction string,
then asks Qwen3-Omni to rank candidate interaction descriptions for the video
window. It writes a real task-specific artifact; no score is emitted unless the
caption manifest is complete and the held-out predictions are generated.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from eval_qwen3_omni_retrieval_task_probes import (
    SYSTEM_PROMPT,
    append_jsonl,
    extract_ranking,
    generate_messages,
    media_video_path,
    read_jsonl_if_exists,
    row_end,
    row_start,
    stable_score,
    write_csv,
    write_json,
    write_jsonl,
)
from eval_qwen3_omni_lora import load_model_processor
from qwen3_omni_dataset_utils import class_metrics, load_jsonl
from run_128_raw_interaction_text_task import (
    assign_interaction_labels,
    build_episode_interactions,
    load_caption_rows,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = (
    ROOT
    / "results/omni_finetune/xperience10m_qwen3_omni_128ep_multiscale_cap96_v5_full8gpu_lora_dataset"
    / "dataset_a100_eval.jsonl"
)
DEFAULT_CAPTION_DIR = ROOT / "results/omni_finetune/xperience10m_128_raw_caption_interactions_task15_20260619_full"
TASK_ID = "interaction_text_prediction"
TASK_NUMBER = 15
TASK_LABEL = "Interaction Text Prediction"
METRIC_KEY = "macro_f1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-jsonl", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--caption-jsonl", type=Path, default=DEFAULT_CAPTION_DIR / "caption_interactions.jsonl")
    parser.add_argument("--caption-manifest", type=Path, default=DEFAULT_CAPTION_DIR / "caption_interactions_manifest.json")
    parser.add_argument("--run-id", default="xperience10m_qwen3_omni_v6_interaction_text_task15_a100_20260619T000000Z")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--candidate-count", type=int, default=4)
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--use-audio-in-video", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--allow-partial-captions", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress-jsonl", type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def check_caption_manifest(args: argparse.Namespace) -> dict[str, Any]:
    manifest = read_json(args.caption_manifest)
    if manifest.get("status") != "pass" and not args.allow_partial_captions:
        raise SystemExit(
            f"Caption extraction is not complete: status={manifest.get('status')} "
            f"processed={manifest.get('processed_file_count')}/{manifest.get('requested_file_count')}. "
            "Task-15 Qwen scoring requires a pass manifest."
        )
    return manifest


def prediction_id(sample: dict[str, Any]) -> str:
    return f"{TASK_ID}::{sample.get('id')}"


def select_eval_indices(samples: list[dict[str, Any]], labels: list[str], args: argparse.Namespace) -> list[int]:
    if args.sample_stride < 1:
        raise ValueError("--sample-stride must be >= 1")
    if args.sample_offset < 0 or args.sample_offset >= args.sample_stride:
        raise ValueError("--sample-offset must satisfy 0 <= offset < stride")
    indices = [
        idx
        for idx, sample in enumerate(samples)
        if sample.get("split") == args.eval_split and labels[idx] and media_video_path(sample)
    ]
    if args.sample_stride > 1:
        indices = [idx for local_idx, idx in enumerate(indices) if local_idx % args.sample_stride == args.sample_offset]
    if args.sample_limit > 0:
        indices = indices[: args.sample_limit]
    return indices


def build_candidate_labels(
    samples: list[dict[str, Any]],
    labels: list[str],
    eval_pool: list[int],
    sample_idx: int,
    candidate_count: int,
) -> tuple[list[dict[str, Any]], str]:
    if candidate_count < 2 or candidate_count > 8:
        raise ValueError("--candidate-count must be between 2 and 8")
    true_label = labels[sample_idx]
    candidates_by_label: dict[str, int] = {true_label: sample_idx}
    negatives = [
        idx
        for idx in eval_pool
        if idx != sample_idx and labels[idx] and labels[idx] != true_label
    ]
    negatives.sort(key=lambda idx: stable_score(TASK_ID, samples[sample_idx].get("id"), samples[idx].get("id"), labels[idx]))
    for idx in negatives:
        candidates_by_label.setdefault(labels[idx], idx)
        if len(candidates_by_label) >= candidate_count:
            break
    if len(candidates_by_label) < candidate_count:
        raise RuntimeError(f"not enough distinct interaction-text candidates for sample {samples[sample_idx].get('id')}")

    ordered = list(candidates_by_label.items())
    ordered.sort(key=lambda item: stable_score(TASK_ID, "order", samples[sample_idx].get("id"), item[0]))
    letters = [chr(ord("A") + pos) for pos in range(len(ordered))]
    records = []
    true_letter = ""
    for letter, (label, idx) in zip(letters, ordered):
        if label == true_label:
            true_letter = letter
        records.append(
            {
                "letter": letter,
                "interaction_text": label,
                "source_sample_id": samples[idx].get("id"),
                "source_episode_id": samples[idx].get("episode_id"),
                "is_target": label == true_label,
            }
        )
    return records, true_letter


def build_messages(sample: dict[str, Any], candidate_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidate_lines = [
        f"{record['letter']}. {record['interaction_text']}"
        for record in candidate_records
    ]
    prompt = "\n".join(
        [
            f"Task {TASK_NUMBER}: {TASK_LABEL}",
            "Watch the held-out video window and rank the candidate raw interaction descriptions by how well they describe the human-object interaction in the window.",
            "Return JSON only with this schema:",
            '{"ranked_candidates":["<best letter>","<next letter>", "..."]}',
            "Use each candidate letter at most once. Do not explain.",
            "",
            f"Window frames: {row_start(sample)}-{row_end(sample)}",
            "Candidate interaction descriptions:",
            *candidate_lines,
        ]
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "video", "video": media_video_path(sample)},
            ],
        },
    ]


def score_rows(rows: list[dict[str, Any]], args: argparse.Namespace, manifest: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[list[int]]]:
    y_true = [str(row["true_interaction_text"]) for row in rows]
    y_pred = [str(row["predicted_interaction_text"]) for row in rows]
    label_options = sorted(set(y_true))
    metrics, per_class, confusion = class_metrics(y_true, y_pred, label_options)
    reciprocal_ranks = [float(row.get("reciprocal_rank", 0.0)) for row in rows]
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    metrics.update(
        {
            "title": "Qwen3-Omni v6 Interaction Text Prediction",
            "status": "pass",
            "run_id": args.run_id,
            "task_id": TASK_ID,
            "task_number": TASK_NUMBER,
            "task_label": TASK_LABEL,
            "metric_key": METRIC_KEY,
            "primary_metric": METRIC_KEY,
            "primary_score": metrics["macro_f1"],
            "interaction_text_prediction_macro_f1": metrics["macro_f1"],
            "interaction_text_prediction_accuracy": metrics["accuracy"],
            "interaction_text_prediction_mrr": mrr,
            "model_id": args.model_id,
            "adapter_dir": str(args.adapter_dir),
            "dataset_jsonl": str(args.dataset_jsonl),
            "caption_jsonl": str(args.caption_jsonl),
            "caption_manifest": str(args.caption_manifest),
            "caption_manifest_status": manifest.get("status"),
            "requested_annotation_file_count": manifest.get("requested_file_count"),
            "processed_annotation_file_count": manifest.get("processed_file_count"),
            "eval_split": args.eval_split,
            "candidate_count": args.candidate_count,
            "sample_offset": args.sample_offset,
            "sample_stride": args.sample_stride,
            "scope": "held_out_test_qwen3_interaction_text_task15",
            "score_policy": (
                "GPU-backed Qwen3-Omni v6 task-15 probe over raw caption interaction text extracted "
                "from official annotation.hdf5 files. The model ranks shuffled raw interaction text "
                "candidates for each held-out video window; macro-F1 and accuracy are computed from "
                "the top-ranked candidate. No hashed caption proxy is used for this Qwen score."
            ),
        }
    )
    return metrics, per_class, confusion


def write_outputs(rows: list[dict[str, Any]], args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, Any]:
    task_dir = args.output_dir / TASK_ID
    task_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(task_dir / "predictions.jsonl", rows)
    write_csv(
        task_dir / "predictions.csv",
        [
            {
                "id": row["id"],
                "episode_id": row["episode_id"],
                "split": row["split"],
                "start_frame": row["start_frame"],
                "end_frame": row["end_frame"],
                "true_interaction_text": row["true_interaction_text"],
                "predicted_interaction_text": row["predicted_interaction_text"],
                "true_letter": row["true_letter"],
                "predicted_ranking": json.dumps(row["predicted_ranking"], ensure_ascii=False),
                "reciprocal_rank": row["reciprocal_rank"],
                "top1_correct": row["top1_correct"],
                "raw_prediction": row["raw_prediction"],
            }
            for row in rows
        ],
        [
            "id",
            "episode_id",
            "split",
            "start_frame",
            "end_frame",
            "true_interaction_text",
            "predicted_interaction_text",
            "true_letter",
            "predicted_ranking",
            "reciprocal_rank",
            "top1_correct",
            "raw_prediction",
        ],
    )
    metrics, per_class, confusion = score_rows(rows, args, manifest)
    write_json(task_dir / "metrics.json", metrics)
    write_csv(task_dir / "per_class_metrics.csv", per_class, ["class_name", "support", "predicted", "precision", "recall", "f1"])
    confusion_fieldnames = ["class_name", *[str(label) for label in metrics["labels"]]]
    write_csv(
        task_dir / "confusion_matrix.csv",
        [
            {"class_name": label, **{str(col): value for col, value in zip(metrics["labels"], row)}}
            for label, row in zip(metrics["labels"], confusion)
        ],
        confusion_fieldnames,
    )
    report = "\n".join(
        [
            "# Qwen3-Omni v6 Interaction Text Prediction",
            "",
            f"- Status: {metrics['status']}",
            f"- Samples: {metrics['num_samples']}",
            f"- Macro-F1: {metrics['macro_f1']:.6f}",
            f"- Accuracy: {metrics['accuracy']:.6f}",
            f"- MRR: {metrics['interaction_text_prediction_mrr']:.6f}",
            f"- Caption files: {metrics.get('processed_annotation_file_count')}/{metrics.get('requested_annotation_file_count')}",
            "",
        ]
    )
    (task_dir / "RUN_REPORT.md").write_text(report, encoding="utf-8")
    return metrics


def main() -> int:
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = ROOT / "results/omni_finetune" / args.run_id
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.progress_jsonl = args.progress_jsonl or args.output_dir / "progress.jsonl"

    manifest = check_caption_manifest(args)
    samples = load_jsonl(args.dataset_jsonl)
    caption_rows = load_caption_rows(args.caption_jsonl)
    interactions, _episode_summaries = build_episode_interactions(caption_rows, samples)
    labels, assigned_rows = assign_interaction_labels(samples, interactions)
    eval_pool = [
        idx
        for idx, sample in enumerate(samples)
        if sample.get("split") == args.eval_split and labels[idx] and media_video_path(sample)
    ]
    eval_indices = select_eval_indices(samples, labels, args)
    if not eval_indices:
        raise RuntimeError("No held-out samples with video and raw interaction labels were selected.")

    partial_path = args.output_dir / TASK_ID / "predictions.partial.jsonl"
    partial = {
        row.get("prediction_id"): row
        for row in read_jsonl_if_exists(partial_path)
        if row.get("prediction_id")
    }
    append_jsonl(
        args.progress_jsonl,
        {
            "event": "eval_start",
            "timestamp": time.time(),
            "run_id": args.run_id,
            "task_id": TASK_ID,
            "num_eval_samples": len(eval_indices),
            "sample_offset": args.sample_offset,
            "sample_stride": args.sample_stride,
            "candidate_count": args.candidate_count,
        },
    )

    model, processor = load_model_processor(args)
    for local_pos, sample_idx in enumerate(eval_indices, start=1):
        sample = samples[sample_idx]
        pred_id = prediction_id(sample)
        if pred_id in partial:
            continue
        started = time.time()
        candidate_records, true_letter = build_candidate_labels(
            samples,
            labels,
            eval_pool,
            sample_idx,
            args.candidate_count,
        )
        messages = build_messages(sample, candidate_records)
        raw = generate_messages(model, processor, messages, args)
        letters = [record["letter"] for record in candidate_records]
        ranking = extract_ranking(raw, letters)
        rank = ranking.index(true_letter) + 1 if true_letter in ranking else len(ranking) + 1
        by_letter = {record["letter"]: record["interaction_text"] for record in candidate_records}
        predicted_text = by_letter.get(ranking[0], "") if ranking else ""
        row = {
            "prediction_id": pred_id,
            "id": sample.get("id"),
            "task_id": TASK_ID,
            "task_label": TASK_LABEL,
            "split": sample.get("split"),
            "episode_id": sample.get("episode_id"),
            "start_frame": row_start(sample),
            "end_frame": row_end(sample),
            "assigned_interaction": assigned_rows[sample_idx],
            "true_interaction_text": labels[sample_idx],
            "predicted_interaction_text": predicted_text,
            "candidates": candidate_records,
            "true_letter": true_letter,
            "predicted_ranking": ranking,
            "reciprocal_rank": 1.0 / rank,
            "top1_correct": int(predicted_text == labels[sample_idx]),
            "raw_prediction": raw,
        }
        partial[pred_id] = row
        append_jsonl(partial_path, row)
        append_jsonl(
            args.progress_jsonl,
            {
                "event": "sample_done",
                "timestamp": time.time(),
                "sample_index": local_pos,
                "num_eval_samples": len(eval_indices),
                "completed_samples": len(partial),
                "sample_id": sample.get("id"),
                "seconds": round(time.time() - started, 3),
            },
        )

    rows = [partial[prediction_id(samples[idx])] for idx in eval_indices]
    metrics = write_outputs(rows, args, manifest)
    write_json(
        args.output_dir / "summary.json",
        {
            "title": "Qwen3-Omni v6 Interaction Text Task-15 Probe",
            "status": "pass",
            "run_id": args.run_id,
            "task_metrics": {TASK_ID: metrics},
            "output_dir": str(args.output_dir),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Evaluate Qwen3-Omni on target-backed retrieval probes.

This runner covers model-friendly retrieval tasks whose targets can be formed
from the staged 128-episode JSON export without inventing labels. It currently
implements Task 08, language grounding, as text-query-to-video-window retrieval:
the query is derived from the held-out window's action/subtask/object labels,
and Qwen ranks shuffled candidate mosaic video windows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch

from eval_qwen3_omni_lora import load_model_processor, move_inputs
from qwen3_omni_dataset_utils import has_empty_audio_items, is_empty_audio_exception, load_jsonl


TASK_SPECS: OrderedDict[str, dict[str, Any]] = OrderedDict(
    [
        (
            "caption_grounding",
            {
                "task_number": 8,
                "label": "Language Grounding",
                "family": "retrieval",
                "metric_key": "caption_grounding_mrr",
                "prediction_key": "ranked_candidates",
            },
        ),
    ]
)

SYSTEM_PROMPT = (
    "You are an embodied episode-understanding model for Ropedia/Xperience-10M. "
    "Return exactly one compact valid JSON object and no markdown, prose, code "
    "fences, explanations, or repeated text."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    parser.add_argument("--run-id", default="qwen3_retrieval_task_probes")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--tasks", default="caption_grounding")
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
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress-jsonl", type=Path)
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().strip("`'\". ").split())


def answer(sample: dict[str, Any]) -> dict[str, Any]:
    payload = sample.get("answer_json")
    return payload if isinstance(payload, dict) else {}


def row_start(sample: dict[str, Any]) -> int:
    window = sample.get("center_window") if isinstance(sample.get("center_window"), dict) else {}
    return int(window.get("start_frame", 0) or 0)


def row_end(sample: dict[str, Any]) -> int:
    window = sample.get("center_window") if isinstance(sample.get("center_window"), dict) else {}
    return int(window.get("end_frame", row_start(sample)) or row_start(sample))


def media_video_path(sample: dict[str, Any]) -> str | None:
    media = sample.get("media") if isinstance(sample.get("media"), dict) else {}
    return media.get("mosaic_video_path") or sample.get("primary_video_path")


def parse_json_object(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            return {}
        try:
            payload = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return {}
    return payload if isinstance(payload, dict) else {}


def select_tasks(spec: str) -> list[str]:
    if spec.strip().lower() == "all":
        return list(TASK_SPECS)
    tasks = [item.strip() for item in spec.split(",") if item.strip()]
    unknown = [task for task in tasks if task not in TASK_SPECS]
    if unknown:
        raise ValueError(f"unknown tasks: {unknown}")
    return tasks


def select_eval_indices(samples: list[dict[str, Any]], args: argparse.Namespace) -> list[int]:
    indices = [
        idx
        for idx, sample in enumerate(samples)
        if sample.get("split") == args.eval_split and media_video_path(sample) and answer(sample)
    ]
    if args.sample_stride < 1:
        raise ValueError("--sample-stride must be >= 1")
    if args.sample_offset < 0 or args.sample_offset >= args.sample_stride:
        raise ValueError("--sample-offset must satisfy 0 <= offset < stride")
    if args.sample_stride > 1:
        indices = [idx for local_idx, idx in enumerate(indices) if local_idx % args.sample_stride == args.sample_offset]
    if args.sample_limit > 0:
        indices = indices[: args.sample_limit]
    return indices


def prediction_id(task_id: str, sample: dict[str, Any]) -> str:
    return f"{task_id}::{sample.get('id')}"


def stable_score(*parts: Any) -> str:
    return hashlib.sha1("::".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def build_candidate_indices(
    samples: list[dict[str, Any]],
    eval_pool: list[int],
    sample_idx: int,
    task_id: str,
    candidate_count: int,
) -> list[int]:
    if candidate_count < 2 or candidate_count > 8:
        raise ValueError("--candidate-count must be between 2 and 8")
    sample = samples[sample_idx]
    true_action = normalize_text(answer(sample).get("action")).casefold()
    true_episode = sample.get("episode_id")
    negatives = [
        idx
        for idx in eval_pool
        if idx != sample_idx
        and media_video_path(samples[idx])
        and samples[idx].get("episode_id") != true_episode
        and normalize_text(answer(samples[idx]).get("action")).casefold() != true_action
    ]
    if len(negatives) < candidate_count - 1:
        negatives = [idx for idx in eval_pool if idx != sample_idx and media_video_path(samples[idx])]
    negatives.sort(key=lambda idx: stable_score(task_id, sample.get("id"), samples[idx].get("id")))
    selected = [sample_idx] + negatives[: candidate_count - 1]
    selected.sort(key=lambda idx: stable_score(task_id, "order", sample.get("id"), samples[idx].get("id")))
    return selected


def query_text(sample: dict[str, Any]) -> str:
    payload = answer(sample)
    objects = payload.get("objects") if isinstance(payload.get("objects"), list) else []
    object_text = ", ".join(normalize_text(item) for item in objects[:8] if normalize_text(item))
    return "\n".join(
        [
            f"Action: {normalize_text(payload.get('action')) or 'unknown'}",
            f"Procedure step: {normalize_text(payload.get('subtask')) or 'unknown'}",
            f"Relevant objects: {object_text or 'unknown'}",
        ]
    )


def build_messages(
    samples: list[dict[str, Any]],
    sample_idx: int,
    candidate_indices: list[int],
    task_id: str,
    spec: dict[str, Any],
) -> tuple[list[dict[str, Any]], str, list[dict[str, Any]]]:
    letters = [chr(ord("A") + pos) for pos in range(len(candidate_indices))]
    true_letter = letters[candidate_indices.index(sample_idx)]
    candidate_records: list[dict[str, Any]] = []
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": "\n".join(
                [
                    f"Task {spec['task_number']}: {spec['label']}",
                    "Rank the candidate video windows by how well they match the text query.",
                    "Return JSON only with this schema:",
                    '{"ranked_candidates":["<best letter>","<next letter>", "..."]}',
                    "Use each candidate letter at most once.",
                    "",
                    "Text query:",
                    query_text(samples[sample_idx]),
                ]
            ),
        }
    ]
    for letter, idx in zip(letters, candidate_indices):
        sample = samples[idx]
        candidate_records.append(
            {
                "letter": letter,
                "id": sample.get("id"),
                "episode_id": sample.get("episode_id"),
                "start_frame": row_start(sample),
                "end_frame": row_end(sample),
                "is_target": idx == sample_idx,
            }
        )
        content.append({"type": "text", "text": f"Candidate {letter} video window:"})
        content.append({"type": "video", "video": media_video_path(sample)})
    return (
        [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": content},
        ],
        true_letter,
        candidate_records,
    )


def generate_messages(model, processor, messages: list[dict[str, Any]], args: argparse.Namespace) -> str:
    from qwen_omni_utils import process_mm_info

    for include_audio in (False,):
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(messages, use_audio_in_video=args.use_audio_in_video)
        if include_audio and has_empty_audio_items(audios):
            continue
        try:
            inputs = processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=args.use_audio_in_video,
            )
            break
        except RuntimeError as exc:
            if include_audio and is_empty_audio_exception(exc):
                continue
            raise
    else:
        raise RuntimeError("Unable to prepare retrieval prompt.")
    inputs = move_inputs(inputs, model)
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            thinker_return_dict_in_generate=True,
            use_audio_in_video=args.use_audio_in_video,
            return_audio=False,
            max_new_tokens=args.max_new_tokens,
        )
    text_ids = generated[0] if isinstance(generated, tuple) else generated
    sequences = text_ids.sequences if hasattr(text_ids, "sequences") else text_ids
    output_ids = sequences[:, inputs["input_ids"].shape[1] :]
    decoded = processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return decoded[0] if decoded else ""


def extract_ranking(raw: str, valid_letters: list[str]) -> list[str]:
    payload = parse_json_object(raw)
    value = payload.get("ranked_candidates") or payload.get("ranking") or payload.get("candidates")
    letters: list[str] = []
    if isinstance(value, list):
        source = " ".join(str(item) for item in value)
    else:
        source = str(value or raw)
    valid = set(valid_letters)
    for match in re.findall(r"\b[A-H]\b", source.upper()):
        if match in valid and match not in letters:
            letters.append(match)
    for letter in valid_letters:
        if letter not in letters:
            letters.append(letter)
    return letters


def score_retrieval(rows: list[dict[str, Any]]) -> dict[str, float]:
    reciprocal_ranks = []
    top1 = 0
    for row in rows:
        ranking = row.get("predicted_ranking") or []
        true_letter = row.get("true_letter")
        rank = ranking.index(true_letter) + 1 if true_letter in ranking else len(ranking) + 1
        reciprocal_ranks.append(1.0 / rank)
        top1 += int(bool(ranking) and ranking[0] == true_letter)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    return {
        "num_samples": len(rows),
        "mrr": mrr,
        "caption_grounding_mrr": mrr,
        "top1_accuracy": top1 / len(rows) if rows else 0.0,
    }


def score_task(task_id: str, spec: dict[str, Any], rows: list[dict[str, Any]], output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    task_dir = output_dir / task_id
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
            "true_letter",
            "predicted_ranking",
            "reciprocal_rank",
            "top1_correct",
            "raw_prediction",
        ],
    )
    metrics = score_retrieval(rows)
    primary_score = metrics["caption_grounding_mrr"]
    metrics.update(
        {
            "title": f"Qwen3-Omni v6 {spec['label']}",
            "status": "pass",
            "run_id": args.run_id,
            "task_id": task_id,
            "task_number": spec["task_number"],
            "task_label": spec["label"],
            "metric_key": spec["metric_key"],
            "primary_metric": spec["metric_key"],
            "primary_score": primary_score,
            "model_id": args.model_id,
            "adapter_dir": str(args.adapter_dir),
            "dataset_jsonl": str(args.dataset_jsonl),
            "eval_split": args.eval_split,
            "candidate_count": args.candidate_count,
            "sample_offset": args.sample_offset,
            "sample_stride": args.sample_stride,
            "scope": "held_out_test_qwen3_retrieval_task_probe",
            "score_policy": (
                "GPU-backed Qwen3-Omni v6 text-to-video retrieval probe. The text query is built "
                "from held-out action/subtask/object labels, candidates are shuffled staged mosaic "
                "video windows, and the score is MRR of the true window. This does not score tasks "
                "whose numeric/raw targets are absent from the export."
            ),
        }
    )
    write_json(task_dir / "metrics.json", metrics)
    return metrics


def main() -> int:
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = Path(__file__).resolve().parents[2] / "results/omni_finetune" / args.run_id
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.progress_jsonl = args.progress_jsonl or args.output_dir / "progress.jsonl"
    selected_tasks = select_tasks(args.tasks)
    samples = load_jsonl(args.dataset_jsonl)
    eval_pool = [idx for idx, sample in enumerate(samples) if sample.get("split") == args.eval_split and media_video_path(sample)]
    eval_indices = select_eval_indices(samples, args)
    if not eval_indices:
        raise ValueError("No evaluation samples with retrieval candidates selected.")

    append_jsonl(
        args.progress_jsonl,
        {
            "event": "eval_start",
            "timestamp": time.time(),
            "run_id": args.run_id,
            "tasks": selected_tasks,
            "num_eval_samples": len(eval_indices),
            "sample_offset": args.sample_offset,
            "sample_stride": args.sample_stride,
            "candidate_count": args.candidate_count,
        },
    )

    model, processor = load_model_processor(args)
    partial_by_task = {
        task_id: {
            row.get("prediction_id"): row
            for row in read_jsonl_if_exists(args.output_dir / task_id / "predictions.partial.jsonl")
            if row.get("prediction_id")
        }
        for task_id in selected_tasks
    }

    for task_id in selected_tasks:
        spec = TASK_SPECS[task_id]
        partial_path = args.output_dir / task_id / "predictions.partial.jsonl"
        for local_pos, sample_idx in enumerate(eval_indices, start=1):
            sample = samples[sample_idx]
            pred_id = prediction_id(task_id, sample)
            if pred_id in partial_by_task[task_id]:
                continue
            started = time.time()
            candidate_indices = build_candidate_indices(samples, eval_pool, sample_idx, task_id, args.candidate_count)
            messages, true_letter, candidate_records = build_messages(samples, sample_idx, candidate_indices, task_id, spec)
            raw = generate_messages(model, processor, messages, args)
            valid_letters = [record["letter"] for record in candidate_records]
            ranking = extract_ranking(raw, valid_letters)
            rank = ranking.index(true_letter) + 1 if true_letter in ranking else len(ranking) + 1
            row = {
                "prediction_id": pred_id,
                "id": sample.get("id"),
                "task_id": task_id,
                "task_label": spec["label"],
                "split": sample.get("split"),
                "episode_id": sample.get("episode_id"),
                "start_frame": row_start(sample),
                "end_frame": row_end(sample),
                "query_text": query_text(sample),
                "candidates": candidate_records,
                "true_letter": true_letter,
                "predicted_ranking": ranking,
                "reciprocal_rank": 1.0 / rank,
                "top1_correct": int(bool(ranking) and ranking[0] == true_letter),
                "raw_prediction": raw,
            }
            partial_by_task[task_id][pred_id] = row
            append_jsonl(partial_path, row)
            append_jsonl(
                args.progress_jsonl,
                {
                    "event": "sample_done",
                    "timestamp": time.time(),
                    "task_id": task_id,
                    "sample_index": local_pos,
                    "num_eval_samples": len(eval_indices),
                    "completed_samples_for_task": len(partial_by_task[task_id]),
                    "sample_id": sample.get("id"),
                    "seconds": round(time.time() - started, 3),
                },
            )

    task_metrics = {}
    for task_id in selected_tasks:
        rows = [partial_by_task[task_id][prediction_id(task_id, samples[idx])] for idx in eval_indices]
        task_metrics[task_id] = score_task(task_id, TASK_SPECS[task_id], rows, args.output_dir, args)

    summary = {
        "title": "Qwen3-Omni v6 Retrieval Task Probes",
        "status": "pass",
        "run_id": args.run_id,
        "model_id": args.model_id,
        "adapter_dir": str(args.adapter_dir),
        "dataset_jsonl": str(args.dataset_jsonl),
        "eval_split": args.eval_split,
        "candidate_count": args.candidate_count,
        "sample_offset": args.sample_offset,
        "sample_stride": args.sample_stride,
        "tasks": {
            task_id: {
                "task_number": metrics["task_number"],
                "task_label": metrics["task_label"],
                "metric_key": metrics["metric_key"],
                "primary_score": metrics["primary_score"],
                "num_samples": metrics["num_samples"],
                "metrics_json": str(args.output_dir / task_id / "metrics.json"),
            }
            for task_id, metrics in task_metrics.items()
        },
    }
    write_json(args.output_dir / "summary.json", summary)
    report_lines = [
        "# Qwen3-Omni v6 Retrieval Task Probes",
        "",
        f"- Run ID: `{args.run_id}`",
        f"- Dataset: `{args.dataset_jsonl}`",
        f"- Candidate count: `{args.candidate_count}`",
        f"- Shard: offset `{args.sample_offset}` / stride `{args.sample_stride}`",
        "",
        "| Task | Metric | Score | Samples |",
        "| --- | --- | ---: | ---: |",
    ]
    for task_id, metrics in task_metrics.items():
        report_lines.append(
            f"| {metrics['task_label']} | {metrics['metric_key']} | {metrics['primary_score']:.6f} | {metrics['num_samples']} |"
        )
    (args.output_dir / "RUN_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    append_jsonl(args.progress_jsonl, {"event": "eval_complete", "timestamp": time.time(), "run_id": args.run_id})
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

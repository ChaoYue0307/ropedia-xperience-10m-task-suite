#!/usr/bin/env python3
"""Evaluate Qwen3-Omni on future-target task probes from the 128-episode JSON.

This runner scores only task targets that can be derived from the current
multi-episode JSON export:

- Task 13: long-horizon next action, +100 frames.
- Task 14: long-horizon next subtask, +100 frames.
- Task 17: future object set, +100 frames.

It does not fabricate scores for regression, retrieval, raw-caption, or
missing-modality targets.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from eval_qwen3_omni_lora import dtype_arg, load_model_processor, move_inputs
from qwen3_omni_dataset_utils import (
    class_metrics,
    has_empty_audio_items,
    is_empty_audio_exception,
    load_jsonl,
    match_label,
)


TASK_SPECS: OrderedDict[str, dict[str, Any]] = OrderedDict(
    [
        (
            "long_horizon_next_action",
            {
                "task_number": 13,
                "label": "Long-Horizon Next-Action Forecasting",
                "family": "classification",
                "metric_key": "macro_f1",
                "prediction_key": "long_horizon_next_action",
                "target_field": "action",
                "option_field": "action_options",
            },
        ),
        (
            "next_subtask_forecast",
            {
                "task_number": 14,
                "label": "Long-Horizon Next-Subtask Forecasting",
                "family": "classification",
                "metric_key": "macro_f1",
                "prediction_key": "next_subtask_forecast",
                "target_field": "subtask",
                "option_field": "subtask_options",
            },
        ),
        (
            "object_set_forecast",
            {
                "task_number": 17,
                "label": "Future Object-Set Forecasting",
                "family": "multi_label",
                "metric_key": "micro_f1",
                "prediction_key": "object_set_forecast",
                "target_field": "objects",
                "option_field": None,
            },
        ),
    ]
)


SYSTEM_PROMPT = (
    "You are an embodied episode-understanding model for Ropedia/Xperience-10M. "
    "Return exactly one compact valid JSON object and no markdown, prose, code "
    "fences, explanations, or repeated text. If the target cannot be inferred "
    "from the visible/audio evidence, return unknown for the requested field."
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    parser.add_argument("--run-id", default="qwen3_future_task_probes")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--tasks", default="long_horizon_next_action,next_subtask_forecast,object_set_forecast")
    parser.add_argument("--future-frames", type=int, default=100)
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=48)
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


def normalize_objects(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    seen: set[str] = set()
    out: list[str] = []
    for item in value:
        text = normalize_text(item).casefold()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return sorted(out)


def answer(sample: dict[str, Any]) -> dict[str, Any]:
    payload = sample.get("answer_json")
    return payload if isinstance(payload, dict) else {}


def row_start(sample: dict[str, Any]) -> int:
    window = sample.get("center_window") if isinstance(sample.get("center_window"), dict) else {}
    return int(window.get("start_frame", 0) or 0)


def row_end(sample: dict[str, Any]) -> int:
    window = sample.get("center_window") if isinstance(sample.get("center_window"), dict) else {}
    return int(window.get("end_frame", row_start(sample)) or row_start(sample))


def by_episode_sorted(samples: list[dict[str, Any]]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = {}
    for idx, sample in enumerate(samples):
        grouped.setdefault(str(sample.get("episode_id")), []).append(idx)
    for indices in grouped.values():
        indices.sort(key=lambda i: row_start(samples[i]))
    return grouped


def future_index_map(samples: list[dict[str, Any]], frame_offset: int) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for indices in by_episode_sorted(samples).values():
        starts = np.asarray([row_start(samples[i]) for i in indices], dtype=np.int64)
        for idx in indices:
            target_start = row_start(samples[idx]) + frame_offset
            future_pos = int(np.searchsorted(starts, target_start, side="left"))
            if future_pos < len(indices):
                mapping[idx] = indices[future_pos]
    return mapping


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


def task_options(sample: dict[str, Any], spec: dict[str, Any]) -> list[str]:
    option_field = spec.get("option_field")
    options = sample.get(option_field) if option_field else None
    if isinstance(options, list) and options:
        return [str(item) for item in options]
    if spec["target_field"] == "action":
        options = sample.get("label_options")
        return [str(item) for item in options] if isinstance(options, list) else []
    return []


def build_task_prompt(sample: dict[str, Any], future_sample: dict[str, Any], task_id: str, spec: dict[str, Any], future_frames: int) -> str:
    start = row_start(sample)
    end = row_end(sample)
    future_start = row_start(future_sample)
    prediction_key = spec["prediction_key"]
    lines = [
        f"Task {spec['task_number']}: {spec['label']}",
        f"Episode: {sample.get('episode_id')}",
        f"Current visible/audio context frames: {start}-{end}",
        f"Predict the target at the future window starting near frame {start + future_frames} (resolved target start frame {future_start}).",
    ]
    options = task_options(sample, spec)
    if task_id == "long_horizon_next_action":
        lines.extend(
            [
                "Return JSON only with this schema:",
                f'{{"{prediction_key}":"<exact action option or unknown>"}}',
                "Copy exactly one action label from this list:",
                "\n".join(f"- {option}" for option in options),
            ]
        )
    elif task_id == "next_subtask_forecast":
        lines.extend(
            [
                "Return JSON only with this schema:",
                f'{{"{prediction_key}":"<exact subtask option or unknown>"}}',
                "Copy exactly one subtask label from this list:",
                "\n".join(f"- {option}" for option in options),
            ]
        )
    elif task_id == "object_set_forecast":
        lines.extend(
            [
                "Return JSON only with this schema:",
                f'{{"{prediction_key}":["<0 to 8 short object names>"]}}',
                "List the objects likely to be active or manipulated in that future window. Use short object names.",
            ]
        )
    else:
        raise ValueError(f"unknown task: {task_id}")
    return "\n".join(lines)


def build_messages(
    sample: dict[str, Any],
    future_sample: dict[str, Any],
    task_id: str,
    spec: dict[str, Any],
    future_frames: int,
    *,
    include_audio: bool = True,
) -> list[dict[str, Any]]:
    media = sample.get("media") if isinstance(sample.get("media"), dict) else {}
    video_path = media.get("mosaic_video_path") or sample.get("primary_video_path")
    audio_path = media.get("audio_path")
    content: list[dict[str, Any]] = []
    if video_path:
        content.append({"type": "video", "video": video_path})
    if include_audio and audio_path:
        content.append({"type": "audio", "audio": audio_path})
    content.append({"type": "text", "text": build_task_prompt(sample, future_sample, task_id, spec, future_frames)})
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]


def generate_messages(
    model,
    processor,
    sample: dict[str, Any],
    future_sample: dict[str, Any],
    task_id: str,
    spec: dict[str, Any],
    args: argparse.Namespace,
) -> str:
    from qwen_omni_utils import process_mm_info

    for include_audio in (True, False):
        messages = build_messages(sample, future_sample, task_id, spec, args.future_frames, include_audio=include_audio)
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
        raise RuntimeError("Unable to prepare multimodal task prompt after dropping empty audio.")
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


def select_tasks(spec: str) -> list[str]:
    if spec.strip().lower() == "all":
        return list(TASK_SPECS)
    tasks = [item.strip() for item in spec.split(",") if item.strip()]
    unknown = [task for task in tasks if task not in TASK_SPECS]
    if unknown:
        raise ValueError(f"unknown tasks: {unknown}")
    return tasks


def select_eval_indices(samples: list[dict[str, Any]], args: argparse.Namespace) -> list[int]:
    indices = [idx for idx, sample in enumerate(samples) if sample.get("split") == args.eval_split]
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


def task_target(future_sample: dict[str, Any], spec: dict[str, Any]) -> Any:
    value = answer(future_sample).get(spec["target_field"])
    if spec["family"] == "multi_label":
        return normalize_objects(value)
    return normalize_text(value)


def extract_prediction(raw: str, sample: dict[str, Any], spec: dict[str, Any]) -> Any:
    payload = parse_json_object(raw)
    value = payload.get(spec["prediction_key"])
    if spec["family"] == "multi_label":
        return normalize_objects(value)
    options = task_options(sample, spec)
    return match_label(str(value or raw), options) if options else normalize_text(value)


def object_set_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    tp = fp = fn = exact = 0
    for row in rows:
        true_set = set(row.get("true_value") or [])
        pred_set = set(row.get("predicted_value") or [])
        tp += len(true_set & pred_set)
        fp += len(pred_set - true_set)
        fn += len(true_set - pred_set)
        exact += int(true_set == pred_set)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    micro_f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "num_samples": len(rows),
        "micro_f1": micro_f1,
        "precision": precision,
        "recall": recall,
        "exact_match": exact / len(rows) if rows else 0.0,
    }


def score_task(task_id: str, spec: dict[str, Any], rows: list[dict[str, Any]], output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    task_dir = output_dir / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(task_dir / "predictions.jsonl", rows)
    csv_rows = []
    for row in rows:
        csv_rows.append(
            {
                "id": row["id"],
                "episode_id": row["episode_id"],
                "split": row["split"],
                "start_frame": row["start_frame"],
                "end_frame": row["end_frame"],
                "future_start_frame": row["future_start_frame"],
                "true_value": json.dumps(row["true_value"], ensure_ascii=False)
                if isinstance(row["true_value"], list)
                else row["true_value"],
                "predicted_value": json.dumps(row["predicted_value"], ensure_ascii=False)
                if isinstance(row["predicted_value"], list)
                else row["predicted_value"],
                "raw_prediction": row["raw_prediction"],
                "correct": row.get("correct"),
            }
        )
    write_csv(
        task_dir / "predictions.csv",
        csv_rows,
        [
            "id",
            "episode_id",
            "split",
            "start_frame",
            "end_frame",
            "future_start_frame",
            "true_value",
            "predicted_value",
            "raw_prediction",
            "correct",
        ],
    )

    if spec["family"] == "classification":
        options = sorted({str(row["true_value"]) for row in rows if row.get("true_value")})
        metrics, per_class, _ = class_metrics(
            [str(row["true_value"]) for row in rows],
            [str(row["predicted_value"]) for row in rows],
            options,
        )
        metrics[f"{task_id}_macro_f1"] = metrics["macro_f1"]
        metrics[f"{task_id}_accuracy"] = metrics["accuracy"]
        write_csv(task_dir / "per_class_metrics.csv", per_class, ["class_name", "support", "predicted", "precision", "recall", "f1"])
        primary_score = metrics["macro_f1"]
    else:
        metrics = object_set_metrics(rows)
        metrics[f"{task_id}_micro_f1"] = metrics["micro_f1"]
        metrics[f"{task_id}_exact_match"] = metrics["exact_match"]
        primary_score = metrics["micro_f1"]

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
            "future_frames": args.future_frames,
            "sample_offset": args.sample_offset,
            "sample_stride": args.sample_stride,
            "scope": "held_out_test_qwen3_future_task_probe",
            "score_policy": (
                "GPU-backed Qwen3-Omni v6 generation for targets derivable from the current "
                "128-episode JSON export. This package does not score tasks whose target fields "
                "are absent from the export."
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
    future_map = future_index_map(samples, args.future_frames)
    eval_indices = [idx for idx in select_eval_indices(samples, args) if idx in future_map]
    if not eval_indices:
        raise ValueError("No evaluation samples with future targets selected.")

    append_jsonl(
        args.progress_jsonl,
        {
            "event": "eval_start",
            "timestamp": time.time(),
            "run_id": args.run_id,
            "tasks": selected_tasks,
            "num_eval_samples_with_future": len(eval_indices),
            "sample_offset": args.sample_offset,
            "sample_stride": args.sample_stride,
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
            future_sample = samples[future_map[sample_idx]]
            pred_id = prediction_id(task_id, sample)
            if pred_id in partial_by_task[task_id]:
                continue
            started = time.time()
            raw = generate_messages(model, processor, sample, future_sample, task_id, spec, args)
            true_value = task_target(future_sample, spec)
            predicted_value = extract_prediction(raw, sample, spec)
            row = {
                "prediction_id": pred_id,
                "id": sample.get("id"),
                "target_future_id": future_sample.get("id"),
                "task_id": task_id,
                "task_label": spec["label"],
                "split": sample.get("split"),
                "episode_id": sample.get("episode_id"),
                "start_frame": row_start(sample),
                "end_frame": row_end(sample),
                "future_start_frame": row_start(future_sample),
                "future_end_frame": row_end(future_sample),
                "true_value": true_value,
                "predicted_value": predicted_value,
                "raw_prediction": raw,
                "correct": int(true_value == predicted_value) if spec["family"] == "classification" else int(set(true_value) == set(predicted_value)),
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
        "title": "Qwen3-Omni v6 Future Task Probes",
        "status": "pass",
        "run_id": args.run_id,
        "model_id": args.model_id,
        "adapter_dir": str(args.adapter_dir),
        "dataset_jsonl": str(args.dataset_jsonl),
        "eval_split": args.eval_split,
        "future_frames": args.future_frames,
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
        "# Qwen3-Omni v6 Future Task Probes",
        "",
        f"- Run ID: `{args.run_id}`",
        f"- Dataset: `{args.dataset_jsonl}`",
        f"- Future offset: `{args.future_frames}` frames",
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

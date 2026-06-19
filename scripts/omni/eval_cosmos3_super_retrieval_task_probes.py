#!/usr/bin/env python3
"""Evaluate Cosmos3-Super Reasoner on target-backed retrieval probes.

This runner mirrors the Qwen3-Omni retrieval-task contract, but calls an
OpenAI-compatible Cosmos3-Super server. It is intentionally metrics-only: it
does not fine-tune weights, invent targets, or fill matrix cells unless the
task writes a real held-out metrics.json artifact.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from eval_qwen3_omni_retrieval_task_probes import (
    SENSOR_TARGET_TASKS,
    TASK_SPECS,
    SensorFeatureCache,
    answer,
    artifact_query_text,
    build_candidate_indices,
    build_messages,
    extract_ranking,
    future_index_map,
    has_camera_view_pair,
    has_sensor_feature,
    media_video_path,
    prediction_id,
    read_jsonl_if_exists,
    row_end,
    row_start,
    score_retrieval,
    select_eval_indices,
    select_tasks,
    write_json,
    write_jsonl,
)
from qwen3_omni_dataset_utils import load_jsonl


SYSTEM_PROMPT = (
    "You are an embodied episode-understanding model for Ropedia/Xperience-10M. "
    "Return exactly one compact valid JSON object and no markdown, prose, code "
    "fences, explanations, or repeated text."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    parser.add_argument("--run-id", default="xperience10m_cosmos3_super_retrieval_task_probes")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="cosmos3-super-local")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--tasks", default="cross_modal_retrieval")
    parser.add_argument("--candidate-count", type=int, default=4)
    parser.add_argument("--future-frames", type=int, default=100)
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--media-mode", choices=["video_url", "text_only"], default="video_url")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress-jsonl", type=Path)
    return parser.parse_args()


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


def normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def file_url(path_text: str) -> str:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = path.resolve()
    return path.as_uri()


def http_json(method: str, url: str, payload: dict[str, Any] | None, timeout: float) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
    return json.loads(body) if body else {}


def server_info(args: argparse.Namespace) -> dict[str, Any]:
    try:
        return http_json("GET", f"{normalize_base_url(args.base_url)}/models", None, min(args.request_timeout, 30.0))
    except Exception as exc:  # noqa: BLE001 - diagnostic only.
        return {"error": f"{type(exc).__name__}: {exc}"}


def qwen_content_to_openai(content: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for item in content:
        kind = item.get("type")
        if kind == "text":
            converted.append({"type": "text", "text": str(item.get("text", ""))})
        elif kind == "video":
            path = str(item.get("video") or "")
            if args.media_mode == "video_url" and path:
                converted.append({"type": "video_url", "video_url": {"url": file_url(path)}})
            elif path:
                converted.append({"type": "text", "text": f"[video omitted in text_only mode: {path}]"})
    return converted


def openai_messages(qwen_messages: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for message in qwen_messages:
        role = str(message.get("role") or "user")
        content = message.get("content")
        if isinstance(content, list):
            messages.append({"role": role, "content": qwen_content_to_openai(content, args)})
        else:
            messages.append({"role": role, "content": str(content or "")})
    return messages


def chat_completion(qwen_messages: list[dict[str, Any]], args: argparse.Namespace) -> tuple[str, dict[str, Any], float]:
    payload = {
        "model": args.model,
        "messages": openai_messages(qwen_messages, args),
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
    }
    started = time.time()
    response = http_json(
        "POST",
        f"{normalize_base_url(args.base_url)}/chat/completions",
        payload,
        args.request_timeout,
    )
    choices = response.get("choices") if isinstance(response.get("choices"), list) else []
    message = choices[0].get("message") if choices and isinstance(choices[0], dict) else {}
    content = message.get("content") if isinstance(message, dict) else ""
    if isinstance(content, list):
        text = "\n".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
    else:
        text = str(content or "")
    return text, response, time.time() - started


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
                "target_id": row.get("target_id"),
                "target_start_frame": row.get("target_start_frame"),
                "target_end_frame": row.get("target_end_frame"),
                "true_letter": row["true_letter"],
                "predicted_ranking": json.dumps(row["predicted_ranking"], ensure_ascii=False),
                "reciprocal_rank": row["reciprocal_rank"],
                "top1_correct": row["top1_correct"],
                "latency_seconds": row.get("latency_seconds"),
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
            "target_id",
            "target_start_frame",
            "target_end_frame",
            "true_letter",
            "predicted_ranking",
            "reciprocal_rank",
            "top1_correct",
            "latency_seconds",
            "raw_prediction",
        ],
    )
    metrics = score_retrieval(rows)
    primary_score = metrics[spec["metric_key"]]
    metrics.update(
        {
            "title": f"Cosmos3-Super Reasoner {spec['label']}",
            "status": "pass",
            "run_id": args.run_id,
            "task_id": task_id,
            "task_number": spec["task_number"],
            "task_label": spec["label"],
            "metric_key": spec["metric_key"],
            "primary_metric": spec["metric_key"],
            "primary_score": primary_score,
            "model": args.model,
            "base_url": args.base_url,
            "dataset_jsonl": str(args.dataset_jsonl),
            "eval_split": args.eval_split,
            "candidate_count": args.candidate_count,
            "future_frames": args.future_frames,
            "sample_offset": args.sample_offset,
            "sample_stride": args.sample_stride,
            "media_mode": args.media_mode,
            "scope": "held_out_test_cosmos3_super_retrieval_task_probe",
            "score_policy": (
                "GPU-backed Cosmos3-Super Reasoner retrieval probe over real held-out "
                "candidate windows or staged sensor targets. The score is MRR of the "
                "true candidate; no labels are fabricated and no weights are updated."
            ),
        }
    )
    write_json(task_dir / "metrics.json", metrics)
    return metrics


def main() -> int:
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = Path(__file__).resolve().parents[2] / "results" / "omni_finetune" / args.run_id
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.progress_jsonl = args.progress_jsonl or args.output_dir / "progress.jsonl"
    selected_tasks = select_tasks(args.tasks)
    samples = load_jsonl(args.dataset_jsonl)
    eval_pool = [idx for idx, sample in enumerate(samples) if sample.get("split") == args.eval_split and media_video_path(sample)]
    eval_indices = select_eval_indices(samples, args)
    if "cross_modal_retrieval" in selected_tasks:
        eval_indices = [idx for idx in eval_indices if has_sensor_feature(samples[idx])]
        eval_pool = [idx for idx in eval_pool if has_sensor_feature(samples[idx])]
    if any(task_id in SENSOR_TARGET_TASKS for task_id in selected_tasks):
        eval_indices = [idx for idx in eval_indices if has_sensor_feature(samples[idx])]
        eval_pool = [idx for idx in eval_pool if has_sensor_feature(samples[idx])]
    future_targets = future_index_map(samples, args.future_frames) if "hand_trajectory_forecast" in selected_tasks else {}
    if "hand_trajectory_forecast" in selected_tasks:
        eval_indices = [
            idx
            for idx in eval_indices
            if idx in future_targets and has_sensor_feature(samples[future_targets[idx]])
        ]
    if "camera_view_sync_retrieval" in selected_tasks:
        eval_indices = [idx for idx in eval_indices if has_camera_view_pair(samples[idx])]
        eval_pool = [idx for idx in eval_pool if has_camera_view_pair(samples[idx])]
    if not eval_indices:
        raise ValueError("No evaluation samples with retrieval candidates selected.")

    write_json(args.output_dir / "server_info.json", server_info(args))
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
            "future_frames": args.future_frames,
            "model": args.model,
            "base_url": args.base_url,
            "media_mode": args.media_mode,
        },
    )

    sensor_cache = (
        SensorFeatureCache()
        if "cross_modal_retrieval" in selected_tasks
        or any(task_id in SENSOR_TARGET_TASKS for task_id in selected_tasks)
        else None
    )
    camera_clip_dir = args.output_dir / "camera_view_sync_clips" if "camera_view_sync_retrieval" in selected_tasks else None
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
            if args.resume and pred_id in partial_by_task[task_id]:
                continue
            started = time.time()
            target_idx = future_targets[sample_idx] if task_id == "hand_trajectory_forecast" else sample_idx
            candidate_indices = build_candidate_indices(
                samples,
                eval_pool,
                sample_idx,
                task_id,
                args.candidate_count,
                target_idx=target_idx,
            )
            qwen_messages, true_letter, candidate_records = build_messages(
                samples,
                sample_idx,
                target_idx,
                candidate_indices,
                task_id,
                spec,
                sensor_cache=sensor_cache,
                camera_clip_dir=camera_clip_dir,
                future_frames=args.future_frames,
            )
            raw, response, latency = chat_completion(qwen_messages, args)
            valid_letters = [record["letter"] for record in candidate_records]
            ranking = extract_ranking(raw, valid_letters)
            rank = ranking.index(true_letter) + 1 if true_letter in ranking else len(ranking) + 1
            usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
            row = {
                "prediction_id": pred_id,
                "id": sample.get("id"),
                "task_id": task_id,
                "task_label": spec["label"],
                "split": sample.get("split"),
                "episode_id": sample.get("episode_id"),
                "start_frame": row_start(sample),
                "end_frame": row_end(sample),
                "query_text": artifact_query_text(task_id, sample, sensor_cache, future_frames=args.future_frames),
                "target_id": samples[target_idx].get("id"),
                "target_start_frame": row_start(samples[target_idx]),
                "target_end_frame": row_end(samples[target_idx]),
                "candidates": candidate_records,
                "true_letter": true_letter,
                "predicted_ranking": ranking,
                "reciprocal_rank": 1.0 / rank,
                "top1_correct": int(bool(ranking) and ranking[0] == true_letter),
                "latency_seconds": round(latency, 3),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
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
        "title": "Cosmos3-Super Reasoner Retrieval Task Probes",
        "status": "pass",
        "run_id": args.run_id,
        "model": args.model,
        "base_url": args.base_url,
        "dataset_jsonl": str(args.dataset_jsonl),
        "eval_split": args.eval_split,
        "candidate_count": args.candidate_count,
        "future_frames": args.future_frames,
        "sample_offset": args.sample_offset,
        "sample_stride": args.sample_stride,
        "media_mode": args.media_mode,
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
        "# Cosmos3-Super Reasoner Retrieval Task Probes",
        "",
        f"- Run ID: `{args.run_id}`",
        f"- Model: `{args.model}`",
        f"- API base URL: `{args.base_url}`",
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

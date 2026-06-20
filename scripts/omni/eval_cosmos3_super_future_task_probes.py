#!/usr/bin/env python3
"""Evaluate Cosmos3-Super Reasoner on target-backed future-task probes.

This is the server-backed Cosmos3-Super counterpart to
eval_qwen3_omni_future_task_probes.py. It keeps the same 128-episode task
contracts and metrics, but calls an OpenAI-compatible Cosmos3-Super server
instead of loading Qwen locally. In text_only mode the run is explicitly a
text-only model-output probe; it does not claim video/audio evidence was used.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from eval_qwen3_omni_future_task_probes import (
    TASK_SPECS,
    append_jsonl,
    build_messages,
    extract_prediction,
    future_index_map,
    prediction_id,
    read_jsonl_if_exists,
    row_end,
    row_start,
    score_task as qwen_score_task,
    select_eval_indices,
    select_tasks,
    task_target_value,
    time_to_transition_map,
    write_json,
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
    parser.add_argument("--run-id", default="xperience10m_cosmos3_super_future_task_probes")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="cosmos3-super-local")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--tasks", default="temporal_order,misalignment_detection,next_subtask_forecast,object_set_forecast")
    parser.add_argument("--future-frames", type=int, default=100)
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--media-mode", choices=["video_url", "text_only"], default="text_only")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress-jsonl", type=Path)
    return parser.parse_args()


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
        elif kind == "audio":
            path = str(item.get("audio") or "")
            if path:
                converted.append({"type": "text", "text": f"[audio omitted in text_only mode: {path}]"})
    return converted


def openai_messages(qwen_messages: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for message in qwen_messages:
        role = str(message.get("role") or "user")
        if role == "system":
            continue
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
    fake_args = argparse.Namespace(
        run_id=args.run_id,
        model_id=args.model,
        adapter_dir=Path(""),
        dataset_jsonl=args.dataset_jsonl,
        eval_split=args.eval_split,
        future_frames=args.future_frames,
        sample_offset=args.sample_offset,
        sample_stride=args.sample_stride,
    )
    metrics = qwen_score_task(task_id, spec, rows, output_dir, fake_args)
    metrics.update(
        {
            "title": f"Cosmos3-Super Reasoner {spec['label']}",
            "model": args.model,
            "base_url": args.base_url,
            "media_mode": args.media_mode,
            "scope": "held_out_test_cosmos3_super_future_task_probe",
            "score_policy": (
                "GPU-backed Cosmos3-Super Reasoner future-task probe over real held-out "
                "targets derivable from the 128-episode JSON export. In text_only mode, "
                "raw video/audio is omitted and the artifact is labeled as a text-only "
                "model-output probe; no labels are fabricated and no weights are updated."
            ),
        }
    )
    write_json(output_dir / task_id / "metrics.json", metrics)
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
    transition_targets = time_to_transition_map(samples)
    eval_indices = [idx for idx in select_eval_indices(samples, args) if idx in future_map]
    if not eval_indices:
        raise ValueError("No evaluation samples with future targets selected.")

    write_json(args.output_dir / "server_info.json", server_info(args))
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
            "future_frames": args.future_frames,
            "model": args.model,
            "base_url": args.base_url,
            "media_mode": args.media_mode,
        },
    )

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
            if args.resume and pred_id in partial_by_task[task_id]:
                continue
            started = time.time()
            qwen_messages = build_messages(
                sample,
                future_sample,
                task_id,
                spec,
                args.future_frames,
                include_audio=args.media_mode != "text_only",
            )
            raw, response, latency = chat_completion(qwen_messages, args)
            true_value = task_target_value(task_id, sample, future_sample, spec, transition_targets, sample_idx)
            predicted_value = extract_prediction(raw, sample, spec)
            if spec["family"] == "classification":
                correct = int(true_value == predicted_value)
            elif spec["family"] == "multi_label":
                correct = int(set(true_value) == set(predicted_value))
            else:
                correct = int(predicted_value is not None and abs(float(true_value) - float(predicted_value)) <= 20.0)
            usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
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
                "correct": correct,
                "latency_seconds": round(latency, 3),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
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
        "title": "Cosmos3-Super Reasoner Future Task Probes",
        "status": "pass",
        "run_id": args.run_id,
        "model": args.model,
        "base_url": args.base_url,
        "dataset_jsonl": str(args.dataset_jsonl),
        "eval_split": args.eval_split,
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
        "# Cosmos3-Super Reasoner Future Task Probes",
        "",
        f"- Run ID: `{args.run_id}`",
        f"- Model: `{args.model}`",
        f"- API base URL: `{args.base_url}`",
        f"- Dataset: `{args.dataset_jsonl}`",
        f"- Future offset: `{args.future_frames}` frames",
        f"- Media mode: `{args.media_mode}`",
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

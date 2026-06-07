#!/usr/bin/env python3
"""Evaluate Cosmos3-Super Reasoner through a local OpenAI-compatible vLLM API."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from qwen3_omni_dataset_utils import (
    SYSTEM_PROMPT,
    build_user_prompt,
    class_metrics,
    json_validity_rate,
    label_counts,
    load_jsonl,
    match_label,
    parse_answer_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    workspace_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    parser.add_argument("--run-id", default="xperience10m_cosmos3_super_reasoner_eval")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="cosmos3-super-local")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--media-mode", choices=["video_url", "text_only"], default="video_url")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress-jsonl", type=Path)
    parser.add_argument("--partial-predictions-jsonl", type=Path)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


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


def server_info(base_url: str, timeout: float) -> dict[str, Any]:
    try:
        return http_json("GET", f"{normalize_base_url(base_url)}/models", None, timeout)
    except Exception as exc:  # noqa: BLE001 - server metadata is diagnostic only.
        return {"error": f"{type(exc).__name__}: {exc}"}


def file_url(path_text: str) -> str:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = path.resolve()
    return path.as_uri()


def sample_video_path(sample: dict[str, Any]) -> str | None:
    media = sample.get("media") if isinstance(sample.get("media"), dict) else {}
    value = media.get("mosaic_video_path") or sample.get("primary_video_path")
    return str(value) if value else None


def build_messages(sample: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    prompt = build_user_prompt(sample, sample.get("label_options") or [])
    content: list[dict[str, Any]] = []
    video_path = sample_video_path(sample)
    if args.media_mode == "video_url" and video_path:
        content.append({"type": "video_url", "video_url": {"url": file_url(video_path)}})
    content.append({"type": "text", "text": prompt})
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def response_text(response: dict[str, Any]) -> str:
    choices = response.get("choices") if isinstance(response.get("choices"), list) else []
    if not choices:
        return ""
    message = choices[0].get("message") if isinstance(choices[0], dict) else {}
    content = message.get("content") if isinstance(message, dict) else ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return str(content or "")


def chat_completion(sample: dict[str, Any], args: argparse.Namespace) -> tuple[str, dict[str, Any], float]:
    payload = {
        "model": args.model,
        "messages": build_messages(sample, args),
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
    return response_text(response), response, time.time() - started


def field_accuracy(rows: list[dict[str, Any]], field: str) -> float | None:
    valid_rows = [row for row in rows if row["true_json"].get(field) != "unknown"]
    if not valid_rows:
        return None
    return sum(row["pred_json"].get(field) == row["true_json"].get(field) for row in valid_rows) / len(valid_rows)


def object_micro_f1(rows: list[dict[str, Any]]) -> float | None:
    tp = fp = fn = 0
    for row in rows:
        true_objects = set(row["true_json"].get("objects") or [])
        pred_objects = set(row["pred_json"].get("objects") or [])
        tp += len(true_objects & pred_objects)
        fp += len(pred_objects - true_objects)
        fn += len(true_objects - pred_objects)
    if tp + fp + fn == 0:
        return None
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def prediction_row(sample: dict[str, Any], args: argparse.Namespace, train_labels: set[str]) -> dict[str, Any]:
    raw, response, seconds = chat_completion(sample, args)
    pred_json = parse_answer_json(raw)
    true_json = sample.get("answer_json", {})
    label_options = sample.get("action_options") or sample.get("label_options") or []
    predicted = match_label(str(pred_json.get("action", raw)), label_options)
    true_action = true_json.get("action", sample.get("label", "unknown"))
    usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
    return {
        "id": sample["id"],
        "target": sample.get("target"),
        "split": sample.get("split", "unspecified"),
        "episode_id": sample["episode_id"],
        "center_window": sample.get("center_window"),
        "media_mode": args.media_mode,
        "video_path": sample_video_path(sample),
        "true_label": true_action,
        "raw_prediction": raw,
        "true_json": true_json,
        "pred_json": pred_json,
        "predicted_label": predicted,
        "correct": int(predicted == true_action),
        "true_label_seen_in_train": int(true_action in train_labels),
        "latency_seconds": round(seconds, 3),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def selected_samples(samples: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    eval_samples = [sample for sample in samples if sample.get("split") == args.eval_split]
    if args.sample_stride < 1:
        raise ValueError("--sample-stride must be >= 1")
    if args.sample_offset < 0 or args.sample_offset >= args.sample_stride:
        raise ValueError("--sample-offset must satisfy 0 <= offset < stride")
    if args.sample_stride > 1:
        eval_samples = [sample for idx, sample in enumerate(eval_samples) if idx % args.sample_stride == args.sample_offset]
    if args.sample_limit > 0:
        eval_samples = eval_samples[: args.sample_limit]
    return eval_samples


def evaluate(samples: list[dict[str, Any]], args: argparse.Namespace, train_labels: set[str]) -> list[dict[str, Any]]:
    sample_ids = [sample["id"] for sample in samples]
    completed_by_id = {}
    if args.resume:
        for row in read_jsonl_if_exists(args.partial_predictions_jsonl):
            if row.get("id") in sample_ids:
                completed_by_id[row["id"]] = row
    elif args.partial_predictions_jsonl.exists():
        args.partial_predictions_jsonl.unlink()

    append_jsonl(
        args.progress_jsonl,
        {
            "event": "eval_start",
            "timestamp": time.time(),
            "run_id": args.run_id,
            "model": args.model,
            "base_url": args.base_url,
            "eval_split": args.eval_split,
            "media_mode": args.media_mode,
            "sample_offset": args.sample_offset,
            "sample_stride": args.sample_stride,
            "num_eval_samples": len(samples),
            "completed_before_start": len(completed_by_id),
            "concurrency": args.concurrency,
            "resume": args.resume,
        },
    )

    pending = [sample for sample in samples if sample["id"] not in completed_by_id]
    if pending and args.concurrency == 1:
        for index, sample in enumerate(samples, start=1):
            if sample["id"] in completed_by_id:
                continue
            started = time.time()
            try:
                row = prediction_row(sample, args, train_labels)
            except Exception as exc:  # noqa: BLE001 - fail-fast with structured progress.
                append_jsonl(
                    args.progress_jsonl,
                    {
                        "event": "sample_error",
                        "timestamp": time.time(),
                        "sample_index": index,
                        "num_eval_samples": len(samples),
                        "sample_id": sample["id"],
                        "episode_id": sample.get("episode_id"),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                raise
            completed_by_id[sample["id"]] = row
            append_jsonl(args.partial_predictions_jsonl, row)
            append_jsonl(
                args.progress_jsonl,
                {
                    "event": "sample_done",
                    "timestamp": time.time(),
                    "sample_index": index,
                    "num_eval_samples": len(samples),
                    "completed_samples": len(completed_by_id),
                    "sample_id": sample["id"],
                    "episode_id": sample.get("episode_id"),
                    "seconds": round(time.time() - started, 3),
                },
            )
    elif pending:
        index_by_id = {sample["id"]: idx for idx, sample in enumerate(samples, start=1)}
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {executor.submit(prediction_row, sample, args, train_labels): sample for sample in pending}
            for future in concurrent.futures.as_completed(futures):
                sample = futures[future]
                index = index_by_id[sample["id"]]
                try:
                    row = future.result()
                except Exception as exc:  # noqa: BLE001 - fail-fast with structured progress.
                    append_jsonl(
                        args.progress_jsonl,
                        {
                            "event": "sample_error",
                            "timestamp": time.time(),
                            "sample_index": index,
                            "num_eval_samples": len(samples),
                            "sample_id": sample["id"],
                            "episode_id": sample.get("episode_id"),
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        },
                    )
                    raise
                completed_by_id[sample["id"]] = row
                append_jsonl(args.partial_predictions_jsonl, row)
                append_jsonl(
                    args.progress_jsonl,
                    {
                        "event": "sample_done",
                        "timestamp": time.time(),
                        "sample_index": index,
                        "num_eval_samples": len(samples),
                        "completed_samples": len(completed_by_id),
                        "sample_id": sample["id"],
                        "episode_id": sample.get("episode_id"),
                        "seconds": row.get("latency_seconds"),
                    },
                )

    rows = [completed_by_id[sample_id] for sample_id in sample_ids if sample_id in completed_by_id]
    if len(rows) != len(samples):
        raise RuntimeError(f"Only {len(rows)} of {len(samples)} evaluation samples completed.")
    return rows


def main() -> int:
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = Path(__file__).resolve().parents[2] / "results" / "omni_finetune" / args.run_id
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.progress_jsonl = args.progress_jsonl or args.output_dir / "progress.jsonl"
    args.partial_predictions_jsonl = args.partial_predictions_jsonl or args.output_dir / "predictions.partial.jsonl"

    samples = load_jsonl(args.dataset_jsonl)
    eval_samples = selected_samples(samples, args)
    if not eval_samples:
        raise ValueError("No evaluation samples selected.")
    train_labels = {
        sample.get("answer_json", {}).get("action", sample.get("label", "unknown"))
        for sample in samples
        if sample.get("split") == args.train_split
    }
    eval_labels = {
        sample.get("answer_json", {}).get("action", sample.get("label", "unknown"))
        for sample in eval_samples
    }
    unseen_labels = sorted(eval_labels - train_labels)
    server_payload = server_info(args.base_url, min(args.request_timeout, 30.0))
    write_json(args.output_dir / "server_info.json", server_payload)

    rows = evaluate(eval_samples, args, train_labels)
    metrics, per_class, cm = class_metrics(
        [row["true_label"] for row in rows],
        [row["predicted_label"] for row in rows],
        eval_samples[0].get("label_options") or [],
    )
    seen_rows = [row for row in rows if row["true_label_seen_in_train"]]
    unseen_rows = [row for row in rows if not row["true_label_seen_in_train"]]
    latencies = [row["latency_seconds"] for row in rows if isinstance(row.get("latency_seconds"), (int, float))]
    prompt_tokens = [row["prompt_tokens"] for row in rows if isinstance(row.get("prompt_tokens"), int)]
    completion_tokens = [row["completion_tokens"] for row in rows if isinstance(row.get("completion_tokens"), int)]
    metrics.update(
        {
            "model": args.model,
            "base_url": args.base_url,
            "dataset_jsonl": str(args.dataset_jsonl),
            "eval_split": args.eval_split,
            "train_split": args.train_split,
            "media_mode": args.media_mode,
            "sample_offset": args.sample_offset,
            "sample_stride": args.sample_stride,
            "concurrency": args.concurrency,
            "num_eval_episodes": len({row["episode_id"] for row in rows}),
            "held_out_episode_count": len({row["episode_id"] for row in rows}),
            "unseen_eval_labels": unseen_labels,
            "num_unseen_label_samples": len(unseen_rows),
            "seen_label_accuracy": sum(row["correct"] for row in seen_rows) / len(seen_rows) if seen_rows else None,
            "unseen_label_accuracy": sum(row["correct"] for row in unseen_rows) / len(unseen_rows) if unseen_rows else None,
            "eval_label_counts": label_counts(eval_samples),
            "json_validity_rate": json_validity_rate([row["raw_prediction"] for row in rows]),
            "action_macro_f1": metrics["macro_f1"],
            "subtask_accuracy": field_accuracy(rows, "subtask"),
            "transition_accuracy": field_accuracy(rows, "transition"),
            "next_action_accuracy": field_accuracy(rows, "next_action"),
            "contact_accuracy": field_accuracy(rows, "contact"),
            "object_micro_f1": object_micro_f1(rows),
            "mean_latency_seconds": sum(latencies) / len(latencies) if latencies else None,
            "mean_prompt_tokens": sum(prompt_tokens) / len(prompt_tokens) if prompt_tokens else None,
            "mean_completion_tokens": sum(completion_tokens) / len(completion_tokens) if completion_tokens else None,
            "server_info_file": "server_info.json",
            "run_kind": "cosmos3_super_reasoner_vllm_zero_shot_eval",
            "weights_policy": "No new Cosmos3-Super fine-tuned weights are produced by this evaluator; it runs the staged base Reasoner weights through vLLM.",
        }
    )

    write_jsonl(args.output_dir / "predictions.jsonl", rows)
    write_csv(
        args.output_dir / "predictions.csv",
        rows,
        [
            "id",
            "target",
            "split",
            "episode_id",
            "center_window",
            "media_mode",
            "video_path",
            "true_label",
            "raw_prediction",
            "predicted_label",
            "correct",
            "true_label_seen_in_train",
            "latency_seconds",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
        ],
    )
    write_csv(
        args.output_dir / "per_class_metrics.csv",
        per_class,
        ["class_name", "support", "predicted", "precision", "recall", "f1"],
    )
    labels = metrics["labels"]
    with (args.output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["true\\pred"] + labels)
        for label, row in zip(labels, cm):
            writer.writerow([label] + row)
    write_json(args.output_dir / "metrics.json", metrics)
    append_jsonl(
        args.progress_jsonl,
        {
            "event": "eval_complete",
            "timestamp": time.time(),
            "run_id": args.run_id,
            "num_eval_samples": len(rows),
            "metrics_json": str(args.output_dir / "metrics.json"),
        },
    )
    report = [
        "# Cosmos3-Super Reasoner Evaluation",
        "",
        f"- Model: `{args.model}`",
        f"- API base URL: `{args.base_url}`",
        f"- Dataset: `{args.dataset_jsonl}`",
        f"- Eval split: `{args.eval_split}`",
        f"- Media mode: `{args.media_mode}`",
        f"- Samples: `{len(rows)}`",
        f"- Episodes: `{metrics['num_eval_episodes']}`",
        f"- Accuracy: `{metrics['accuracy']:.4f}`",
        f"- Macro-F1: `{metrics['macro_f1']:.4f}`",
        f"- JSON validity: `{metrics['json_validity_rate']:.4f}`",
        "",
        "This run uses the staged Cosmos3-Super Reasoner base weights through vLLM. It is an 8-GPU zero-shot/in-context evaluation, not a fine-tuned Cosmos adapter release.",
    ]
    (args.output_dir / "RUN_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

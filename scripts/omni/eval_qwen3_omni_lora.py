#!/usr/bin/env python3
"""Evaluate Qwen3-Omni LoRA action/subtask predictions on held-out episodes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from qwen3_omni_dataset_utils import (
    build_messages,
    class_metrics,
    DEFAULT_MODEL_ID,
    has_empty_audio_items,
    is_empty_audio_exception,
    json_validity_rate,
    label_counts,
    load_jsonl,
    match_label,
    parse_answer_json,
    sample_has_audio,
    sample_without_audio,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    workspace_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Evaluate a Qwen3-Omni LoRA adapter on Ropedia windows.")
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    parser.add_argument("--run-id", default="qwen_lora_eval")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--adapter-dir", type=Path, help="PEFT LoRA adapter directory from train_qwen3_omni_lora.py.")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--use-audio-in-video", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def dtype_arg(value: str):
    if value == "auto":
        return "auto"
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[value]


def load_model_processor(args: argparse.Namespace):
    from qwen3_omni_compat import patch_qwen3_omni_config

    patch_qwen3_omni_config()
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

    model_kwargs = {
        "dtype": dtype_arg(args.dtype),
        "device_map": args.device_map,
        "local_files_only": args.local_files_only,
    }
    if args.trust_remote_code:
        model_kwargs["trust_remote_code"] = True
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(args.model_id, **model_kwargs)
    if hasattr(model, "disable_talker"):
        model.disable_talker()
    if args.adapter_dir:
        from peft import PeftModel

        model.thinker = PeftModel.from_pretrained(model.thinker, args.adapter_dir)
    processor_kwargs = {"local_files_only": args.local_files_only}
    if args.trust_remote_code:
        processor_kwargs["trust_remote_code"] = True
    processor = Qwen3OmniMoeProcessor.from_pretrained(args.adapter_dir or args.model_id, **processor_kwargs)
    model.eval()
    return model, processor


def move_inputs(inputs, model):
    dtype = next(model.parameters()).dtype
    for key, value in list(inputs.items()):
        if hasattr(value, "to"):
            if getattr(value, "is_floating_point", lambda: False)():
                inputs[key] = value.to(device=model.device, dtype=dtype)
            else:
                inputs[key] = value.to(model.device)
    return inputs


def generate_one(model, processor, sample: dict, args: argparse.Namespace) -> str:
    from qwen_omni_utils import process_mm_info

    active_sample = sample
    for attempt in range(2):
        messages = build_messages(active_sample, active_sample["label_options"], include_answer=False)
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(messages, use_audio_in_video=args.use_audio_in_video)
        if attempt == 0 and sample_has_audio(active_sample) and has_empty_audio_items(audios):
            active_sample = sample_without_audio(active_sample)
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
            if attempt == 0 and sample_has_audio(active_sample) and is_empty_audio_exception(exc):
                active_sample = sample_without_audio(active_sample)
                continue
            raise
    else:
        raise RuntimeError("Unable to prepare multimodal sample after dropping empty audio.")
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


def field_accuracy(rows: list[dict], field: str) -> float | None:
    valid_rows = [row for row in rows if row["true_json"].get(field) != "unknown"]
    if not valid_rows:
        return None
    return sum(row["pred_json"].get(field) == row["true_json"].get(field) for row in valid_rows) / len(valid_rows)


def object_micro_f1(rows: list[dict]) -> float | None:
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


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = Path(__file__).resolve().parents[2] / "results" / "omni_finetune" / args.run_id
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_jsonl(args.dataset_jsonl)
    eval_samples = [sample for sample in samples if sample.get("split") == args.eval_split]
    if args.sample_limit > 0:
        eval_samples = eval_samples[: args.sample_limit]
    if not eval_samples:
        raise ValueError("No evaluation samples selected.")

    train_labels = {sample.get("answer_json", {}).get("action", sample.get("label", "unknown")) for sample in samples if sample.get("split") == args.train_split}
    eval_labels = {sample.get("answer_json", {}).get("action", sample.get("label", "unknown")) for sample in eval_samples}
    unseen_labels = sorted(eval_labels - train_labels)

    model, processor = load_model_processor(args)
    rows = []
    for sample in eval_samples:
        raw = generate_one(model, processor, sample, args)
        pred_json = parse_answer_json(raw)
        true_json = sample.get("answer_json", {})
        predicted = match_label(str(pred_json.get("action", raw)), sample.get("action_options") or sample["label_options"])
        true_action = true_json.get("action", sample.get("label", "unknown"))
        rows.append({
            "id": sample["id"],
            "target": sample["target"],
            "split": sample.get("split", "unspecified"),
            "episode_id": sample["episode_id"],
            "center_window": sample.get("center_window"),
            "true_label": true_action,
            "raw_prediction": raw,
            "true_json": true_json,
            "pred_json": pred_json,
            "predicted_label": predicted,
            "correct": int(predicted == true_action),
            "true_label_seen_in_train": int(true_action in train_labels),
        })

    metrics, per_class, cm = class_metrics(
        [row["true_label"] for row in rows],
        [row["predicted_label"] for row in rows],
        eval_samples[0]["label_options"],
    )
    seen_rows = [row for row in rows if row["true_label_seen_in_train"]]
    unseen_rows = [row for row in rows if not row["true_label_seen_in_train"]]
    metrics.update({
        "model_id": args.model_id,
        "adapter_dir": str(args.adapter_dir) if args.adapter_dir else None,
        "dataset_jsonl": str(args.dataset_jsonl),
        "eval_split": args.eval_split,
        "train_split": args.train_split,
        "num_eval_episodes": len({row["episode_id"] for row in rows}),
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
        "caption_window_grounding": {
            "mrr": None,
            "recall_at_5": None,
            "note": "Grounding ranking requires a retrieval candidate set; JSON evidence_window is stored for later scoring.",
        },
    })

    write_jsonl(args.output_dir / "predictions.jsonl", rows)
    write_csv(
        args.output_dir / "predictions.csv",
        rows,
        ["id", "target", "split", "episode_id", "center_window", "true_label", "raw_prediction", "predicted_label", "correct", "true_label_seen_in_train"],
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
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report = [
        "# Qwen3-Omni LoRA Evaluation",
        "",
        f"- Base model: `{args.model_id}`",
        f"- Adapter: `{args.adapter_dir or 'none'}`",
        f"- Dataset: `{args.dataset_jsonl}`",
        f"- Eval split: `{args.eval_split}`",
        f"- Samples: `{len(rows)}`",
        f"- Episodes: `{metrics['num_eval_episodes']}`",
        f"- Accuracy: `{metrics['accuracy']:.4f}`",
        f"- Macro-F1: `{metrics['macro_f1']:.4f}`",
        f"- Unseen eval labels: `{len(unseen_labels)}`",
        "",
        "Artifacts include `metrics.json`, `predictions.csv`, `per_class_metrics.csv`, and `confusion_matrix.csv`.",
    ]
    (args.output_dir / "RUN_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

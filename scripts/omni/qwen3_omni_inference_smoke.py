#!/usr/bin/env python3
"""Verify Qwen3-Omni can classify exported Ropedia windows before training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from qwen3_omni_dataset_utils import (
    build_messages,
    class_metrics,
    DEFAULT_MODEL_ID,
    json_validity_rate,
    load_jsonl,
    match_label,
    parse_answer_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    workspace_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Run a small Qwen3-Omni inference setup check.")
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    parser.add_argument("--run-id", default="qwen_zero_shot")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--sample-limit", type=int, default=3)
    parser.add_argument("--split", default="all", choices=["all", "train", "val", "test", "unspecified"])
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--use-audio-in-video", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--disable-talker", action="store_true", default=True)
    return parser.parse_args()


def dtype_arg(value: str):
    if value == "auto":
        return "auto"
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[value]


def load_model_and_processor(args: argparse.Namespace):
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
    if args.disable_talker and hasattr(model, "disable_talker"):
        model.disable_talker()
    processor_kwargs = {"local_files_only": args.local_files_only}
    if args.trust_remote_code:
        processor_kwargs["trust_remote_code"] = True
    processor = Qwen3OmniMoeProcessor.from_pretrained(args.model_id, **processor_kwargs)
    return model, processor


def move_inputs(inputs, model):
    device = model.device
    dtype = next(model.parameters()).dtype
    for key, value in list(inputs.items()):
        if hasattr(value, "to"):
            if getattr(value, "is_floating_point", lambda: False)():
                inputs[key] = value.to(device=device, dtype=dtype)
            else:
                inputs[key] = value.to(device)
    return inputs


def generate_one(model, processor, sample: dict, args: argparse.Namespace) -> str:
    from qwen_omni_utils import process_mm_info

    messages = build_messages(sample, sample["label_options"], include_answer=False)
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(messages, use_audio_in_video=args.use_audio_in_video)
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=args.use_audio_in_video,
    )
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
    if hasattr(text_ids, "sequences"):
        sequences = text_ids.sequences
    else:
        sequences = text_ids
    output_ids = sequences[:, inputs["input_ids"].shape[1] :]
    decoded = processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return decoded[0] if decoded else ""


def main() -> int:
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = Path(__file__).resolve().parents[2] / "results" / "omni_finetune" / args.run_id / "qwen_zero_shot"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_jsonl(args.dataset_jsonl)
    if args.split != "all":
        samples = [sample for sample in samples if sample.get("split") == args.split]
    if args.sample_limit > 0:
        samples = samples[: args.sample_limit]
    if not samples:
        raise ValueError("No samples selected for the inference setup check.")

    model, processor = load_model_and_processor(args)
    rows = []
    for sample in samples:
        raw = generate_one(model, processor, sample, args)
        payload = parse_answer_json(raw)
        predicted_action = match_label(str(payload.get("action", raw)), sample.get("action_options") or sample["label_options"])
        true_action = sample.get("answer_json", {}).get("action", sample.get("label", "unknown"))
        rows.append({
            "id": sample["id"],
            "target": sample["target"],
            "split": sample.get("split", "unspecified"),
            "episode_id": sample["episode_id"],
            "center_window": sample.get("center_window"),
            "true_label": true_action,
            "raw_prediction": raw,
            "predicted_label": predicted_action,
            "parsed_prediction": payload,
            "correct": predicted_action == true_action,
        })

    metrics, per_class, cm = class_metrics(
        [row["true_label"] for row in rows],
        [row["predicted_label"] for row in rows],
        samples[0]["label_options"],
    )
    metrics.update({
        "model_id": args.model_id,
        "dataset_jsonl": str(args.dataset_jsonl),
        "qwen3_loaded": True,
        "num_requested_samples": len(samples),
        "json_validity_rate": json_validity_rate([row["raw_prediction"] for row in rows]),
    })
    write_jsonl(args.output_dir / "predictions.jsonl", rows)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.output_dir / "per_class_metrics.json").write_text(json.dumps(per_class, indent=2), encoding="utf-8")
    (args.output_dir / "confusion_matrix.json").write_text(json.dumps(cm, indent=2), encoding="utf-8")
    report = [
        "# Qwen3-Omni Inference Setup Check",
        "",
        f"- Model: `{args.model_id}`",
        f"- Dataset: `{args.dataset_jsonl}`",
        f"- Samples: `{len(rows)}`",
        f"- Accuracy: `{metrics['accuracy']:.4f}`",
        f"- Macro-F1: `{metrics['macro_f1']:.4f}`",
        "",
        "This is a pre-training setup check. It verifies that exported Ropedia video/audio/text prompts can pass through Qwen3-Omni on the target runtime.",
    ]
    (args.output_dir / "RUN_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

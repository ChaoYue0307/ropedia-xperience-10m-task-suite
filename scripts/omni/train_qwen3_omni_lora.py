#!/usr/bin/env python3
"""Conservative LoRA SFT for Qwen3-Omni action/subtask label generation."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from types import MethodType

import torch

from qwen3_omni_dataset_utils import build_messages, DEFAULT_MODEL_ID, load_jsonl


def parse_args() -> argparse.Namespace:
    workspace_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Train Qwen3-Omni LoRA on exported Ropedia windows.")
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    parser.add_argument("--run-id", default="qwen_lora_text_video_audio")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--include-unspecified-in-train", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--use-audio-in-video", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated module names passed to PEFT LoRAConfig.",
    )
    return parser.parse_args()


def dtype_arg(value: str):
    if value == "auto":
        return "auto"
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[value]


def select_samples(samples: list[dict], split: str, include_unspecified: bool) -> list[dict]:
    rows = [sample for sample in samples if sample.get("split") == split]
    if include_unspecified:
        rows.extend(sample for sample in samples if sample.get("split") == "unspecified")
    return rows


def patch_rotary_position_device(model) -> bool:
    """Keep Qwen3-Omni rotary position ids aligned under model-parallel device maps."""
    inner_model = getattr(model, "model", None)
    rotary = getattr(inner_model, "rotary_emb", None)
    if rotary is None or getattr(rotary, "_ropedia_position_device_patch", False):
        return False

    original_forward = rotary.forward

    def forward_with_aligned_position_ids(self, x, position_ids, *args, **kwargs):
        if hasattr(self, "inv_freq") and hasattr(x, "device") and self.inv_freq.device != x.device:
            self._buffers["inv_freq"] = self.inv_freq.to(x.device)
        if hasattr(position_ids, "to") and hasattr(x, "device") and position_ids.device != x.device:
            position_ids = position_ids.to(x.device)
        return original_forward(x, position_ids, *args, **kwargs)

    rotary.forward = MethodType(forward_with_aligned_position_ids, rotary)
    rotary._ropedia_position_device_patch = True
    return True


def patch_qwen3_omni_rotary_classes() -> None:
    """Patch Qwen3-Omni MRoPE classes before Accelerate installs device hooks."""
    from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as qwen3_omni_moe

    def patch_mrope_class(class_name: str) -> None:
        rotary_cls = getattr(qwen3_omni_moe, class_name, None)
        if rotary_cls is None or getattr(rotary_cls, "_ropedia_class_device_patch", False):
            return

        @torch.no_grad()
        def forward(self, x, position_ids):
            if position_ids.ndim == 2:
                position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
            target_device = x.device
            inv_freq = self.inv_freq.to(target_device)
            position_ids = position_ids.to(target_device)
            inv_freq_expanded = inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
            position_ids_expanded = position_ids[:, :, None, :].float()

            device_type = target_device.type if isinstance(target_device.type, str) and target_device.type != "mps" else "cpu"
            with qwen3_omni_moe.maybe_autocast(device_type=device_type, enabled=False):
                freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
                freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
                emb = torch.cat((freqs, freqs), dim=-1)
                cos = emb.cos() * self.attention_scaling
                sin = emb.sin() * self.attention_scaling

            return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

        rotary_cls.forward = forward
        rotary_cls._ropedia_class_device_patch = True

    patch_mrope_class("Qwen3OmniMoeThinkerTextRotaryEmbedding")
    patch_mrope_class("Qwen3OmniMoeTalkerRotaryEmbedding")


def patch_qwen3_omni_norm_classes() -> None:
    """Patch Qwen3-Omni RMSNorm classes for model-parallel device maps."""
    from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as qwen3_omni_moe

    def patch_norm_class(class_name: str) -> None:
        norm_cls = getattr(qwen3_omni_moe, class_name, None)
        if norm_cls is None or getattr(norm_cls, "_ropedia_class_device_patch", False):
            return

        def forward(self, hidden_states):
            input_dtype = hidden_states.dtype
            norm_states = hidden_states.to(torch.float32)
            variance = norm_states.pow(2).mean(-1, keepdim=True)
            norm_states = norm_states * torch.rsqrt(variance + self.variance_epsilon)
            weight = self.weight.to(hidden_states.device)
            return weight * norm_states.to(input_dtype)

        norm_cls.forward = forward
        norm_cls._ropedia_class_device_patch = True

    patch_norm_class("Qwen3OmniMoeRMSNorm")
    patch_norm_class("Qwen3OmniMoeTextRMSNorm")
    patch_norm_class("Qwen3OmniMoeThinkerTextRMSNorm")
    patch_norm_class("Qwen3OmniMoeCode2WavRMSNorm")


def cast_floating_parameters(model, target_dtype) -> None:
    if isinstance(target_dtype, str):
        return
    for param in model.parameters():
        if param.is_floating_point() and param.dtype != target_dtype:
            param.data = param.data.to(target_dtype)


def build_trainable_cpu_state_dict(model) -> dict[str, torch.Tensor]:
    state_dict = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        clean_name = name
        if clean_name.startswith("module."):
            clean_name = clean_name[len("module.") :]
        state_dict[clean_name] = param.detach().to("cpu", copy=True)
    return state_dict


def load_model_processor(args: argparse.Namespace):
    from qwen3_omni_compat import patch_qwen3_omni_config

    patch_qwen3_omni_config()
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

    patch_qwen3_omni_rotary_classes()
    patch_qwen3_omni_norm_classes()

    model_kwargs = {
        "dtype": dtype_arg(args.dtype),
        "local_files_only": args.local_files_only,
    }
    if args.device_map and args.device_map.lower() != "none":
        model_kwargs["device_map"] = args.device_map
    if args.trust_remote_code:
        model_kwargs["trust_remote_code"] = True
    omni_model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(args.model_id, **model_kwargs)
    if hasattr(omni_model, "disable_talker"):
        omni_model.disable_talker()
    model = omni_model.thinker
    if args.device_map and args.device_map.lower() != "none":
        patch_rotary_position_device(model)
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    processor_kwargs = {"local_files_only": args.local_files_only}
    if args.trust_remote_code:
        processor_kwargs["trust_remote_code"] = True
    processor = Qwen3OmniMoeProcessor.from_pretrained(args.model_id, **processor_kwargs)

    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=[item.strip() for item in args.lora_target_modules.split(",") if item.strip()],
    )
    model = get_peft_model(model, config)
    cast_floating_parameters(model, dtype_arg(args.dtype))
    model.print_trainable_parameters()
    return model, processor


def move_inputs(inputs, device, dtype=None):
    for key, value in list(inputs.items()):
        if hasattr(value, "to"):
            if dtype is not None and getattr(value, "is_floating_point", lambda: False)():
                inputs[key] = value.to(device=device, dtype=dtype)
            else:
                inputs[key] = value.to(device)
    return inputs


def prepare_sample(processor, sample: dict, use_audio_in_video: bool, device, dtype=None) -> dict:
    from qwen_omni_utils import process_mm_info

    full_messages = build_messages(sample, sample["label_options"], include_answer=True)
    prompt_messages = build_messages(sample, sample["label_options"], include_answer=False)
    full_text = processor.apply_chat_template(full_messages, tokenize=False)
    prompt_text = processor.apply_chat_template(prompt_messages, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(full_messages, use_audio_in_video=use_audio_in_video)
    inputs = processor(
        text=full_text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=use_audio_in_video,
    )
    labels = inputs["input_ids"].clone()
    prompt_ids = processor.tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    prompt_len = min(prompt_ids.shape[1], labels.shape[1])
    labels[:, :prompt_len] = -100
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is not None:
        labels[inputs["input_ids"] == pad_id] = -100
    inputs["labels"] = labels
    return move_inputs(inputs, device, dtype=dtype)


def write_progress(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def distributed_slice(samples: list[dict], process_index: int, num_processes: int) -> list[dict]:
    if num_processes <= 1:
        return list(samples)
    shard = list(samples[process_index::num_processes])
    max_len = math.ceil(len(samples) / num_processes)
    if not samples:
        return []
    if not shard:
        shard = [samples[process_index % len(samples)]]
    while len(shard) < max_len:
        shard.append(random.choice(shard))
    return shard


def evaluate_loss(model, processor, samples: list[dict], args: argparse.Namespace, device, dtype=None, accelerator=None) -> float | None:
    if not samples:
        return None
    losses = []
    model.eval()
    with torch.no_grad():
        for sample in samples:
            inputs = prepare_sample(processor, sample, args.use_audio_in_video, device, dtype=dtype)
            output = model(**inputs)
            losses.append(float(output.loss.detach().cpu()))
    model.train()
    local = torch.tensor([sum(losses), len(losses)], dtype=torch.float32, device=device)
    if accelerator is not None:
        gathered = accelerator.gather(local)
        total_loss = float(gathered[0::2].sum().detach().cpu())
        total_count = float(gathered[1::2].sum().detach().cpu())
        return total_loss / total_count if total_count else None
    return sum(losses) / len(losses) if losses else None


def main() -> int:
    args = parse_args()
    from accelerate import Accelerator

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    workspace_default = Path(__file__).resolve().parents[2]
    if args.output_dir is None:
        args.output_dir = workspace_default / "checkpoints" / args.run_id / "adapter_lora"
    if args.results_dir is None:
        args.results_dir = workspace_default / "results" / "omni_finetune" / args.run_id
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.results_dir / "progress.jsonl"
    if accelerator.is_main_process and progress_path.exists():
        progress_path.unlink()
    torch.manual_seed(args.seed + accelerator.process_index)
    random.seed(args.seed + accelerator.process_index)

    samples = load_jsonl(args.dataset_jsonl)
    train_samples = select_samples(samples, args.train_split, args.include_unspecified_in_train)
    val_samples = [sample for sample in samples if sample.get("split") == args.val_split]
    if args.max_train_samples > 0:
        train_samples = train_samples[: args.max_train_samples]
    if args.max_val_samples > 0:
        val_samples = val_samples[: args.max_val_samples]
    if not train_samples:
        raise ValueError("No training samples selected. Check --train-split or use --include-unspecified-in-train.")
    rank_train_samples = distributed_slice(train_samples, accelerator.process_index, accelerator.num_processes)
    rank_val_samples = distributed_slice(val_samples, accelerator.process_index, accelerator.num_processes) if val_samples else []

    if accelerator.is_main_process:
        write_progress(progress_path, {
            "event": "setup_done",
            "run_id": args.run_id,
            "dataset_jsonl": str(args.dataset_jsonl),
            "num_processes": accelerator.num_processes,
            "num_train_samples": len(train_samples),
            "num_val_samples": len(val_samples),
            "rank0_samples_per_epoch": len(rank_train_samples),
            "timestamp": time.time(),
        })
    if accelerator.num_processes > 1 and args.device_map == "auto":
        args.device_map = "none"
    if accelerator.is_main_process:
        write_progress(progress_path, {
            "event": "model_load_start",
            "run_id": args.run_id,
            "model_id": args.model_id,
            "device_map": args.device_map,
            "dtype": args.dtype,
            "timestamp": time.time(),
        })
    model, processor = load_model_processor(args)
    if accelerator.is_main_process:
        write_progress(progress_path, {
            "event": "model_load_done",
            "run_id": args.run_id,
            "timestamp": time.time(),
        })
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate, weight_decay=args.weight_decay)
    if accelerator.is_main_process:
        write_progress(progress_path, {
            "event": "accelerator_prepare_start",
            "run_id": args.run_id,
            "timestamp": time.time(),
        })
    model, optimizer = accelerator.prepare(model, optimizer)
    if accelerator.is_main_process:
        write_progress(progress_path, {
            "event": "accelerator_prepare_done",
            "run_id": args.run_id,
            "timestamp": time.time(),
        })
    device = accelerator.device
    model_dtype = next(model.parameters()).dtype

    history = []
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    model.train()
    if accelerator.is_main_process:
        write_progress(progress_path, {
            "event": "train_loop_start",
            "run_id": args.run_id,
            "model_id": args.model_id,
            "dataset_jsonl": str(args.dataset_jsonl),
            "num_processes": accelerator.num_processes,
            "num_train_samples": len(train_samples),
            "num_val_samples": len(val_samples),
            "rank_samples_per_epoch": len(rank_train_samples),
            "epochs": args.epochs,
            "timestamp": time.time(),
        })
    for epoch in range(1, args.epochs + 1):
        random.shuffle(rank_train_samples)
        epoch_loss = 0.0
        seen = 0
        steps_in_epoch = math.ceil(len(rank_train_samples) / max(args.batch_size, 1))
        for batch_start in range(0, len(rank_train_samples), args.batch_size):
            batch = rank_train_samples[batch_start : batch_start + args.batch_size]
            batch_loss = 0.0
            for sample in batch:
                with accelerator.accumulate(model):
                    inputs = prepare_sample(processor, sample, args.use_audio_in_video, device, dtype=model_dtype)
                    output = model(**inputs)
                    accelerator.backward(output.loss)
                    batch_loss += float(output.loss.detach().cpu())
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            seen += len(batch)
            epoch_loss += batch_loss
            global_step += 1
            if accelerator.is_main_process and (global_step % args.progress_every == 0 or batch_start // max(args.batch_size, 1) == steps_in_epoch - 1):
                write_progress(progress_path, {
                    "event": "train_step",
                    "epoch": epoch,
                    "global_step": global_step,
                    "rank0_seen": seen,
                    "rank0_samples_per_epoch": len(rank_train_samples),
                    "rank0_batch_loss": batch_loss / max(len(batch), 1),
                    "timestamp": time.time(),
                })
        val_loss = evaluate_loss(model, processor, rank_val_samples, args, device, dtype=model_dtype, accelerator=accelerator)
        epoch_row = {
            "epoch": epoch,
            "train_loss": epoch_loss / max(len(rank_train_samples), 1),
            "val_loss": val_loss,
            "global_step": global_step,
        }
        history.append(epoch_row)
        if accelerator.is_main_process:
            print(json.dumps(epoch_row, indent=2))
            write_progress(progress_path, {"event": "epoch_end", **epoch_row, "timestamp": time.time()})

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        write_progress(progress_path, {
            "event": "save_start",
            "checkpoint_dir": str(args.output_dir),
            "save_mode": "trainable_lora_state_dict",
            "timestamp": time.time(),
        })
    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        adapter_state = build_trainable_cpu_state_dict(unwrapped)
        write_progress(progress_path, {
            "event": "save_state_dict_built",
            "checkpoint_dir": str(args.output_dir),
            "trainable_tensors": len(adapter_state),
            "trainable_bytes": sum(t.numel() * t.element_size() for t in adapter_state.values()),
            "timestamp": time.time(),
        })
        unwrapped.save_pretrained(args.output_dir, state_dict=adapter_state, is_main_process=True)
        processor.save_pretrained(args.output_dir)
        write_progress(progress_path, {
            "event": "save_done",
            "checkpoint_dir": str(args.output_dir),
            "timestamp": time.time(),
        })
    metadata = {
        "run_id": args.run_id,
        "model_id": args.model_id,
        "dataset_jsonl": str(args.dataset_jsonl),
        "checkpoint_dir": str(args.output_dir),
        "num_processes": accelerator.num_processes,
        "num_train_samples": len(train_samples),
        "num_val_samples": len(val_samples),
        "history": history,
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": [item.strip() for item in args.lora_target_modules.split(",") if item.strip()],
        },
        "use_audio_in_video": args.use_audio_in_video,
    }
    if accelerator.is_main_process:
        (args.output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        (args.results_dir / "config.yaml").write_text(
            "\n".join([
                f"run_id: {args.run_id}",
                "stage: qwen_lora_text_video_audio",
                f"model_id: {args.model_id}",
                f"dataset_jsonl: {args.dataset_jsonl}",
                f"checkpoint_dir: {args.output_dir}",
                f"num_processes: {accelerator.num_processes}",
                f"epochs: {args.epochs}",
                f"learning_rate: {args.learning_rate}",
                f"lora_r: {args.lora_r}",
                f"lora_alpha: {args.lora_alpha}",
            ]) + "\n",
            encoding="utf-8",
        )
        (args.results_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        report = [
            "# Qwen3-Omni LoRA Training",
            "",
            f"- Base model: `{args.model_id}`",
            f"- Dataset: `{args.dataset_jsonl}`",
            f"- Train samples: `{len(train_samples)}`",
            f"- Validation samples: `{len(val_samples)}`",
            f"- Processes: `{accelerator.num_processes}`",
            f"- Epochs: `{args.epochs}`",
            f"- Final train loss: `{history[-1]['train_loss']:.6f}`",
            "",
            "Only LoRA parameters are trained; the base Qwen3-Omni weights remain frozen.",
        ]
        if history[-1]["val_loss"] is not None:
            report.append(f"- Final val loss: `{history[-1]['val_loss']:.6f}`")
        (args.results_dir / "RUN_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        write_progress(progress_path, {"event": "complete", "checkpoint_dir": str(args.output_dir), "timestamp": time.time()})
        print(f"Wrote LoRA adapter to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

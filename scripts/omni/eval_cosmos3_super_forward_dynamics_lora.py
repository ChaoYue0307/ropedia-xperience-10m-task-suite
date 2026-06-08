#!/usr/bin/env python3
"""Evaluate Cosmos3-Super forward-dynamics LoRA loss on held-out rows."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

from qwen3_omni_dataset_utils import load_jsonl
from train_cosmos3_super_forward_dynamics_lora import (
    DEFAULT_DATASET,
    append_jsonl,
    dtype_from_name,
    instantiate_action,
    lora_targets,
    model_inner_config,
    read_json,
    select_rows,
    training_step,
    write_json,
)


def parse_args() -> argparse.Namespace:
    workspace_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=workspace_default)
    parser.add_argument("--dataset-jsonl", type=Path, default=workspace_default / DEFAULT_DATASET)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--adapter-name", default="xperience_forward_dynamics")
    parser.add_argument("--adapter-weight-name", default="pytorch_lora_weights.safetensors")
    parser.add_argument("--adapter-prefix", default="none", help="Use 'none' for prefix=None when loading custom Cosmos LoRA.")
    parser.add_argument("--run-id", default="xperience10m_cosmos3_super_forward_dynamics_lora_eval")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--episode-id")
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--eval-num-shards", type=int, default=1)
    parser.add_argument("--eval-shard-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default="balanced", help="Use 'none' to load the full pipeline onto --device.")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--prompt", default="Predict the embodied future under the provided camera-pose action condition.")
    parser.add_argument("--negative-prompt")
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--num-train-timesteps", type=int)
    parser.add_argument("--timestep-sampling", default="uniform", choices=["uniform", "logitnormal"])
    parser.add_argument("--resolution-shift", type=float)
    parser.add_argument("--override-resolution-tier", type=int, choices=[256, 480, 704, 720])
    parser.add_argument("--loss-scale", type=float)
    parser.add_argument("--require-media-exists", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def shape_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    shim = argparse.Namespace(split=args.split, episode_id=args.episode_id, max_train_samples=args.max_eval_samples)
    return select_rows(rows, shim)


def summarize(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "min": None, "max": None}
    return {
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


def shard_rows(rows: list[dict[str, Any]], num_shards: int, shard_index: int) -> list[dict[str, Any]]:
    if num_shards <= 1:
        return rows
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"eval_shard_index must be in [0, {num_shards}), got {shard_index}")
    return rows[shard_index::num_shards]


def adapter_parameter_audit(transformer: Any, adapter_name: str) -> dict[str, Any]:
    params = []
    marker = f".{adapter_name}."
    for name, param in transformer.named_parameters():
        if ".lora_A." not in name and ".lora_B." not in name:
            continue
        if marker not in name:
            continue
        params.append((name, param))
    meta_params = [name for name, param in params if getattr(param, "is_meta", False)]
    audit = {
        "adapter_name": adapter_name,
        "parameter_count": len(params),
        "meta_parameter_count": len(meta_params),
        "meta_parameters_sample": meta_params[:16],
        "trainable_parameter_numel": sum(param.numel() for _name, param in params if param.requires_grad),
        "parameter_numel": sum(param.numel() for _name, param in params),
        "sample_shapes": {name: list(param.shape) for name, param in params[:8]},
    }
    if not params:
        raise RuntimeError(f"no LoRA parameters loaded for adapter {adapter_name!r}")
    if meta_params:
        raise RuntimeError(f"LoRA adapter has {len(meta_params)} meta parameters; use --device-map none for eval")
    return audit


def write_report(output_dir: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Cosmos3-Super Forward-Dynamics LoRA Evaluation",
        "",
        f"- Run id: `{payload['run_id']}`",
        f"- Status: `{payload['status']}`",
        f"- Split: `{payload['split']}`",
        f"- Eval samples: `{payload['num_eval_samples']}`",
        f"- Mean loss: `{payload['loss_summary']['mean']}`",
        f"- Adapter dir: `{payload['adapter_dir']}`",
        "",
        "The metric is rectified-flow vision velocity MSE under camera-pose action conditioning. It does not evaluate semantic JSON action labels.",
    ]
    (output_dir / "RUN_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    from accelerate import Accelerator

    accelerator = Accelerator()
    args.workspace = args.workspace.expanduser().resolve()
    args.dataset_jsonl = args.dataset_jsonl.expanduser().resolve()
    args.model_dir = args.model_dir.expanduser().resolve()
    args.adapter_dir = args.adapter_dir.expanduser().resolve()
    output_dir = args.output_dir or args.workspace / "results" / "omni_finetune" / args.run_id
    output_dir = output_dir.expanduser().resolve()
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        for old_progress in output_dir.glob("progress*.jsonl"):
            old_progress.unlink()
    accelerator.wait_for_everyone()
    progress_path = output_dir / (
        f"progress_rank{accelerator.process_index}.jsonl" if accelerator.num_processes > 1 else "progress.jsonl"
    )
    started = time.time()
    append_jsonl(
        progress_path,
        {
            "event": "eval_start",
            "run_id": args.run_id,
            "timestamp": started,
            "num_processes": accelerator.num_processes,
            "process_index": accelerator.process_index,
        },
    )

    random.seed(args.seed + accelerator.process_index)
    rows = shard_rows(shape_rows(load_jsonl(args.dataset_jsonl), args), args.eval_num_shards, args.eval_shard_index)
    if accelerator.num_processes > 1:
        rows = rows[accelerator.process_index :: accelerator.num_processes]
    inner = model_inner_config(args.model_dir)
    target_modules = lora_targets(argparse.Namespace(target_modules=None), inner, args.model_dir)
    train_cfg = inner.get("rectified_flow_training_config") or {}
    num_train_timesteps = int(args.num_train_timesteps or ((inner.get("rectified_flow_inference_config") or {}).get("num_train_timesteps") or 1000))
    shift_table = train_cfg.get("shift") if isinstance(train_cfg.get("shift"), dict) else {}
    resolution_key = str(args.override_resolution_tier or 480)
    sigma_shift = float(args.resolution_shift or shift_table.get(resolution_key) or 1.0)
    loss_scale = args.loss_scale if args.loss_scale is not None else train_cfg.get("loss_scale")
    args.loss_scale = float(loss_scale) if loss_scale is not None else None

    import torch
    from diffusers import Cosmos3OmniPipeline

    torch.set_grad_enabled(False)
    dtype = dtype_from_name(args.dtype)
    load_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "local_files_only": args.local_files_only,
        "enable_safety_checker": False,
    }
    if accelerator.num_processes > 1 and args.device_map != "none":
        args.device_map = "none"
    if args.device_map != "none":
        load_kwargs["device_map"] = args.device_map
    pipe = Cosmos3OmniPipeline.from_pretrained(str(args.model_dir), **load_kwargs)
    if accelerator.num_processes > 1:
        device = accelerator.device
        for component_name in ("vae", "sound_tokenizer"):
            component = getattr(pipe, component_name, None)
            if component is not None:
                component.to(device)
    elif args.device_map == "none":
        pipe.to(args.device)
        device = args.device
    else:
        device = str(pipe._get_execution_device())
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)

    prefix = None if args.adapter_prefix.lower() == "none" else args.adapter_prefix
    pipe.transformer.load_lora_adapter(
        str(args.adapter_dir),
        weight_name=args.adapter_weight_name,
        adapter_name=args.adapter_name,
        prefix=prefix,
    )
    pipe.transformer.set_adapter(args.adapter_name)
    pipe.transformer.eval()
    adapter_audit = adapter_parameter_audit(pipe.transformer, args.adapter_name)
    if args.device_map == "none":
        pipe.transformer.to(dtype=dtype)
    if accelerator.num_processes > 1:
        pipe.transformer = accelerator.prepare(pipe.transformer)

    append_jsonl(
        progress_path,
        {
            "event": "model_ready",
            "timestamp": time.time(),
            "device": str(device),
            "device_map": args.device_map,
            "num_eval_samples": len(rows),
            "target_modules": target_modules,
            "adapter_dir": str(args.adapter_dir),
            "adapter_audit": adapter_audit,
        },
    )

    losses: list[float] = []
    examples: list[dict[str, Any]] = []
    status = "complete"
    try:
        for index, row in enumerate(rows, start=1):
            loss, info = training_step(pipe, row, args, device, dtype, num_train_timesteps, sigma_shift, grad_enabled=False)
            loss_value = float(loss.detach().float().cpu())
            losses.append(loss_value)
            if len(examples) < 16:
                examples.append({"index": index, "loss": loss_value, **info})
            if index == 1 or index % args.progress_every == 0 or index == len(rows):
                append_jsonl(progress_path, {"event": "eval_step", "timestamp": time.time(), "index": index, "loss": loss_value, **info})
    except Exception as exc:
        status = "failed"
        append_jsonl(progress_path, {"event": "failed", "timestamp": time.time(), "error": repr(exc)})
        raise
    finally:
        finished = time.time()
        payload = {
            "run_id": args.run_id,
            "run_kind": "cosmos3_super_forward_dynamics_lora_eval",
            "status": status,
            "process_index": accelerator.process_index,
            "started_at_unix": started,
            "finished_at_unix": finished,
            "elapsed_seconds": finished - started,
            "dataset_jsonl": str(args.dataset_jsonl),
            "model_dir": str(args.model_dir),
            "adapter_dir": str(args.adapter_dir),
            "adapter_weight_name": args.adapter_weight_name,
            "adapter_prefix": args.adapter_prefix,
            "split": args.split,
            "episode_id": args.episode_id,
            "eval_num_shards": args.eval_num_shards,
            "eval_shard_index": args.eval_shard_index,
            "num_eval_samples": len(rows),
            "num_train_timesteps": num_train_timesteps,
            "sigma_shift": sigma_shift,
            "loss_scale": args.loss_scale,
            "loss_summary": summarize(losses),
            "losses": losses,
            "examples": examples,
            "adapter_audit": adapter_audit if "adapter_audit" in locals() else None,
            "loss_surface": "vision_velocity_conditioned_on_camera_pose",
            "action_loss_expected": False,
        }
        rank_metrics_path = output_dir / f"rank_{accelerator.process_index}_metrics.json"
        write_json(rank_metrics_path, payload)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            rank_payloads = []
            for path in sorted(output_dir.glob("rank_*_metrics.json")):
                rank_payloads.append(json.loads(path.read_text(encoding="utf-8")))
            all_losses: list[float] = []
            all_examples: list[dict[str, Any]] = []
            for rank_payload in rank_payloads:
                all_losses.extend(float(loss) for loss in rank_payload.get("losses", []))
                all_examples.extend(rank_payload.get("examples", []))
            aggregate = dict(payload)
            aggregate.update(
                {
                    "process_index": None,
                    "num_eval_samples": len(all_losses),
                    "loss_summary": summarize(all_losses),
                    "losses": all_losses,
                    "examples": all_examples[:32],
                    "rank_metrics": rank_payloads,
                    "status": "complete"
                    if all(rank_payload.get("status") == "complete" for rank_payload in rank_payloads)
                    else "failed",
                }
            )
            write_json(output_dir / "metrics.json", aggregate)
            write_report(output_dir, aggregate)
        append_jsonl(progress_path, {"event": "complete", "timestamp": time.time(), "status": status})
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        aggregate = read_json(output_dir / "metrics.json")
        print(json.dumps({"status": aggregate.get("status"), "output_dir": str(output_dir), "mean_loss": aggregate.get("loss_summary", {}).get("mean")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

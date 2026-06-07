#!/usr/bin/env python3
"""LoRA overfit trainer for Cosmos3-Super camera-pose forward dynamics.

This trains the first real Cosmos3-Super adapter path for Xperience-10M. The
current camera-pose targets are forward-dynamics targets: raw camera-pose
actions are conditioning, and the supervised loss is the future vision velocity
under rectified-flow noise. This script therefore updates LoRA weights on the
Cosmos3 transformer and does not claim supervised action-token prediction.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

from pack_cosmos3_super_action_batch import (
    find_action_target,
    media_video_path,
    row_contract,
    tokenize_prompt,
)
from qwen3_omni_dataset_utils import load_jsonl


DEFAULT_DATASET = (
    "results/omni_finetune/"
    "xperience10m_cosmos3_camera_pose_targets_20260608/"
    "dataset_with_cosmos_actions.jsonl"
)


def parse_args() -> argparse.Namespace:
    workspace_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=workspace_default)
    parser.add_argument("--dataset-jsonl", type=Path, default=workspace_default / DEFAULT_DATASET)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--run-id", default="xperience10m_cosmos3_super_forward_dynamics_lora_overfit")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--episode-id", help="Optional single episode to overfit before scaling.")
    parser.add_argument("--max-train-samples", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lora-rank", type=int)
    parser.add_argument("--lora-alpha", type=int)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--adapter-name", default="xperience_forward_dynamics")
    parser.add_argument("--target-modules", help="Comma-separated LoRA target modules. Defaults to model config.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default="balanced", help="Use 'none' to load the full pipeline onto --device.")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--prompt", default="Predict the embodied future under the provided camera-pose action condition.")
    parser.add_argument("--negative-prompt")
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--num-train-timesteps", type=int)
    parser.add_argument("--timestep-sampling", default="uniform", choices=["uniform", "logitnormal"])
    parser.add_argument("--resolution-shift", type=float, help="Override rectified-flow sigma shift.")
    parser.add_argument("--override-resolution-tier", type=int, choices=[256, 480, 704, 720])
    parser.add_argument("--loss-scale", type=float)
    parser.add_argument("--require-media-exists", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Pack batches but do not update weights.")
    return parser.parse_args()


def dtype_from_name(name: str):
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def model_inner_config(model_dir: Path) -> dict[str, Any]:
    config = read_json(model_dir / "config.json")
    return ((config.get("model") or {}).get("config") or {}) if config else {}


def select_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    candidates = []
    for row in rows:
        if row.get("split") != args.split:
            continue
        if args.episode_id and row.get("episode_id") != args.episode_id:
            continue
        if find_action_target(row)[1] is None:
            continue
        candidates.append(row)
    if args.max_train_samples > 0:
        candidates = candidates[: args.max_train_samples]
    if not candidates:
        raise ValueError(f"no Cosmos action-target rows found for split={args.split!r}")
    return candidates


def lora_targets(args: argparse.Namespace, inner: dict[str, Any]) -> list[str]:
    raw = args.target_modules or inner.get("lora_target_modules") or ""
    modules = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not modules:
        modules = ["q_proj_moe_gen", "k_proj_moe_gen", "v_proj_moe_gen", "o_proj_moe_gen"]
    return modules


def instantiate_action(row: dict[str, Any], resolution_tier: int | None):
    import torch
    from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import CosmosActionCondition

    _, target = find_action_target(row)
    if target is None:
        raise ValueError(f"row has no Cosmos action target: {row.get('id')}")
    raw_actions = target.get("raw_actions")
    raw_actions_tensor = torch.tensor(raw_actions, dtype=torch.float32) if raw_actions is not None else None
    video_path = media_video_path(row, target)
    if not video_path:
        raise ValueError(f"row has no video path for Cosmos action target: {row.get('id')}")
    return CosmosActionCondition(
        mode=str(target.get("mode")),
        chunk_size=int(target.get("chunk_size")),
        domain_name=str(target.get("domain_name")),
        resolution_tier=int(resolution_tier or target.get("resolution_tier", 480)),
        raw_actions=raw_actions_tensor,
        video=[video_path],
        view_point=str(target.get("view_point", "ego_view")),
    )


def action_domain_id(domain_name: str, device: str):
    import torch
    from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import _EMBODIMENT_TO_DOMAIN_ID

    if domain_name not in _EMBODIMENT_TO_DOMAIN_ID:
        raise ValueError(f"unknown Cosmos3 action domain: {domain_name}")
    return torch.tensor([_EMBODIMENT_TO_DOMAIN_ID[domain_name]], dtype=torch.long, device=device)


def clean_vision_latents(pipe: Any, action: Any, device: str, dtype: Any):
    target_frames = action.chunk_size + 1
    conditioning_clip = [action.image] if action.image is not None else action.video
    vision_tensor, action_image_size, height, width = pipe._prepare_action_video_conditioning(
        conditioning_clip,
        action.resolution_tier,
        target_frames,
        device=device,
        dtype=dtype,
    )
    x0 = pipe._encode_video(vision_tensor).contiguous().float()
    if action_image_size is not None:
        x0 = pipe._remove_action_video_padding_from_latent(x0, action_image_size)
    return x0, action_image_size, height, width


def action_latents(action: Any, pipe: Any, device: str, dtype: Any):
    import torch

    raw = action.raw_actions
    if raw is None:
        raise ValueError("forward_dynamics requires raw action targets")
    raw = raw.to(device=device, dtype=dtype)
    action_chunk_size = int(action.chunk_size)
    if raw.shape[0] < action_chunk_size:
        raw = torch.cat([raw, raw[-1:].expand(action_chunk_size - raw.shape[0], -1)], dim=0)
    raw = raw[:action_chunk_size]
    raw_action_dim = int(raw.shape[-1])
    action_dim = int(pipe.transformer.action_dim)
    if raw_action_dim > action_dim:
        raise ValueError(f"raw action dim {raw_action_dim} exceeds model action_dim {action_dim}")
    if raw_action_dim < action_dim:
        pad = torch.zeros(raw.shape[0], action_dim - raw_action_dim, device=device, dtype=dtype)
        raw = torch.cat([raw, pad], dim=-1)
    condition_mask = torch.ones((action_chunk_size, 1), device=device, dtype=dtype)
    return raw, condition_mask, raw_action_dim, list(range(action_chunk_size))


def shifted_sigma(timestep: int, num_train_timesteps: int, shift: float) -> float:
    sigma = max(1e-5, min(1.0, float(timestep) / float(num_train_timesteps)))
    if shift and shift != 1.0:
        sigma = shift * sigma / (1.0 + (shift - 1.0) * sigma)
    return float(max(1e-5, min(1.0, sigma)))


def sample_timestep(args: argparse.Namespace, num_train_timesteps: int) -> int:
    if args.timestep_sampling == "logitnormal":
        sigma = 1.0 / (1.0 + math.exp(-random.gauss(0.0, 1.0)))
        return max(1, min(num_train_timesteps, int(round(sigma * num_train_timesteps))))
    return random.randint(1, num_train_timesteps)


def loss_mask_from_condition(vision_condition_mask: Any, x0: Any):
    mask = 1.0 - vision_condition_mask
    while mask.ndim < x0.ndim:
        mask = mask.unsqueeze(0)
    return mask.to(device=x0.device, dtype=x0.dtype)


def pack_static(pipe: Any, input_ids: list[int], latents: Any, action_tokens: Any, action_condition_frames: list[int], fps: float, device: str):
    import torch

    text_segment = pipe._prepare_text_segment(input_ids, device=device)
    vision_condition_indexes = [0]
    vision_segment = pipe._prepare_vision_segment(
        input_vision_tokens=latents,
        has_image_condition=True,
        mrope_offset=text_segment["vision_start_temporal_offset"],
        vision_fps=fps,
        curr=text_segment["und_len"],
        device=device,
        condition_frame_indexes=vision_condition_indexes,
    )
    action_segment = pipe._prepare_action_segment(
        input_action_tokens=action_tokens,
        condition_frame_indexes=action_condition_frames,
        mrope_offset=text_segment["vision_start_temporal_offset"],
        action_fps=fps,
        curr=text_segment["und_len"] + vision_segment["num_vision_tokens"],
        device=device,
    )
    position_ids = torch.cat(
        [
            text_segment["text_mrope_ids"],
            vision_segment["vision_mrope_ids"],
            action_segment["action_mrope_ids"],
        ],
        dim=1,
    )
    return {
        **text_segment,
        **vision_segment,
        **action_segment,
        "position_ids": position_ids,
        "sequence_length": text_segment["und_len"] + vision_segment["num_vision_tokens"] + action_segment["action_len"],
    }


def training_step(pipe: Any, row: dict[str, Any], args: argparse.Namespace, device: str, dtype: Any, num_train_timesteps: int, sigma_shift: float):
    import torch

    contract = row_contract(row, require_media_exists=args.require_media_exists)
    if contract["issues"]:
        raise ValueError(f"row contract issues for {contract['row_id']}: {contract['issues']}")
    if contract["mode"] != "forward_dynamics":
        raise ValueError(f"expected forward_dynamics target, got {contract['mode']}")

    action = instantiate_action(row, args.override_resolution_tier)
    with torch.no_grad():
        x0, _action_image_size, height, width = clean_vision_latents(pipe, action, device, dtype)
        act_tokens, act_mask, raw_action_dim, act_condition_frames = action_latents(action, pipe, device, dtype)
        input_ids = tokenize_prompt(pipe, args, action, height, width)

    timestep = sample_timestep(args, num_train_timesteps)
    sigma = shifted_sigma(timestep, num_train_timesteps, sigma_shift)
    noise = torch.randn_like(x0, dtype=x0.dtype, device=x0.device)
    velocity_target = noise - x0

    latent_t = int(x0.shape[1])
    vision_condition_mask = torch.zeros((latent_t, 1, 1), device=x0.device, dtype=x0.dtype)
    vision_condition_mask[0, 0, 0] = 1.0
    loss_mask = loss_mask_from_condition(vision_condition_mask, x0)
    noised = x0 + sigma * velocity_target
    latents = vision_condition_mask * x0.to(dtype) + (1.0 - vision_condition_mask) * noised.to(dtype)

    packed = pack_static(pipe, input_ids, latents, act_tokens, act_condition_frames, args.fps, device)
    vision_timesteps = torch.full((packed["num_noisy_vision_tokens"],), timestep, device=device)
    action_timesteps = torch.full((packed["num_noisy_action_tokens"],), timestep, device=device)

    preds_vision, preds_sound, preds_action = pipe.transformer(
        input_ids=packed["input_ids"],
        text_indexes=packed["text_indexes"],
        position_ids=packed["position_ids"],
        und_len=packed["und_len"],
        sequence_length=packed["sequence_length"],
        vision_tokens=[latents.to(device=device, dtype=dtype)],
        vision_token_shapes=packed["vision_token_shapes"],
        vision_sequence_indexes=packed["vision_sequence_indexes"],
        vision_mse_loss_indexes=packed["vision_mse_loss_indexes"],
        vision_timesteps=vision_timesteps,
        vision_noisy_frame_indexes=packed["vision_noisy_frame_indexes"],
        action_tokens=[act_tokens.to(device=device, dtype=dtype)],
        action_token_shapes=packed["action_token_shapes"],
        action_sequence_indexes=packed["action_sequence_indexes"],
        action_mse_loss_indexes=packed["action_mse_loss_indexes"],
        action_timesteps=action_timesteps,
        action_noisy_frame_indexes=packed["action_noisy_frame_indexes"],
        action_domain_ids=[action_domain_id(action.domain_name, device)],
    )
    pred_velocity, _pred_sound, _pred_action = pipe._mask_velocity_predictions(
        preds_vision,
        preds_sound,
        vision_condition_mask=[vision_condition_mask.to(dtype=preds_vision[0].dtype)],
        preds_action=preds_action,
        action_condition_mask=[act_mask],
        raw_action_dim=raw_action_dim,
    )
    target = velocity_target.to(device=pred_velocity.device, dtype=pred_velocity.dtype)
    mask = loss_mask.to(device=pred_velocity.device, dtype=pred_velocity.dtype)
    denom = mask.expand_as(pred_velocity).sum()
    loss = ((pred_velocity - target) ** 2 * mask).sum() / denom.clamp_min(1.0)
    if args.loss_scale:
        loss = loss * args.loss_scale
    return loss, {
        "row_id": contract["row_id"],
        "episode_id": contract["episode_id"],
        "timestep": timestep,
        "sigma": sigma,
        "height": height,
        "width": width,
        "vision_latents_shape": list(x0.shape),
        "action_latents_shape": list(act_tokens.shape),
        "vision_loss_tokens": int(packed["vision_mse_loss_indexes"].numel()),
        "action_loss_tokens": int(packed["action_mse_loss_indexes"].numel()),
    }


def save_adapter(pipe: Any, output_dir: Path, adapter_name: str) -> Path:
    adapter_dir = output_dir / "adapter_lora"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    pipe.transformer.save_lora_adapter(str(adapter_dir), adapter_name=adapter_name)
    return adapter_dir


def write_report(output_dir: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Cosmos3-Super Forward-Dynamics LoRA",
        "",
        f"- Run id: `{payload['run_id']}`",
        f"- Status: `{payload['status']}`",
        f"- Weights updated: `{payload['weights_updated']}`",
        f"- Dataset: `{payload['dataset_jsonl']}`",
        f"- Train samples: `{payload['train_samples']}`",
        f"- Max steps: `{payload['max_steps']}`",
        f"- Final loss: `{payload.get('final_loss')}`",
        f"- Adapter dir: `{payload.get('adapter_dir')}`",
        "",
        "## Scope",
        "",
        "This adapter trains Cosmos3-Super camera-pose forward dynamics. Raw camera-pose actions are conditioning, and the loss supervises future vision velocity tokens. It is not a JSON Reasoner SFT run and does not supervise `preds_action`.",
    ]
    (output_dir / "RUN_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.workspace = args.workspace.expanduser().resolve()
    args.dataset_jsonl = args.dataset_jsonl.expanduser().resolve()
    args.model_dir = args.model_dir.expanduser().resolve()
    output_dir = args.output_dir or args.workspace / "results" / "omni_finetune" / args.run_id
    output_dir = output_dir.expanduser().resolve()
    progress_path = output_dir / "progress.jsonl"
    if progress_path.exists():
        progress_path.unlink()

    random.seed(args.seed)
    started = time.time()
    append_jsonl(progress_path, {"event": "start", "run_id": args.run_id, "timestamp": started})

    inner = model_inner_config(args.model_dir)
    train_rows = select_rows(load_jsonl(args.dataset_jsonl), args)
    target_modules = lora_targets(args, inner)
    lora_rank = int(args.lora_rank or inner.get("lora_rank") or 16)
    lora_alpha = int(args.lora_alpha or inner.get("lora_alpha") or 32)
    train_cfg = inner.get("rectified_flow_training_config") or {}
    num_train_timesteps = int(args.num_train_timesteps or ((inner.get("rectified_flow_inference_config") or {}).get("num_train_timesteps") or 1000))
    shift_table = train_cfg.get("shift") if isinstance(train_cfg.get("shift"), dict) else {}
    resolution_key = str(args.override_resolution_tier or 480)
    sigma_shift = float(args.resolution_shift or shift_table.get(resolution_key) or 1.0)
    loss_scale = args.loss_scale if args.loss_scale is not None else train_cfg.get("loss_scale")
    args.loss_scale = float(loss_scale) if loss_scale is not None else None

    append_jsonl(
        progress_path,
        {
            "event": "dataset_ready",
            "timestamp": time.time(),
            "train_samples": len(train_rows),
            "target_modules": target_modules,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "sigma_shift": sigma_shift,
            "loss_scale": args.loss_scale,
        },
    )

    import torch
    from diffusers import Cosmos3OmniPipeline
    from peft import LoraConfig

    dtype = dtype_from_name(args.dtype)
    load_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "local_files_only": args.local_files_only,
        "enable_safety_checker": False,
    }
    if args.device_map != "none":
        load_kwargs["device_map"] = args.device_map
    pipe = Cosmos3OmniPipeline.from_pretrained(str(args.model_dir), **load_kwargs)
    if args.device_map == "none":
        pipe.to(args.device)
        device = args.device
    else:
        device = str(pipe._get_execution_device())
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)

    pipe.transformer.requires_grad_(False)
    if args.gradient_checkpointing and hasattr(pipe.transformer, "enable_gradient_checkpointing"):
        pipe.transformer.enable_gradient_checkpointing()
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    pipe.transformer.add_adapter(lora_config, adapter_name=args.adapter_name)
    pipe.transformer.set_adapter(args.adapter_name)
    pipe.transformer.train()
    for component_name in ("vae", "sound_tokenizer"):
        component = getattr(pipe, component_name, None)
        if component is not None:
            component.requires_grad_(False)
            component.eval()

    trainable = [param for param in pipe.transformer.parameters() if param.requires_grad]
    trainable_params = sum(param.numel() for param in trainable)
    append_jsonl(progress_path, {"event": "model_ready", "timestamp": time.time(), "device": device, "trainable_params": trainable_params})
    if not trainable:
        raise RuntimeError("no trainable LoRA parameters found")

    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    losses: list[float] = []
    status = "dry_run_complete" if args.dry_run else "complete"
    adapter_dir: Path | None = None
    try:
        for step in range(1, args.max_steps + 1):
            row = train_rows[(step - 1) % len(train_rows)]
            optimizer.zero_grad(set_to_none=True)
            loss, info = training_step(pipe, row, args, device, dtype, num_train_timesteps, sigma_shift)
            loss_value = float(loss.detach().float().cpu())
            losses.append(loss_value)
            if not args.dry_run:
                loss.backward()
                optimizer.step()
            if step == 1 or step % args.progress_every == 0 or step == args.max_steps:
                append_jsonl(
                    progress_path,
                    {
                        "event": "train_step",
                        "timestamp": time.time(),
                        "step": step,
                        "loss": loss_value,
                        **info,
                    },
                )
        if not args.dry_run:
            adapter_dir = save_adapter(pipe, output_dir, args.adapter_name)
            append_jsonl(progress_path, {"event": "adapter_saved", "timestamp": time.time(), "adapter_dir": str(adapter_dir)})
    except Exception as exc:
        status = "failed"
        append_jsonl(progress_path, {"event": "failed", "timestamp": time.time(), "error": repr(exc)})
        raise
    finally:
        finished = time.time()
        payload = {
            "run_id": args.run_id,
            "run_kind": "cosmos3_super_forward_dynamics_lora",
            "status": status,
            "started_at_unix": started,
            "finished_at_unix": finished,
            "elapsed_seconds": finished - started,
            "workspace": str(args.workspace),
            "dataset_jsonl": str(args.dataset_jsonl),
            "model_dir": str(args.model_dir),
            "split": args.split,
            "episode_id": args.episode_id,
            "train_samples": len(train_rows),
            "max_steps": args.max_steps,
            "learning_rate": args.learning_rate,
            "target_modules": target_modules,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "trainable_params": trainable_params if "trainable_params" in locals() else None,
            "num_train_timesteps": num_train_timesteps,
            "sigma_shift": sigma_shift,
            "loss_scale": args.loss_scale,
            "final_loss": losses[-1] if losses else None,
            "losses": losses,
            "adapter_dir": str(adapter_dir) if adapter_dir else None,
            "weights_updated": bool(adapter_dir),
            "loss_surface": "vision_velocity_conditioned_on_camera_pose",
            "action_loss_expected": False,
        }
        write_json(output_dir / "training_metadata.json", payload)
        write_report(output_dir, payload)
        append_jsonl(progress_path, {"event": "complete", "timestamp": time.time(), "status": status})

    print(json.dumps({"status": status, "output_dir": str(output_dir), "adapter_dir": str(adapter_dir) if adapter_dir else None}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

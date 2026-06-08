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
import shutil
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
DEFAULT_COSMOS3_SUPER_LORA_TARGETS = [
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


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


def distributed_slice(samples: list[dict[str, Any]], process_index: int, num_processes: int) -> list[dict[str, Any]]:
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


def checkpoint_module_suffixes(model_dir: Path) -> set[str]:
    suffixes: set[str] = set()
    for index_path in (
        model_dir / "model.safetensors.index.json",
        model_dir / "transformer" / "diffusion_pytorch_model.safetensors.index.json",
    ):
        index = read_json(index_path)
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict):
            continue
        for key in weight_map:
            if key.endswith(".weight"):
                suffixes.add(key[:-7].split(".")[-1])
    return suffixes


def lora_targets(args: argparse.Namespace, inner: dict[str, Any], model_dir: Path) -> list[str]:
    raw = args.target_modules or inner.get("lora_target_modules") or ""
    modules = [item.strip() for item in str(raw).split(",") if item.strip()]
    if args.target_modules:
        return modules

    available = checkpoint_module_suffixes(model_dir)
    if modules and (not available or any(module in available for module in modules)):
        return modules

    fallback = [module for module in DEFAULT_COSMOS3_SUPER_LORA_TARGETS if not available or module in available]
    return fallback or modules or DEFAULT_COSMOS3_SUPER_LORA_TARGETS


def parameter_dtype_counts(module: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for param in module.parameters():
        key = str(param.dtype).replace("torch.", "")
        counts[key] = counts.get(key, 0) + param.numel()
    return counts


def adapter_tensor_numel(adapter_dir: Path) -> int:
    weight_path = adapter_dir / "pytorch_lora_weights.safetensors"
    if not weight_path.exists():
        return -1
    from safetensors.torch import load_file

    state = load_file(str(weight_path), device="cpu")
    return sum(tensor.numel() for tensor in state.values())


def expected_lora_shape(transformer: Any, key: str, lora_rank: int) -> tuple[int, ...] | None:
    marker = ".lora_"
    if marker not in key:
        return None
    module_name, suffix = key.split(marker, 1)
    candidates = [module_name]
    parts = module_name.split(".")
    if len(parts) >= 2 and parts[0] == "layers":
        candidates.append(".".join([parts[0], parts[1], "_fsdp_wrapped_module", *parts[2:]]))
    module = None
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            module = transformer.get_submodule(candidate)
            break
        except AttributeError as exc:
            last_error = exc
            continue
    try:
        if module is None:
            raise last_error or AttributeError(module_name)
    except AttributeError:
        try:
            module = transformer
            for part in module_name.split("."):
                module = getattr(module, part)
        except AttributeError:
            return None
    base = getattr(module, "base_layer", module)
    in_features = getattr(base, "in_features", None)
    out_features = getattr(base, "out_features", None)
    if in_features is None or out_features is None:
        return None
    if suffix.startswith("lora_A."):
        return (int(lora_rank), int(in_features))
    if suffix.startswith("lora_B."):
        return (int(out_features), int(lora_rank))
    return None


def fallback_flat_lora_shape(key: str, tensor_numel: int, lora_rank: int) -> tuple[int, ...] | None:
    if lora_rank <= 0 or tensor_numel % lora_rank:
        return None
    if ".lora_A." in key:
        return (int(lora_rank), int(tensor_numel // lora_rank))
    if ".lora_B." in key:
        return (int(tensor_numel // lora_rank), int(lora_rank))
    return None


def repair_lora_adapter_shapes(adapter_dir: Path, transformer: Any, lora_rank: int) -> dict[str, Any]:
    weight_path = adapter_dir / "pytorch_lora_weights.safetensors"
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    with safe_open(str(weight_path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    state = load_file(str(weight_path), device="cpu")
    repaired: dict[str, Any] = {}
    for key, tensor in list(state.items()):
        expected_shape = expected_lora_shape(transformer, key, lora_rank)
        if expected_shape is None and tensor.ndim == 1:
            expected_shape = fallback_flat_lora_shape(key, tensor.numel(), lora_rank)
        if expected_shape is None or tuple(tensor.shape) == expected_shape:
            continue
        if tensor.ndim == 1 and tensor.numel() == math.prod(expected_shape):
            state[key] = tensor.reshape(expected_shape).contiguous()
            repaired[key] = {"from": list(tensor.shape), "to": list(expected_shape)}
    if repaired:
        save_file(state, str(weight_path), metadata=metadata)
    return {"adapter_file": str(weight_path), "tensor_numel": sum(t.numel() for t in state.values()), "repaired": repaired}


def json_safe(value: Any) -> Any:
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def canonical_lora_key(name: str, adapter_name: str) -> str | None:
    for marker in (".lora_A.", ".lora_B."):
        if marker not in name:
            continue
        prefix, suffix = name.split(marker, 1)
        prefix = prefix.replace("._fsdp_wrapped_module", "")
        adapter_prefix = f"{adapter_name}."
        if suffix.startswith(adapter_prefix):
            suffix = suffix[len(adapter_prefix) :]
        return f"{prefix}{marker}{suffix}"
    return None


def distributed_save_lora_adapter(transformer: Any, adapter_dir: Path, adapter_name: str, lora_config: Any, lora_rank: int) -> dict[str, Any]:
    import torch.distributed as dist
    from safetensors.torch import save_file

    adapter_dir.mkdir(parents=True, exist_ok=True)
    local_state: dict[str, Any] = {}
    expected_shapes: dict[str, tuple[int, ...]] = {}
    for name, param in transformer.named_parameters():
        key = canonical_lora_key(name, adapter_name)
        if key is None or not param.requires_grad:
            continue
        expected_shape = expected_lora_shape(transformer, key, lora_rank)
        if expected_shape is not None:
            expected_shapes[key] = expected_shape
        if param.numel() == 0:
            continue
        tensor = param.detach().cpu()
        if expected_shape is not None and tensor.ndim == 1 and tensor.numel() == math.prod(expected_shape):
            tensor = tensor.reshape(expected_shape)
        local_state[key] = tensor.contiguous()

    if dist.is_available() and dist.is_initialized():
        gathered: list[dict[str, Any] | None] = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, local_state)
        rank = dist.get_rank()
    else:
        gathered = [local_state]
        rank = 0

    merged: dict[str, Any] = {}
    duplicate_keys: list[str] = []
    for shard in gathered:
        if not shard:
            continue
        for key, tensor in shard.items():
            if key in merged:
                duplicate_keys.append(key)
            expected_shape = expected_shapes.get(key)
            if expected_shape is None and tensor.ndim == 1:
                expected_shape = fallback_flat_lora_shape(key, tensor.numel(), lora_rank)
                if expected_shape is not None:
                    expected_shapes[key] = expected_shape
            if expected_shape is not None and tensor.ndim == 1 and tensor.numel() == math.prod(expected_shape):
                tensor = tensor.reshape(expected_shape).contiguous()
            merged[key] = tensor

    missing_keys = sorted(set(expected_shapes) - set(merged))
    adapter_file = adapter_dir / "pytorch_lora_weights.safetensors"
    metadata = {
        "format": "pt",
        "lora_adapter_metadata": json.dumps(json_safe(lora_config.to_dict()), indent=2, sort_keys=True),
    }
    if rank == 0:
        if missing_keys:
            raise RuntimeError(f"missing gathered LoRA tensors: {missing_keys[:8]} ({len(missing_keys)} total)")
        save_file(merged, str(adapter_file), metadata=metadata)
    return {
        "adapter_dir": str(adapter_dir),
        "adapter_file": str(adapter_file),
        "local_keys": sorted(local_state),
        "merged_keys": sorted(merged),
        "missing_keys": missing_keys,
        "duplicate_keys": sorted(duplicate_keys),
        "tensor_numel": sum(tensor.numel() for tensor in merged.values()),
        "tensor_shapes": {key: list(tensor.shape) for key, tensor in merged.items()},
    }


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


def load_video_frames(video_path: str | Path, max_frames: int) -> list[Any]:
    import cv2
    from PIL import Image

    path = Path(video_path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"failed to open conditioning video: {path}")

    frames: list[Any] = []
    try:
        while len(frames) < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
    finally:
        capture.release()

    if not frames:
        raise ValueError(f"conditioning video has no decodable frames: {path}")
    return frames


def conditioning_clip_for_action(action: Any, target_frames: int) -> Any:
    if action.image is not None:
        if isinstance(action.image, (str, Path)):
            from PIL import Image

            return [Image.open(action.image).convert("RGB")]
        return [action.image]

    video = action.video
    if isinstance(video, list) and video and isinstance(video[0], (str, Path)):
        return load_video_frames(video[0], target_frames)
    if isinstance(video, (str, Path)):
        return load_video_frames(video, target_frames)
    return video


def action_domain_id(domain_name: str, device: str):
    import torch
    from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import _EMBODIMENT_TO_DOMAIN_ID

    if domain_name not in _EMBODIMENT_TO_DOMAIN_ID:
        raise ValueError(f"unknown Cosmos3 action domain: {domain_name}")
    return torch.tensor([_EMBODIMENT_TO_DOMAIN_ID[domain_name]], dtype=torch.long, device=device)


def clean_vision_latents(pipe: Any, action: Any, device: str, dtype: Any):
    target_frames = action.chunk_size + 1
    conditioning_clip = conditioning_clip_for_action(action, target_frames)
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
    transformer = pipe.transformer
    action_dim = getattr(transformer, "action_dim", None)
    if action_dim is None and hasattr(transformer, "module"):
        action_dim = getattr(transformer.module, "action_dim", None)
    if action_dim is None:
        raise ValueError("Cosmos3 transformer does not expose action_dim")
    action_dim = int(action_dim)
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


def training_step(
    pipe: Any,
    row: dict[str, Any],
    args: argparse.Namespace,
    device: str,
    dtype: Any,
    num_train_timesteps: int,
    sigma_shift: float,
    grad_enabled: bool = True,
):
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

    latent_t = int(x0.shape[2])
    vision_condition_mask = torch.zeros((1, latent_t, 1, 1), device=x0.device, dtype=x0.dtype)
    vision_condition_mask[:, 0, 0, 0] = 1.0
    loss_mask = loss_mask_from_condition(vision_condition_mask, x0)
    latent_condition_mask = vision_condition_mask.unsqueeze(0)
    noised = x0 + sigma * velocity_target
    latents = latent_condition_mask * x0.to(dtype) + (1.0 - latent_condition_mask) * noised.to(dtype)

    packed = pack_static(pipe, input_ids, latents, act_tokens, act_condition_frames, args.fps, device)
    vision_timesteps = torch.full((packed["num_noisy_vision_tokens"],), timestep, device=device)
    action_timesteps = torch.full((packed["num_noisy_action_tokens"],), timestep, device=device)

    grad_context = torch.enable_grad() if grad_enabled else torch.no_grad()
    with grad_context:
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
    from accelerate import Accelerator

    accelerator = Accelerator()
    args.workspace = args.workspace.expanduser().resolve()
    args.dataset_jsonl = args.dataset_jsonl.expanduser().resolve()
    args.model_dir = args.model_dir.expanduser().resolve()
    output_dir = args.output_dir or args.workspace / "results" / "omni_finetune" / args.run_id
    output_dir = output_dir.expanduser().resolve()
    progress_path = output_dir / "progress.jsonl"
    if accelerator.is_main_process and progress_path.exists():
        progress_path.unlink()
    accelerator.wait_for_everyone()

    random.seed(args.seed + accelerator.process_index)
    started = time.time()
    if accelerator.is_main_process:
        append_jsonl(progress_path, {"event": "start", "run_id": args.run_id, "timestamp": started})

    inner = model_inner_config(args.model_dir)
    all_train_rows = select_rows(load_jsonl(args.dataset_jsonl), args)
    train_rows = distributed_slice(all_train_rows, accelerator.process_index, accelerator.num_processes)
    target_modules = lora_targets(args, inner, args.model_dir)
    lora_rank = int(args.lora_rank or inner.get("lora_rank") or 16)
    lora_alpha = int(args.lora_alpha or inner.get("lora_alpha") or 32)
    train_cfg = inner.get("rectified_flow_training_config") or {}
    num_train_timesteps = int(args.num_train_timesteps or ((inner.get("rectified_flow_inference_config") or {}).get("num_train_timesteps") or 1000))
    shift_table = train_cfg.get("shift") if isinstance(train_cfg.get("shift"), dict) else {}
    resolution_key = str(args.override_resolution_tier or 480)
    sigma_shift = float(args.resolution_shift or shift_table.get(resolution_key) or 1.0)
    loss_scale = args.loss_scale if args.loss_scale is not None else train_cfg.get("loss_scale")
    args.loss_scale = float(loss_scale) if loss_scale is not None else None

    if accelerator.is_main_process:
        append_jsonl(
            progress_path,
            {
                "event": "dataset_ready",
                "timestamp": time.time(),
                "num_processes": accelerator.num_processes,
                "train_samples": len(all_train_rows),
                "rank0_samples": len(train_rows),
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

    torch.set_grad_enabled(True)
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
    if args.device_map == "none":
        pipe.transformer.to(dtype=dtype)
    pipe.transformer.train()
    for param in pipe.transformer.parameters():
        if param.requires_grad and param.dtype != dtype:
            param.data = param.data.to(dtype=dtype)
    for component_name in ("vae", "sound_tokenizer"):
        component = getattr(pipe, component_name, None)
        if component is not None:
            component.requires_grad_(False)
            component.eval()

    trainable = [param for param in pipe.transformer.parameters() if param.requires_grad]
    trainable_params = sum(param.numel() for param in trainable)
    if accelerator.is_main_process:
        append_jsonl(
            progress_path,
            {
                "event": "model_ready",
                "timestamp": time.time(),
                "device": str(device),
                "device_map": args.device_map,
                "trainable_params": trainable_params,
                "parameter_dtype_counts": parameter_dtype_counts(pipe.transformer),
            },
        )
    if not trainable:
        raise RuntimeError("no trainable LoRA parameters found")

    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    if accelerator.num_processes > 1:
        if accelerator.is_main_process:
            append_jsonl(progress_path, {"event": "accelerator_prepare_start", "timestamp": time.time()})
        pipe.transformer, optimizer = accelerator.prepare(pipe.transformer, optimizer)
        if accelerator.is_main_process:
            append_jsonl(progress_path, {"event": "accelerator_prepare_done", "timestamp": time.time()})
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
                if accelerator.num_processes > 1:
                    accelerator.backward(loss)
                else:
                    loss.backward()
                optimizer.step()
            if step == 1 or step % args.progress_every == 0 or step == args.max_steps:
                if accelerator.is_main_process:
                    append_jsonl(
                        progress_path,
                        {
                            "event": "train_step",
                            "timestamp": time.time(),
                            "step": step,
                            "rank0_loss": loss_value,
                            **info,
                        },
                    )
        if not args.dry_run:
            if accelerator.is_main_process:
                append_jsonl(progress_path, {"event": "save_start", "timestamp": time.time()})
            accelerator.wait_for_everyone()
            if accelerator.num_processes > 1:
                pipe.transformer = accelerator.unwrap_model(pipe.transformer)
                adapter_dir = output_dir / "adapter_lora"
                adapter_audit = distributed_save_lora_adapter(
                    pipe.transformer,
                    adapter_dir,
                    args.adapter_name,
                    lora_config,
                    lora_rank,
                )
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    append_jsonl(
                        progress_path,
                        {
                            "event": "adapter_saved",
                            "timestamp": time.time(),
                            "adapter_dir": str(adapter_dir),
                            "adapter_audit": adapter_audit,
                        },
                    )
                    write_json(output_dir / "adapter_shape_check.json", adapter_audit)
            else:
                adapter_dir = save_adapter(pipe, output_dir, args.adapter_name)
                if accelerator.is_main_process:
                    append_jsonl(progress_path, {"event": "adapter_saved", "timestamp": time.time(), "adapter_dir": str(adapter_dir)})
            accelerator.wait_for_everyone()
    except Exception as exc:
        status = "failed"
        if accelerator.is_main_process:
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
            "num_processes": accelerator.num_processes,
            "train_samples": len(all_train_rows),
            "rank_samples": len(train_rows),
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
        if accelerator.is_main_process:
            write_json(output_dir / "training_metadata.json", payload)
            write_report(output_dir, payload)
            append_jsonl(progress_path, {"event": "complete", "timestamp": time.time(), "status": status})

    if accelerator.is_main_process:
        print(json.dumps({"status": status, "output_dir": str(output_dir), "adapter_dir": str(adapter_dir) if adapter_dir else None}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

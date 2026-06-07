#!/usr/bin/env python3
"""Pack one Cosmos3-Super action-conditioning batch from Xperience windows.

This is the bridge between the public-safe Xperience JSONL export and a real
Cosmos3 Diffusers trainer. It can run in two modes:

- schema mode: validate the selected row and infer the supervised loss surface
  without loading the huge model.
- pipeline mode: load Cosmos3OmniPipeline and call the installed
  prepare_latents/_prepare_*_segment helpers to verify tensor shapes and loss
  indexes for one sample.

The current camera_pose target export uses mode=forward_dynamics. In the
installed Cosmos3 pipeline that mode treats actions as conditioning and
supervises noisy vision tokens, not preds_action. Policy/inverse-dynamics action
prediction requires a separate target export mode.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from qwen3_omni_dataset_utils import load_jsonl


ACTION_TARGET_KEYS = (
    "cosmos_action_target",
    "cosmos3_action_target",
    "cosmos_action_condition",
    "action_target",
)


def parse_args() -> argparse.Namespace:
    workspace_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=workspace_default)
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    parser.add_argument("--run-id", default="xperience10m_cosmos3_super_action_packer_smoke")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument(
        "--backbone-config",
        type=Path,
        default=workspace_default / "configs" / "omni_backbones" / "cosmos3_super_reasoner.json",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--sample-id")
    parser.add_argument("--prompt", default="Predict the embodied future under the provided camera-pose action condition.")
    parser.add_argument("--negative-prompt")
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--load-pipeline", action="store_true")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-media-exists", action="store_true")
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
    return json.loads(path.read_text(encoding="utf-8"))


def find_action_target(row: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
    for key in ACTION_TARGET_KEYS:
        value = row.get(key)
        if isinstance(value, dict):
            return key, value
    return None, None


def selected_row(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    candidates = [row for row in rows if row.get("split") == args.split and find_action_target(row)[1] is not None]
    if args.sample_id:
        for row in rows:
            if row.get("id") == args.sample_id:
                return row
        raise ValueError(f"sample id not found: {args.sample_id}")
    if not candidates:
        raise ValueError(f"no rows with action targets found for split={args.split!r}")
    if args.sample_index < 0 or args.sample_index >= len(candidates):
        raise ValueError(f"sample-index {args.sample_index} outside 0..{len(candidates)-1}")
    return candidates[args.sample_index]


def numeric_matrix(value: Any) -> tuple[bool, tuple[int, int] | None]:
    if not isinstance(value, list) or not value:
        return False, None
    width = None
    for item in value:
        if not isinstance(item, list) or not item:
            return False, None
        width = len(item) if width is None else width
        if len(item) != width:
            return False, None
        for number in item:
            if not isinstance(number, (int, float)):
                return False, None
    return True, (len(value), int(width or 0))


def media_video_path(row: dict[str, Any], target: dict[str, Any]) -> str | None:
    conditioning = target.get("conditioning") if isinstance(target.get("conditioning"), dict) else {}
    media = row.get("media") if isinstance(row.get("media"), dict) else {}
    for block in (conditioning, media):
        value = block.get("mosaic_video_path")
        if value:
            return str(value)
    for block in (conditioning, media):
        paths = block.get("video_paths")
        if isinstance(paths, list):
            for item in paths:
                if isinstance(item, dict) and item.get("path"):
                    return str(item["path"])
    return None


def row_contract(row: dict[str, Any], require_media_exists: bool) -> dict[str, Any]:
    key, target = find_action_target(row)
    if target is None:
        raise ValueError(f"row has no Cosmos action target: {row.get('id')}")

    video_path = media_video_path(row, target)
    if not video_path:
        raise ValueError(f"row has no video conditioning path: {row.get('id')}")
    if require_media_exists and not Path(video_path).exists():
        raise FileNotFoundError(video_path)

    mode = str(target.get("mode"))
    domain_name = str(target.get("domain_name"))
    chunk_size = int(target.get("chunk_size"))
    raw_actions = target.get("raw_actions")
    ok, shape = numeric_matrix(raw_actions)
    raw_action_dim = int(target.get("raw_action_dim") or (shape[1] if shape else 0))
    issues: list[str] = []
    if mode not in {"forward_dynamics", "policy", "inverse_dynamics"}:
        issues.append(f"unsupported mode={mode!r}")
    if domain_name != "camera_pose":
        issues.append(f"expected camera_pose target for this export, got {domain_name!r}")
    if chunk_size < 1:
        issues.append("chunk_size must be >= 1")
    if mode == "forward_dynamics":
        if not ok:
            issues.append("forward_dynamics requires numeric raw_actions")
        elif shape and shape[1] != raw_action_dim:
            issues.append(f"raw_actions width {shape[1]} does not match raw_action_dim {raw_action_dim}")

    if mode == "forward_dynamics":
        loss_surface = "vision_velocity_conditioned_on_camera_pose"
        action_loss_expected = False
        note = (
            "Cosmos3 forward_dynamics consumes raw_actions as conditioning and predicts noisy vision tokens. "
            "It does not supervise preds_action for this target mode."
        )
    else:
        loss_surface = "action_velocity"
        action_loss_expected = True
        note = (
            "Cosmos3 policy/inverse_dynamics can expose noisy action tokens, but the current camera-pose export "
            "does not yet create that target mode."
        )

    return {
        "row_id": row.get("id"),
        "episode_id": row.get("episode_id"),
        "split": row.get("split"),
        "target_key": key,
        "mode": mode,
        "domain_name": domain_name,
        "chunk_size": chunk_size,
        "raw_action_dim": raw_action_dim,
        "raw_actions_shape": list(shape) if shape else None,
        "video_path": video_path,
        "video_path_exists": Path(video_path).exists(),
        "loss_surface": loss_surface,
        "action_loss_expected": action_loss_expected,
        "interpretation": note,
        "issues": issues,
    }


def instantiate_action_condition(row: dict[str, Any], contract: dict[str, Any]):
    import torch
    from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import CosmosActionCondition

    _, target = find_action_target(row)
    if target is None:
        raise ValueError("missing action target")
    raw_actions = None
    if target.get("raw_actions") is not None:
        raw_actions = torch.tensor(target["raw_actions"], dtype=torch.float32)
    video = [contract["video_path"]]
    return CosmosActionCondition(
        mode=contract["mode"],
        chunk_size=int(contract["chunk_size"]),
        domain_name=contract["domain_name"],
        resolution_tier=int(target.get("resolution_tier", 480)),
        raw_actions=raw_actions,
        video=video,
        view_point=str(target.get("view_point", "ego_view")),
    )


def resolve_action_canvas(pipe, action) -> tuple[int | None, int | None]:
    try:
        from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import _ACTION_RESOLUTION_BINS, VideoProcessor

        conditioning_clip = [action.image] if action.image is not None else action.video
        probe = pipe.video_processor.preprocess_video(conditioning_clip)
        source_h, source_w = int(probe.shape[-2]), int(probe.shape[-1])
        resolution_key = str(action.resolution_tier)
        return VideoProcessor.classify_height_width_bin(source_h, source_w, ratios=_ACTION_RESOLUTION_BINS[resolution_key])
    except Exception:
        return None, None


def tokenize_prompt(pipe, args: argparse.Namespace, action, height: int | None, width: int | None) -> list[int]:
    if hasattr(pipe, "tokenize_prompt"):
        cond_ids, _ = pipe.tokenize_prompt(
            args.prompt,
            args.negative_prompt,
            num_frames=action.chunk_size + 1,
            height=height,
            width=width,
            fps=args.fps,
            action_mode=action.mode,
            action_view_point=action.view_point,
        )
        return list(cond_ids)
    encoded = pipe.tokenizer(args.prompt, add_special_tokens=True)
    return list(encoded["input_ids"])


def pack_with_pipeline(row: dict[str, Any], contract: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from diffusers import Cosmos3OmniPipeline

    if args.model_dir is None:
        raise ValueError("--model-dir is required with --load-pipeline")
    dtype = dtype_from_name(args.dtype)
    pipe = Cosmos3OmniPipeline.from_pretrained(
        str(args.model_dir),
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    )
    pipe.to(args.device)
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)

    action = instantiate_action_condition(row, contract)
    height, width = resolve_action_canvas(pipe, action)
    input_ids = tokenize_prompt(pipe, args, action, height, width)
    text_segment = pipe._prepare_text_segment(input_ids, device=args.device)
    (
        latents,
        sound_latents,
        action_latents,
        fps_vision,
        fps_sound,
        vision_condition_mask,
        sound_condition_mask,
        action_condition_mask,
        action_domain_id,
        action_image_size,
        raw_action_dim_resolved,
        action_condition_frame_indexes,
    ) = pipe.prepare_latents(
        num_frames=action.chunk_size + 1,
        height=height,
        width=width,
        fps=args.fps,
        device=args.device,
        dtype=dtype,
        enable_sound=False,
        action=action,
    )
    vision_condition_indexes = torch.nonzero(vision_condition_mask[:, 0, 0] > 0, as_tuple=False).flatten()
    vision_condition_indexes = [int(idx.item()) for idx in vision_condition_indexes]
    vision_segment = pipe._prepare_vision_segment(
        input_vision_tokens=latents,
        has_image_condition=bool(vision_condition_indexes),
        mrope_offset=text_segment["vision_start_temporal_offset"],
        vision_fps=fps_vision,
        curr=text_segment["und_len"],
        device=args.device,
        condition_frame_indexes=vision_condition_indexes,
    )
    action_segment = {}
    if action_latents is not None:
        action_segment = pipe._prepare_action_segment(
            input_action_tokens=action_latents,
            condition_frame_indexes=action_condition_frame_indexes,
            mrope_offset=text_segment["vision_start_temporal_offset"],
            action_fps=fps_vision,
            curr=text_segment["und_len"] + vision_segment["num_vision_tokens"],
            device=args.device,
        )
    action_loss_tokens = int(action_segment.get("action_mse_loss_indexes", torch.tensor([])).numel())
    vision_loss_tokens = int(vision_segment.get("vision_mse_loss_indexes", torch.tensor([])).numel())
    status = "pass"
    if contract["mode"] == "forward_dynamics" and action_loss_tokens != 0:
        status = "warning_unexpected_action_loss_tokens"
    elif contract["mode"] != "forward_dynamics" and action_loss_tokens == 0:
        status = "warning_no_action_loss_tokens"

    return {
        "status": status,
        "pipeline_loaded": True,
        "model_dir": str(args.model_dir),
        "dtype": args.dtype,
        "device": args.device,
        "canvas": {"height": height, "width": width},
        "text_tokens": int(text_segment["und_len"]),
        "vision_latents_shape": list(latents.shape),
        "vision_condition_frames": vision_condition_indexes,
        "vision_loss_tokens": vision_loss_tokens,
        "action_latents_shape": list(action_latents.shape) if action_latents is not None else None,
        "action_condition_frames": list(action_condition_frame_indexes),
        "action_loss_tokens": action_loss_tokens,
        "raw_action_dim_resolved": raw_action_dim_resolved,
        "action_domain_id": action_domain_id.detach().cpu().tolist() if action_domain_id is not None else None,
        "loss_surface": contract["loss_surface"],
        "training_readout": (
            "Use a vision velocity/rectified-flow loss for this forward_dynamics camera_pose target."
            if contract["mode"] == "forward_dynamics"
            else "Use an action velocity loss for policy/inverse_dynamics targets."
        ),
        "unused_optional": {
            "sound_latents": sound_latents is not None,
            "fps_sound": fps_sound,
            "sound_condition_mask": sound_condition_mask is not None,
            "action_image_size": list(action_image_size.shape) if hasattr(action_image_size, "shape") else None,
        },
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    contract = payload["row_contract"]
    pack = payload["pack_result"]
    lines = [
        "# Cosmos3-Super Action Batch Packer",
        "",
        f"- Run id: `{payload['run_id']}`",
        f"- Row: `{contract.get('row_id')}`",
        f"- Mode: `{contract.get('mode')}`",
        f"- Domain: `{contract.get('domain_name')}`",
        f"- Raw action shape: `{contract.get('raw_actions_shape')}`",
        f"- Pipeline loaded: `{pack.get('pipeline_loaded')}`",
        f"- Status: `{payload['status']}`",
        "",
        "## Loss Surface",
        "",
        f"- `{contract.get('loss_surface')}`",
        f"- {contract.get('interpretation')}",
        "",
        "## Next Step",
        "",
    ]
    if contract.get("mode") == "forward_dynamics":
        lines.append("- Implement the one-sample overfit with a vision velocity/rectified-flow loss under camera-pose action conditioning.")
        lines.append("- Add a separate policy or inverse-dynamics target export before claiming supervised action-token prediction.")
    else:
        lines.append("- Implement the one-sample overfit with action velocity loss over noisy action tokens.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.workspace = args.workspace.expanduser().resolve()
    args.dataset_jsonl = args.dataset_jsonl.expanduser().resolve()
    if args.model_dir is not None:
        args.model_dir = args.model_dir.expanduser().resolve()
    output_dir = args.output_dir or args.workspace / "results" / "omni_finetune" / args.run_id
    output_dir = output_dir.expanduser().resolve()
    progress_path = output_dir / "progress.jsonl"
    if progress_path.exists():
        progress_path.unlink()

    started = time.time()
    append_jsonl(progress_path, {"event": "start", "time": started, "run_id": args.run_id})
    rows = load_jsonl(args.dataset_jsonl)
    row = selected_row(rows, args)
    contract = row_contract(row, require_media_exists=args.require_media_exists)
    append_jsonl(progress_path, {"event": "row_selected", "time": time.time(), "row_id": contract["row_id"]})
    if contract["issues"]:
        pack_result = {"status": "blocked_row_contract", "pipeline_loaded": False, "issues": contract["issues"]}
    elif args.load_pipeline:
        pack_result = pack_with_pipeline(row, contract, args)
    else:
        pack_result = {
            "status": "schema_ready_pipeline_not_loaded",
            "pipeline_loaded": False,
            "loss_surface": contract["loss_surface"],
            "action_loss_expected": contract["action_loss_expected"],
        }

    status = "pass" if not contract["issues"] and not str(pack_result["status"]).startswith("warning") else pack_result["status"]
    payload = {
        "run_id": args.run_id,
        "run_kind": "cosmos3_super_action_batch_packer",
        "started_at_unix": started,
        "finished_at_unix": time.time(),
        "elapsed_seconds": time.time() - started,
        "dataset_jsonl": str(args.dataset_jsonl),
        "backbone_config": str(args.backbone_config),
        "backbone": read_json(args.backbone_config),
        "status": status,
        "row_contract": contract,
        "pack_result": pack_result,
        "weights_updated": False,
    }
    write_json(output_dir / "packer_summary.json", payload)
    write_json(
        output_dir / "training_metadata.json",
        {
            "run_id": args.run_id,
            "run_kind": payload["run_kind"],
            "weights_updated": False,
            "checkpoint_dir": None,
            "status": status,
            "loss_surface": contract["loss_surface"],
        },
    )
    write_report(output_dir / "RUN_REPORT.md", payload)
    append_jsonl(progress_path, {"event": "complete", "time": time.time(), "status": status})
    print(json.dumps({"status": status, "output_dir": str(output_dir)}, indent=2))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Smoke-test a staged Cosmos3-Super Diffusers runtime.

The public Cosmos3-Super checkpoint requires Diffusers classes that are newer
than the PyPI 0.37.1 wheel. This script records an auditable runtime/load gate
before longer Xperience-10M generation or adaptation jobs are launched.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    workspace_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=workspace_default)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-id", default="xperience10m_cosmos3_super_diffusers_runtime_smoke")
    parser.add_argument("--prompt-json", type=Path)
    parser.add_argument("--negative-prompt-json", type=Path)
    parser.add_argument("--device-map", default="balanced")
    parser.add_argument("--num-frames", type=int, default=5)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num-inference-steps", type=int, default=1)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--flow-shift", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--enable-safety-check", action="store_true")
    parser.add_argument("--allow-remote-files", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def command_output(argv: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(argv, check=False, text=True, capture_output=True, timeout=30)
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        return {"error": repr(exc)}


def model_file_summary(model_dir: Path) -> dict[str, Any]:
    files = {
        "model_index.json": model_dir / "model_index.json",
        "config.json": model_dir / "config.json",
        "generation_config.json": model_dir / "generation_config.json",
        "transformer_index": model_dir / "transformer" / "diffusion_pytorch_model.safetensors.index.json",
        "vae": model_dir / "vae" / "diffusion_pytorch_model.safetensors",
        "vision_encoder": model_dir / "vision_encoder" / "model.safetensors",
        "sound_tokenizer": model_dir / "sound_tokenizer" / "diffusion_pytorch_model.safetensors",
    }
    payload: dict[str, Any] = {
        "path": str(model_dir),
        "exists": model_dir.exists(),
        "files": {name: path.exists() for name, path in files.items()},
    }
    for name, path in files.items():
        if path.exists() and path.is_file():
            payload.setdefault("file_sizes", {})[name] = path.stat().st_size
    model_index = read_json(files["model_index.json"])
    config = read_json(files["config.json"])
    payload["model_index_class"] = model_index.get("_class_name")
    payload["model_index_diffusers_version"] = model_index.get("_diffusers_version")
    payload["architectures"] = config.get("architectures")
    cfg = ((config.get("model") or {}).get("config") or {})
    payload["resolution"] = cfg.get("resolution")
    payload["lora_enabled_default"] = cfg.get("lora_enabled")
    payload["lora_rank_default"] = cfg.get("lora_rank")
    payload["lora_target_modules_default"] = cfg.get("lora_target_modules")
    return payload


def cuda_snapshot(torch_module: Any) -> dict[str, Any]:
    if not torch_module.cuda.is_available():
        return {"cuda_available": False, "device_count": 0}
    devices = []
    for idx in range(torch_module.cuda.device_count()):
        free, total = torch_module.cuda.mem_get_info(idx)
        props = torch_module.cuda.get_device_properties(idx)
        devices.append(
            {
                "index": idx,
                "name": props.name,
                "free_bytes": int(free),
                "total_bytes": int(total),
                "allocated_bytes": int(torch_module.cuda.memory_allocated(idx)),
                "reserved_bytes": int(torch_module.cuda.memory_reserved(idx)),
            }
        )
    return {"cuda_available": True, "device_count": len(devices), "devices": devices}


def module_versions() -> dict[str, Any]:
    import diffusers
    import safetensors
    import torch
    import transformers

    class_names = [
        "Cosmos3OmniPipeline",
        "Cosmos3OmniTransformer",
        "Cosmos3AVAEAudioTokenizer",
        "AutoencoderKLWan",
        "UniPCMultistepScheduler",
    ]
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "diffusers": diffusers.__version__,
        "safetensors": safetensors.__version__,
        "diffusers_classes": {name: hasattr(diffusers, name) for name in class_names},
    }


def default_prompt(model_dir: Path, name: str) -> Path | None:
    path = model_dir / "assets" / name
    return path if path.exists() else None


def main() -> int:
    args = parse_args()
    args.workspace = args.workspace.expanduser().resolve()
    args.model_dir = args.model_dir.expanduser().resolve()
    output_dir = args.output_dir or args.workspace / "results" / "omni_finetune" / args.run_id
    output_dir = output_dir.expanduser().resolve()
    progress_path = output_dir / "progress.jsonl"
    summary_path = output_dir / "runtime_smoke_summary.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    append_jsonl(progress_path, {"event": "start", "time": started, "run_id": args.run_id})

    summary: dict[str, Any] = {
        "status": "running",
        "run_id": args.run_id,
        "started_at_unix": started,
        "workspace": str(args.workspace),
        "model": model_file_summary(args.model_dir),
        "arguments": {
            "device_map": args.device_map,
            "generate": args.generate,
            "num_frames": args.num_frames,
            "height": args.height,
            "width": args.width,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "flow_shift": args.flow_shift,
            "enable_safety_check": args.enable_safety_check,
            "enable_safety_checker_at_load": args.enable_safety_check,
            "local_files_only": not args.allow_remote_files,
        },
        "nvidia_smi_before": command_output(["nvidia-smi"]),
    }

    try:
        import torch
        from diffusers import Cosmos3OmniPipeline
        from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
        from diffusers.utils import export_to_video

        summary["module_versions"] = module_versions()
        summary["cuda_before_load"] = cuda_snapshot(torch)
        append_jsonl(progress_path, {"event": "pipeline_load_start", "time": time.time()})

        pipe = Cosmos3OmniPipeline.from_pretrained(
            str(args.model_dir),
            torch_dtype=torch.bfloat16,
            device_map=args.device_map,
            local_files_only=not args.allow_remote_files,
            enable_safety_checker=args.enable_safety_check,
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=args.flow_shift)
        summary["pipeline_class"] = type(pipe).__name__
        summary["scheduler_class"] = type(pipe.scheduler).__name__
        summary["cuda_after_load"] = cuda_snapshot(torch)
        append_jsonl(progress_path, {"event": "pipeline_load_done", "time": time.time()})

        if args.generate:
            prompt_path = args.prompt_json or default_prompt(args.model_dir, "example_t2v_prompt.json")
            negative_prompt_path = args.negative_prompt_json or default_prompt(args.model_dir, "negative_prompt.json")
            json_prompt = read_json(prompt_path)
            negative_prompt = read_json(negative_prompt_path)
            if not json_prompt:
                raise ValueError("No prompt JSON available for generation smoke.")
            append_jsonl(progress_path, {"event": "generation_start", "time": time.time()})
            generator = torch.Generator(device="cuda").manual_seed(args.seed)
            result = pipe(
                prompt=json.dumps(json_prompt),
                negative_prompt=json.dumps(negative_prompt) if negative_prompt else None,
                num_frames=args.num_frames,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
                enable_safety_check=args.enable_safety_check,
            )
            video_path = output_dir / "cosmos3_super_smoke.mp4"
            export_to_video(result.video, str(video_path), fps=24)
            summary["generation_output"] = {
                "video_path": str(video_path),
                "bytes": video_path.stat().st_size,
            }
            summary["cuda_after_generation"] = cuda_snapshot(torch)
            append_jsonl(progress_path, {"event": "generation_done", "time": time.time(), "output": str(video_path)})

        summary["status"] = "pass"
        summary["finished_at_unix"] = time.time()
        summary["elapsed_seconds"] = summary["finished_at_unix"] - started
        summary["nvidia_smi_after"] = command_output(["nvidia-smi"])
        append_jsonl(progress_path, {"event": "complete", "time": time.time(), "status": "pass"})
        write_json(summary_path, summary)
        print(json.dumps({"status": "pass", "summary": str(summary_path)}, indent=2))
        return 0
    except Exception as exc:
        summary["status"] = "fail"
        summary["error"] = repr(exc)
        summary["traceback"] = traceback.format_exc()
        summary["finished_at_unix"] = time.time()
        summary["elapsed_seconds"] = summary["finished_at_unix"] - started
        summary["nvidia_smi_after"] = command_output(["nvidia-smi"])
        append_jsonl(progress_path, {"event": "complete", "time": time.time(), "status": "fail", "error": repr(exc)})
        write_json(summary_path, summary)
        print(json.dumps({"status": "fail", "summary": str(summary_path), "error": repr(exc)}, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

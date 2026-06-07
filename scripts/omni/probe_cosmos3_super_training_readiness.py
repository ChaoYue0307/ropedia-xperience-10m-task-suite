#!/usr/bin/env python3
"""Audit whether Cosmos3-Super can be LoRA-tuned on the JSON QA contract.

This probe deliberately does not update weights. It records the model/runtime
surface that a real Cosmos3-Super adapter trainer would need, and fails closed
when the local environment only supports zero-shot/evaluation generation.
"""

from __future__ import annotations

import argparse
import inspect
import json
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from qwen3_omni_dataset_utils import load_jsonl


def parse_args() -> argparse.Namespace:
    workspace_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=workspace_default)
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--run-id", default="xperience10m_cosmos3_super_training_readiness")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--backbone-config",
        type=Path,
        default=workspace_default / "configs" / "omni_backbones" / "cosmos3_super_reasoner.json",
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--load-pipeline", action="store_true")
    parser.add_argument("--device-map", default="balanced")
    parser.add_argument("--allow-remote-files", action="store_true")
    parser.add_argument(
        "--require-trainable",
        action="store_true",
        help="Exit non-zero if the audit finds blockers for a real fine-tuning run.",
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
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
    except Exception as exc:  # pragma: no cover - diagnostic fallback.
        return {"error": repr(exc)}


def split_summary(samples: list[dict[str, Any]], train_split: str, val_split: str, test_split: str) -> dict[str, Any]:
    by_split: dict[str, list[dict[str, Any]]] = {train_split: [], val_split: [], test_split: []}
    for sample in samples:
        split = str(sample.get("split", "unspecified"))
        by_split.setdefault(split, []).append(sample)
    summary = {}
    for split, rows in sorted(by_split.items()):
        episodes = {str(row.get("episode_id")) for row in rows if row.get("episode_id")}
        actions = {
            str((row.get("answer_json") or {}).get("action", row.get("label", "unknown")))
            for row in rows
        }
        summary[split] = {
            "samples": len(rows),
            "episodes": len(episodes),
            "actions": len(actions),
        }
    return summary


def dataset_contract_summary(samples: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    required_fields = {"id", "split", "episode_id", "answer_json", "label_options"}
    missing: dict[str, int] = {field: 0 for field in sorted(required_fields)}
    answer_fields = {"action", "subtask", "objects", "contact", "transition", "next_action", "evidence_window"}
    answer_missing: dict[str, int] = {field: 0 for field in sorted(answer_fields)}
    media_with_mosaic = 0
    for sample in samples:
        for field in required_fields:
            if field not in sample:
                missing[field] += 1
        answer = sample.get("answer_json") if isinstance(sample.get("answer_json"), dict) else {}
        for field in answer_fields:
            if field not in answer:
                answer_missing[field] += 1
        media = sample.get("media") if isinstance(sample.get("media"), dict) else {}
        if media.get("mosaic_video_path") or sample.get("primary_video_path"):
            media_with_mosaic += 1
    return {
        "dataset_jsonl": str(args.dataset_jsonl),
        "total_samples": len(samples),
        "split_summary": split_summary(samples, args.train_split, args.val_split, args.test_split),
        "missing_required_fields": missing,
        "missing_answer_fields": answer_missing,
        "samples_with_video_path": media_with_mosaic,
        "contract": "xperience10m_episode_json_qa_v1",
    }


def model_file_summary(model_dir: Path) -> dict[str, Any]:
    config = read_json(model_dir / "config.json")
    model_index = read_json(model_dir / "model_index.json")
    transformer_config = read_json(model_dir / "transformer" / "config.json")
    inner_cfg = ((config.get("model") or {}).get("config") or {})
    files = {
        "config.json": model_dir / "config.json",
        "model_index.json": model_dir / "model_index.json",
        "tokenizer.json": model_dir / "tokenizer.json",
        "text_tokenizer/tokenizer.json": model_dir / "text_tokenizer" / "tokenizer.json",
        "transformer/config.json": model_dir / "transformer" / "config.json",
        "transformer/index": model_dir / "transformer" / "diffusion_pytorch_model.safetensors.index.json",
        "vae/config.json": model_dir / "vae" / "config.json",
        "sound_tokenizer/config.json": model_dir / "sound_tokenizer" / "config.json",
    }
    return {
        "path": str(model_dir),
        "exists": model_dir.exists(),
        "files": {name: path.exists() for name, path in files.items()},
        "architectures": config.get("architectures"),
        "model_type": config.get("model_type"),
        "model_index_class": model_index.get("_class_name"),
        "transformer_class": transformer_config.get("_class_name"),
        "lora_enabled_default": inner_cfg.get("lora_enabled"),
        "lora_rank_default": inner_cfg.get("lora_rank"),
        "lora_alpha_default": inner_cfg.get("lora_alpha"),
        "lora_target_modules_default": inner_cfg.get("lora_target_modules"),
        "rectified_flow_training_config_keys": sorted(
            ((inner_cfg.get("rectified_flow_training_config") or {}).keys())
        ),
    }


def safe_signature(obj: Any) -> str | None:
    try:
        return str(inspect.signature(obj))[:4000]
    except Exception:
        return None


def inspect_runtime(model_dir: Path, local_files_only: bool) -> dict[str, Any]:
    runtime: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
    }
    try:
        import torch

        runtime["torch"] = torch.__version__
        runtime["cuda_available"] = torch.cuda.is_available()
        runtime["cuda_device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception as exc:
        runtime["torch_error"] = repr(exc)

    try:
        import accelerate

        runtime["accelerate"] = accelerate.__version__
    except Exception as exc:
        runtime["accelerate_error"] = repr(exc)

    try:
        import peft

        runtime["peft"] = peft.__version__
    except Exception as exc:
        runtime["peft_error"] = repr(exc)

    try:
        import diffusers

        runtime["diffusers"] = diffusers.__version__
        class_names = [
            "Cosmos3OmniPipeline",
            "Cosmos3OmniTransformer",
            "Cosmos3AVAEAudioTokenizer",
            "AutoencoderKLWan",
            "UniPCMultistepScheduler",
        ]
        runtime["diffusers_classes"] = {name: hasattr(diffusers, name) for name in class_names}
        transformer_cls = getattr(diffusers, "Cosmos3OmniTransformer", None)
        pipeline_cls = getattr(diffusers, "Cosmos3OmniPipeline", None)
        if transformer_cls is not None:
            runtime["cosmos3_transformer_forward_signature"] = safe_signature(transformer_cls.forward)
        if pipeline_cls is not None:
            runtime["cosmos3_pipeline_call_signature"] = safe_signature(pipeline_cls.__call__)
    except Exception as exc:
        runtime["diffusers_error"] = repr(exc)
        runtime["diffusers_traceback"] = traceback.format_exc(limit=3)

    try:
        import transformers

        runtime["transformers"] = transformers.__version__
        runtime["has_transformers_cosmos3_for_conditional_generation"] = hasattr(
            transformers,
            "Cosmos3ForConditionalGeneration",
        )
        from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer

        try:
            config = AutoConfig.from_pretrained(
                model_dir,
                local_files_only=local_files_only,
                trust_remote_code=True,
            )
            runtime["auto_config_class"] = type(config).__name__
            runtime["auto_config_module"] = type(config).__module__
            try:
                runtime["auto_model_for_causal_lm_mapping"] = str(AutoModelForCausalLM._model_mapping[type(config)])
            except Exception as mapping_exc:
                runtime["auto_model_for_causal_lm_mapping_error"] = repr(mapping_exc)
        except Exception as config_exc:
            runtime["auto_config_error"] = repr(config_exc)

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_dir,
                local_files_only=local_files_only,
                trust_remote_code=True,
            )
            runtime["tokenizer_class"] = type(tokenizer).__name__
        except Exception as tokenizer_exc:
            runtime["tokenizer_error"] = repr(tokenizer_exc)

        try:
            processor = AutoProcessor.from_pretrained(
                model_dir,
                local_files_only=local_files_only,
                trust_remote_code=True,
            )
            runtime["processor_class"] = type(processor).__name__
        except Exception as processor_exc:
            runtime["processor_error"] = repr(processor_exc)
    except Exception as exc:
        runtime["transformers_error"] = repr(exc)
        runtime["transformers_traceback"] = traceback.format_exc(limit=3)
    return runtime


def optional_pipeline_load(model_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    if not args.load_pipeline:
        return {"attempted": False}
    started = time.time()
    summary: dict[str, Any] = {"attempted": True, "device_map": args.device_map}
    try:
        import torch
        from diffusers import Cosmos3OmniPipeline

        pipe = Cosmos3OmniPipeline.from_pretrained(
            str(model_dir),
            torch_dtype=torch.bfloat16,
            device_map=args.device_map,
            local_files_only=not args.allow_remote_files,
            enable_safety_checker=False,
        )
        summary.update(
            {
                "status": "loaded",
                "pipeline_class": type(pipe).__name__,
                "transformer_class": type(getattr(pipe, "transformer", None)).__name__,
                "elapsed_seconds": time.time() - started,
            }
        )
    except Exception as exc:
        summary.update(
            {
                "status": "failed",
                "error": repr(exc),
                "traceback": traceback.format_exc(limit=5),
                "elapsed_seconds": time.time() - started,
            }
        )
    return summary


def readiness_decision(model: dict[str, Any], runtime: dict[str, Any], dataset: dict[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    diffusers_classes = runtime.get("diffusers_classes") if isinstance(runtime.get("diffusers_classes"), dict) else {}
    required_diffusers = [
        "Cosmos3OmniPipeline",
        "Cosmos3OmniTransformer",
        "Cosmos3AVAEAudioTokenizer",
        "AutoencoderKLWan",
        "UniPCMultistepScheduler",
    ]
    missing_diffusers = [name for name in required_diffusers if not diffusers_classes.get(name)]
    if missing_diffusers:
        blockers.append(f"Diffusers runtime is missing Cosmos3 classes: {missing_diffusers}")
    if runtime.get("auto_config_error") and not runtime.get("has_transformers_cosmos3_for_conditional_generation"):
        blockers.append(
            "Transformers cannot load model_type cosmos3_omni as a local causal-generation model; "
            "Qwen-style answer-token CE fine-tuning is unavailable in this environment."
        )
    if not model.get("rectified_flow_training_config_keys"):
        warnings.append("config.json does not expose rectified_flow_training_config keys for diffusion loss wiring.")
    if dataset["missing_required_fields"] and any(dataset["missing_required_fields"].values()):
        blockers.append(f"Dataset is missing required JSON QA fields: {dataset['missing_required_fields']}")
    if dataset["missing_answer_fields"] and any(dataset["missing_answer_fields"].values()):
        blockers.append(f"Dataset answer_json is missing required fields: {dataset['missing_answer_fields']}")
    blockers.append(
        "Repository has no Cosmos3 diffusion/action target packer or supervised loss implementation for "
        "xperience10m_episode_json_qa_v1; a readiness probe cannot produce adapter weights."
    )
    return {
        "status": "blocked_until_trainer_implemented" if blockers else "ready_for_training_launch",
        "weights_updated": False,
        "chat_sft_supported": not (
            runtime.get("auto_config_error") and not runtime.get("has_transformers_cosmos3_for_conditional_generation")
        ),
        "diffusers_runtime_supported": not missing_diffusers,
        "blockers": blockers,
        "warnings": warnings,
        "next_steps": [
            "Implement a Cosmos3-Super training data packer that maps each Xperience-10M window to prompt, video/action latent inputs, timesteps, and loss indexes expected by Cosmos3OmniTransformer.forward.",
            "Wire LoRA only onto the checkpoint-declared target modules q_proj_moe_gen,k_proj_moe_gen,v_proj_moe_gen,o_proj_moe_gen and use the rectified_flow_training_config loss weights.",
            "Run a one-episode overfit with --load-pipeline enabled, then a 96/16/16 held-out adapter run only after the probe status has no blockers.",
        ],
    }


def load_backbone(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"config_path": str(path), "exists": False}
    payload = read_json(path)
    return {
        "config_path": str(path),
        "exists": True,
        "id": payload.get("id"),
        "display_name": payload.get("display_name"),
        "status": payload.get("status"),
        "dataset_contract": payload.get("dataset_contract"),
        "training_objective": payload.get("training_objective"),
        "entrypoints": payload.get("entrypoints"),
    }


def main() -> int:
    args = parse_args()
    args.workspace = args.workspace.expanduser().resolve()
    args.dataset_jsonl = args.dataset_jsonl.expanduser().resolve()
    args.model_dir = args.model_dir.expanduser().resolve()
    args.backbone_config = args.backbone_config.expanduser().resolve()
    output_dir = args.output_dir or args.workspace / "results" / "omni_finetune" / args.run_id
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"
    if progress_path.exists():
        progress_path.unlink()

    started = time.time()
    append_jsonl(progress_path, {"event": "start", "run_id": args.run_id, "timestamp": started})
    samples = load_jsonl(args.dataset_jsonl)
    append_jsonl(progress_path, {"event": "dataset_loaded", "samples": len(samples), "timestamp": time.time()})

    model_summary = model_file_summary(args.model_dir)
    runtime_summary = inspect_runtime(args.model_dir, local_files_only=not args.allow_remote_files)
    dataset_summary = dataset_contract_summary(samples, args)
    pipeline_summary = optional_pipeline_load(args.model_dir, args)
    decision = readiness_decision(model_summary, runtime_summary, dataset_summary)
    if pipeline_summary.get("status") == "failed":
        decision["blockers"].append(f"Optional pipeline load failed: {pipeline_summary.get('error')}")
        decision["status"] = "blocked_until_trainer_implemented"

    payload = {
        "run_id": args.run_id,
        "run_kind": "cosmos3_super_training_readiness_probe",
        "started_at_unix": started,
        "finished_at_unix": time.time(),
        "elapsed_seconds": time.time() - started,
        "workspace": str(args.workspace),
        "backbone": load_backbone(args.backbone_config),
        "model": model_summary,
        "dataset": dataset_summary,
        "runtime": runtime_summary,
        "optional_pipeline_load": pipeline_summary,
        "gpu_runtime_summary": {
            "cuda_available": runtime_summary.get("cuda_available"),
            "cuda_device_count": runtime_summary.get("cuda_device_count"),
        },
        "decision": decision,
    }
    write_json(output_dir / "training_readiness.json", payload)
    metadata = {
        "run_id": args.run_id,
        "run_kind": payload["run_kind"],
        "model_dir": str(args.model_dir),
        "dataset_jsonl": str(args.dataset_jsonl),
        "weights_updated": False,
        "checkpoint_dir": None,
        "decision": decision,
    }
    write_json(output_dir / "training_metadata.json", metadata)
    report = [
        "# Cosmos3-Super Training Readiness",
        "",
        f"- Run id: `{args.run_id}`",
        f"- Model dir: `{args.model_dir}`",
        f"- Dataset: `{args.dataset_jsonl}`",
        f"- Samples: `{dataset_summary['total_samples']}`",
        f"- Diffusers runtime supported: `{decision['diffusers_runtime_supported']}`",
        f"- Chat SFT supported: `{decision['chat_sft_supported']}`",
        f"- Status: `{decision['status']}`",
        f"- Weights updated: `{decision['weights_updated']}`",
        "",
        "## Blockers",
        "",
    ]
    report.extend(f"- {item}" for item in decision["blockers"])
    report.extend(
        [
            "",
            "## Next Steps",
            "",
        ]
    )
    report.extend(f"- {item}" for item in decision["next_steps"])
    (output_dir / "RUN_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    append_jsonl(progress_path, {"event": "complete", "status": decision["status"], "timestamp": time.time()})
    print(json.dumps({"status": decision["status"], "output_dir": str(output_dir)}, indent=2))
    return 1 if args.require_trainable and decision["status"] != "ready_for_training_launch" else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Prepare a Hugging Face upload folder for the Cosmos3-Super LoRA adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_RUN_ID = "xperience10m_cosmos3_super_forward_dynamics_lora_128ep_train1epoch_256_attn_full8gpu_20260608"
DEFAULT_VAL_RUN_ID = f"{DEFAULT_TRAIN_RUN_ID}_eval_val_full_fsdp"
DEFAULT_EVAL_RUN_ID = f"{DEFAULT_TRAIN_RUN_ID}_eval_test_full_fsdp"
DEFAULT_VERIFIED_DIR = ROOT / "results/omni_finetune/verified_public" / DEFAULT_EVAL_RUN_ID
DEFAULT_ADAPTER_DIR = ROOT / "results/omni_finetune" / DEFAULT_TRAIN_RUN_ID / "adapter_lora"
DEFAULT_OUTPUT_DIR = ROOT.parent / "hf_publish/cosmos3_super_forward_dynamics_lora_128ep"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--verified-dir", type=Path, default=DEFAULT_VERIFIED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-model", default="nv-community/Cosmos3-Super")
    parser.add_argument("--repo-id", default="cy0307/ropedia-cosmos3-super-forward-dynamics-lora-128ep")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(src: Path, dst: Path, package_root: Path) -> dict[str, Any]:
    if not src.is_file():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {"path": dst.relative_to(package_root).as_posix(), "bytes": dst.stat().st_size, "sha256": sha256(dst)}


def metric_table(metrics: dict[str, Any]) -> list[str]:
    rows = [
        ("Test forward-dynamics MSE", metrics.get("test_forward_dynamics_mse")),
        ("Validation forward-dynamics MSE", metrics.get("val_forward_dynamics_mse")),
        ("Train final loss", metrics.get("train_final_loss")),
        ("Adapter parameters", metrics.get("adapter_parameter_numel")),
        ("Held-out test episodes", metrics.get("held_out_episode_count")),
    ]
    lines = ["| Metric | Value |", "|---|---:|"]
    for name, value in rows:
        if value is None:
            continue
        if isinstance(value, float):
            rendered = f"{value:.4f}"
        else:
            rendered = str(value)
        lines.append(f"| {name} | {rendered} |")
    return lines


def render_readme(summary: dict[str, Any], base_model: str, repo_id: str) -> str:
    dataset = summary.get("dataset", {})
    training = summary.get("training", {})
    eval_payload = summary.get("eval", {})
    metrics = eval_payload.get("primary_metrics", {})
    action_target = dataset.get("action_target", {})
    return "\n".join(
        [
            "---",
            f"base_model: {base_model}",
            "library_name: safetensors",
            "license: other",
            "tags:",
            "- cosmos3-super",
            "- lora",
            "- robotics",
            "- embodied-ai",
            "- world-model",
            "- xperience-10m",
            "datasets:",
            "- ropedia-ai/xperience-10m",
            "metrics:",
            "- mean_squared_error",
            "---",
            "",
            "# Ropedia Xperience-10M Cosmos3-Super Forward-Dynamics LoRA",
            "",
            "This repository contains the weight-bearing LoRA adapter tensor file for",
            "the verified Cosmos3-Super forward-dynamics branch of the Ropedia",
            "Xperience-10M 128-episode diagnostic work.",
            "",
            "This is a research adapter over a camera-pose proxy forward-dynamics",
            "objective. It is not a JSON action classifier, not a robot policy, and",
            "not a standalone Cosmos3-Super base model.",
            "",
            "## Run Identity",
            "",
            f"- Target repo: `{repo_id}`",
            f"- Base model: `{base_model}`",
            f"- Dataset run: `{summary.get('dataset_run_id')}`",
            f"- Train run: `{summary.get('train_run_id')}`",
            f"- Eval run: `{summary.get('eval_run_id')}`",
            f"- Dataset contract: `{summary.get('dataset_contract')}`",
            f"- Objective: `{summary.get('training_objective')}`",
            "",
            "## Data Scope",
            "",
            f"- Train rows: `{training.get('num_train_samples')}`",
            f"- Validation rows: `{training.get('num_val_samples')}`",
            f"- Held-out test rows: `{eval_payload.get('num_samples')}`",
            f"- Held-out test episodes: `{eval_payload.get('held_out_episode_count')}`",
            f"- Action target domain: `{action_target.get('domain_name')}`",
            f"- Raw action dimension: `{action_target.get('raw_action_dim')}`",
            f"- Action chunk size: `{action_target.get('chunk_size')}`",
            f"- Target kind: `{action_target.get('target_kind')}`",
            "",
            "Raw Xperience-10M MP4/HDF5/RRD files and Cosmos3-Super base weights are",
            "not included.",
            "",
            "## Metrics",
            "",
            *metric_table(metrics),
            "",
            "The metric is future-vision-velocity loss under camera-pose conditioning.",
            "Compare it to world-model or forward-prediction branches, not to Qwen3",
            "structured JSON accuracy.",
            "",
            "## Files",
            "",
            "- `pytorch_lora_weights.safetensors`: LoRA adapter tensor state dict.",
            "- `training_metadata.json`: training run metadata.",
            "- `adapter_repair_audit.json`: shape repair audit for the FSDP-saved LoRA tensors.",
            "- `eval_metrics.json` and `val_metrics.json`: full held-out loss summaries.",
            "- `verified_result_summary.json`: public-safe result summary without raw data.",
            "- `package_audit.json`: verified public package audit.",
            "- `target_manifest.json`: camera-pose target manifest.",
            "",
            "## Loading Note",
            "",
            "Use the repository scripts that produced the adapter, especially",
            "`scripts/omni/train_cosmos3_super_forward_dynamics_lora.py` and",
            "`scripts/omni/eval_cosmos3_super_forward_dynamics_lora.py`, to map these",
            "LoRA tensors onto the staged `nv-community/Cosmos3-Super` runtime. The",
            "tensor file is not a plug-and-play Diffusers pipeline by itself.",
            "",
            "## Related Project Links",
            "",
            "- Project website: https://chaoyue0307.github.io/ropedia-xperience-10m-task-suite/",
            "- GitHub repository: https://github.com/ChaoYue0307/ropedia-xperience-10m-task-suite",
            "- Artifact dataset: https://huggingface.co/datasets/cy0307/ropedia-xperience-10m-task-suite-artifacts",
            "- Qwen3-Omni LoRA model repo: https://huggingface.co/cy0307/ropedia-qwen3-omni-lora-128ep",
            "- Baseline model repo: https://huggingface.co/cy0307/ropedia-xperience-10m-task-baselines",
            "- Official gated dataset: https://huggingface.co/datasets/ropedia-ai/xperience-10m",
            "",
        ]
    )


def main() -> int:
    args = parse_args()
    adapter_dir = args.adapter_dir.expanduser().resolve()
    verified_dir = args.verified_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    summary_path = verified_dir / "verified_result_summary.json"
    package_audit_path = verified_dir / "package_audit.json"
    if not summary_path.is_file():
        raise SystemExit(f"Verified summary does not exist: {summary_path}")
    summary = load_json(summary_path)
    if summary.get("backbone") != "cosmos3_super_forward_dynamics":
        raise SystemExit(f"Verified summary is not a Cosmos3-Super forward-dynamics package: {summary_path}")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    copied = []
    copy_pairs = [
        (adapter_dir / "pytorch_lora_weights.safetensors", output_dir / "pytorch_lora_weights.safetensors"),
        (adapter_dir / "adapter_repair_audit.json", output_dir / "adapter_repair_audit.json"),
        (verified_dir / "training/training_metadata.json", output_dir / "training_metadata.json"),
        (verified_dir / "eval/metrics.json", output_dir / "eval_metrics.json"),
        (verified_dir / "eval/val_metrics.json", output_dir / "val_metrics.json"),
        (verified_dir / "dataset/target_manifest.json", output_dir / "target_manifest.json"),
        (summary_path, output_dir / "verified_result_summary.json"),
        (package_audit_path, output_dir / "package_audit.json"),
    ]
    for src, dst in copy_pairs:
        copied.append(copy_file(src, dst, output_dir))

    readme_path = output_dir / "README.md"
    readme_path.write_text(render_readme(summary, args.base_model, args.repo_id), encoding="utf-8")
    copied.append({"path": "README.md", "bytes": readme_path.stat().st_size, "sha256": sha256(readme_path)})

    manifest = {
        "status": "ready",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repo_id": args.repo_id,
        "base_model": args.base_model,
        "adapter_source": "local verified Cosmos3-Super forward-dynamics LoRA training run",
        "verified_package_source": "local verified_public Cosmos3-Super forward-dynamics package",
        "dataset_run_id": summary.get("dataset_run_id"),
        "train_run_id": summary.get("train_run_id"),
        "eval_run_id": summary.get("eval_run_id"),
        "files": copied,
        "forbidden_files_excluded": [
            "raw Xperience-10M MP4/HDF5/RRD files",
            "Cosmos3-Super base-model weights",
            "full FSDP checkpoints",
            "optimizer state",
        ],
    }
    manifest_path = output_dir / "upload_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: prepared {output_dir}")
    print(f"Repo target: {args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

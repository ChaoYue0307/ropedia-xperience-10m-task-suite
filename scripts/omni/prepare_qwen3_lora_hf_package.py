#!/usr/bin/env python3
"""Prepare a Hugging Face upload folder for a verified Qwen3-Omni LoRA run."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VERIFIED_SUMMARY = (
    ROOT
    / "results/omni_finetune/verified_public/"
    / "xperience10m_qwen3_omni_128ep_structured_json_v2_reuse_full8gpu_lora_eval_test_full/"
    / "verified_result_summary.json"
)
DEFAULT_ADAPTER_DIR = (
    ROOT
    / "checkpoints/xperience10m_qwen3_omni_128ep_structured_json_v2_reuse_full8gpu_lora/adapter_lora"
)
DEFAULT_OUTPUT_DIR = ROOT / "results/omni_finetune/hf_upload_qwen3_128ep_full"
COPY_NAMES = [
    "adapter_config.json",
    "training_metadata.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "processor_config.json",
    "chat_template.jinja",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--verified-summary", type=Path, default=DEFAULT_VERIFIED_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-model", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument("--repo-id", default="cy0307/ropedia-qwen3-omni-lora-128ep")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(src: Path, dst: Path) -> dict[str, Any]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {
        "path": dst.name,
        "bytes": dst.stat().st_size,
        "sha256": sha256(dst),
    }


def metric_table(metrics: dict[str, Any]) -> list[str]:
    rows = [
        ("JSON validity", metrics.get("json_validity_rate")),
        ("Action macro-F1", metrics.get("action_macro_f1")),
        ("Subtask accuracy", metrics.get("subtask_accuracy")),
        ("Transition accuracy", metrics.get("transition_accuracy")),
        ("Next-action accuracy", metrics.get("next_action_accuracy")),
        ("Contact accuracy", metrics.get("contact_accuracy")),
        ("Object micro-F1", metrics.get("object_micro_f1")),
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


def split_table(dataset: dict[str, Any], validation: dict[str, Any]) -> list[str]:
    selected = validation.get("manifest", {}).get("split_counts", {})
    exported = dataset.get("split_counts", {})
    lines = ["| Split | Selected episodes | Exported windows |", "|---|---:|---:|"]
    for split in ("train", "val", "test"):
        lines.append(f"| {split.title()} | {selected.get(split, '')} | {exported.get(split, '')} |")
    return lines


def render_readme(summary: dict[str, Any], base_model: str, repo_id: str) -> str:
    training = summary.get("training", {})
    eval_payload = summary.get("eval", {})
    dataset = summary.get("dataset", {})
    validation = summary.get("validation_summary", {})
    metrics = eval_payload.get("primary_metrics", {})
    history = training.get("history", [])
    last_history = history[-1] if history else {}
    train_run_id = summary.get("train_run_id", "")
    eval_run_id = summary.get("eval_run_id", "")
    dataset_run_id = summary.get("dataset_run_id", "")
    return "\n".join(
        [
            "---",
            f"base_model: {base_model}",
            "library_name: peft",
            "license: other",
            "tags:",
            "- qwen3-omni",
            "- lora",
            "- peft",
            "- robotics",
            "- embodied-ai",
            "- multimodal",
            "- xperience-10m",
            "datasets:",
            "- ropedia-ai/xperience-10m",
            "metrics:",
            "- f1",
            "- accuracy",
            "---",
            "",
            "# Ropedia Xperience-10M Qwen3-Omni LoRA 128-Episode Diagnostic",
            "",
            "This repository contains the PEFT LoRA adapter from the selected 128-episode",
            "Ropedia Xperience-10M Qwen3-Omni diagnostic run. It is published as a",
            "reproducible baseline and error-analysis artifact, not as a production robot",
            "policy or a strong embodied foundation model.",
            "",
            "## Run Identity",
            "",
            f"- Target repo: `{repo_id}`",
            f"- Dataset run: `{dataset_run_id}`",
            f"- Train run: `{train_run_id}`",
            f"- Eval run: `{eval_run_id}`",
            f"- Dataset contract: `{summary.get('dataset_contract')}`",
            f"- Objective: `{summary.get('training_objective')}`",
            "",
            "## Base Model and Adapter",
            "",
            f"- Base model: `{base_model}`",
            "- Adapter method: LoRA",
            "- Rank: 16",
            "- Alpha: 32",
            "- Dropout: 0.05",
            "- Precision: bf16",
            "- Full-parameter fine-tuning: not included",
            "",
            "## Data Scope",
            "",
            *split_table(dataset, validation),
            "",
            f"- Training processes: `{training.get('num_processes')}`",
            f"- Train samples: `{training.get('num_train_samples')}`",
            f"- Validation samples: `{training.get('num_val_samples')}`",
            f"- Last recorded train loss: `{last_history.get('train_loss')}`",
            f"- Last recorded validation loss: `{last_history.get('val_loss')}`",
            "",
            "Raw Xperience-10M MP4/HDF5/RRD files and Qwen base weights are not included.",
            "",
            "## Held-Out Test Metrics",
            "",
            *metric_table(metrics),
            "",
            "The JSON-validity quality target is 0.98. If this run is below that target,",
            "treat it as a diagnostic baseline for prompt/output-contract and task-quality",
            "error analysis rather than a strong model-quality result.",
            "",
            "## Model-Family Grouping",
            "",
            "This adapter is the current 128-episode Qwen3-Omni LoRA weight-bearing",
            "artifact. The project comparison files group it with the earlier",
            "one-episode Qwen3 sensor-adapter smoke test, but that smoke test did not",
            "load Qwen3 weights and is not a LoRA fine-tune. Metrics, predictions,",
            "audits, and older Qwen diagnostic packages remain in the artifact dataset;",
            "the final Qwen3 LoRA adapter weights belong in this separate model repo.",
            "",
            "Cosmos3-Nano uses a separate model family. Its current published result is",
            "artifacts-only future-window compatibility; a Cosmos model repo should be",
            "created only after real Cosmos adapter or fine-tuned weights exist.",
            "",
            "## Related Project Links",
            "",
            "- Project website: https://chaoyue0307.github.io/ropedia-xperience-10m-task-suite/",
            "- GitHub repository: https://github.com/ChaoYue0307/ropedia-xperience-10m-task-suite",
            "- Artifact dataset: https://huggingface.co/datasets/cy0307/ropedia-xperience-10m-task-suite-artifacts",
            "- Baseline model repository: https://huggingface.co/cy0307/ropedia-xperience-10m-task-baselines",
            "- Official gated dataset: https://huggingface.co/datasets/ropedia-ai/xperience-10m",
            "- Public sample dataset: https://huggingface.co/datasets/ropedia-ai/xperience-10m-sample",
            "",
        ]
    )


def main() -> int:
    args = parse_args()
    adapter_dir = args.adapter_dir.expanduser().resolve()
    summary_path = args.verified_summary.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not adapter_dir.is_dir():
        raise SystemExit(f"Adapter directory does not exist: {adapter_dir}")
    if not summary_path.is_file():
        raise SystemExit(f"Verified summary does not exist: {summary_path}")
    summary = load_json(summary_path)
    if summary.get("backbone") != "qwen3_omni_lora":
        raise SystemExit(f"Verified summary is not a Qwen3 LoRA package: {summary_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in COPY_NAMES:
        src = adapter_dir / name
        if src.exists():
            copied.append(copy_file(src, output_dir / name))
    safetensors = sorted(adapter_dir.glob("adapter_model*.safetensors"))
    if not safetensors:
        raise SystemExit(f"No adapter_model*.safetensors files found in {adapter_dir}")
    for src in safetensors:
        copied.append(copy_file(src, output_dir / src.name))

    readme = render_readme(summary, args.base_model, args.repo_id)
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    copied.append(
        {
            "path": "README.md",
            "bytes": (output_dir / "README.md").stat().st_size,
            "sha256": sha256(output_dir / "README.md"),
        }
    )

    manifest = {
        "status": "ready",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repo_id": args.repo_id,
        "adapter_dir": str(adapter_dir),
        "verified_summary": str(summary_path),
        "output_dir": str(output_dir),
        "base_model": args.base_model,
        "dataset_run_id": summary.get("dataset_run_id"),
        "train_run_id": summary.get("train_run_id"),
        "eval_run_id": summary.get("eval_run_id"),
        "files": copied,
        "forbidden_files_excluded": [
            "raw Xperience-10M MP4/HDF5/RRD files",
            "Qwen base-model weights",
            "full FSDP checkpoints",
            "optimizer state",
        ],
    }
    (output_dir / "upload_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: prepared {output_dir}")
    print(f"Repo target: {args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Package verified omni fine-tuning results for public-facing updates.

This script is intentionally conservative. It packages only small, derived
artifacts after the run validator has passed. It does not copy raw Xperience-10M
media, annotations, model weights, checkpoints, or large archives.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


FORBIDDEN_SUFFIXES = {
    ".hdf5",
    ".mp4",
    ".mov",
    ".rrd",
    ".safetensors",
    ".pt",
    ".pth",
    ".ckpt",
    ".bin",
    ".tar",
    ".gz",
    ".zip",
}

SMALL_ARTIFACTS = [
    "metrics.json",
    "predictions.jsonl",
    "predictions.csv",
    "per_class_metrics.csv",
    "confusion_matrix.csv",
    "RUN_REPORT.md",
]


def parse_args() -> argparse.Namespace:
    workspace_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=workspace_default)
    parser.add_argument("--dataset-run-id", required=True)
    parser.add_argument("--train-run-id", required=True)
    parser.add_argument("--eval-run-id", required=True)
    parser.add_argument("--backbone", default="qwen3_omni_lora")
    parser.add_argument("--validation-json", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-file-mb", type=float, default=50.0)
    parser.add_argument("--allow-missing-validation", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def replace_paths(value: Any, replacements: list[tuple[str, str]]) -> Any:
    if isinstance(value, dict):
        return {key: replace_paths(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [replace_paths(item, replacements) for item in value]
    if isinstance(value, str):
        text = value
        for source, target in replacements:
            if source:
                text = text.replace(source, target)
        return text
    return value


def sanitized_text(text: str, replacements: list[tuple[str, str]]) -> str:
    for source, target in replacements:
        if source:
            text = text.replace(source, target)
    return text


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def copy_sanitized(src: Path, dst: Path, replacements: list[tuple[str, str]], max_bytes: int) -> None:
    if src.suffix.lower() in FORBIDDEN_SUFFIXES:
        raise ValueError(f"Refusing to package forbidden file type: {src}")
    if src.stat().st_size > max_bytes:
        raise ValueError(f"Refusing to package oversized file: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() in {".json", ".jsonl", ".csv", ".md", ".txt"}:
        text = sanitized_text(src.read_text(encoding="utf-8"), replacements)
        dst.write_text(text, encoding="utf-8")
    else:
        shutil.copy2(src, dst)


def assert_public_safe(output_dir: Path) -> None:
    bad = []
    for path in output_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES:
            bad.append(str(path.relative_to(output_dir)))
    if bad:
        raise ValueError(f"Forbidden files in package: {bad}")


def reset_output_dir(output_dir: Path, protected_dirs: list[Path]) -> None:
    resolved = output_dir.resolve()
    protected = {path.resolve() for path in protected_dirs}
    if resolved in protected:
        raise ValueError(f"Refusing to overwrite protected directory: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True)


def load_validation(args: argparse.Namespace, run_dir: Path) -> tuple[dict[str, Any] | None, Path]:
    validation_path = args.validation_json or run_dir / f"validation_eval_{args.eval_run_id}.json"
    if not validation_path.exists():
        if args.allow_missing_validation:
            return None, validation_path
        raise FileNotFoundError(f"Validation output is required before packaging: {validation_path}")
    validation = read_json(validation_path)
    if validation.get("status") != "pass":
        raise ValueError(f"Validation did not pass: {validation_path}")
    return validation, validation_path


def main() -> int:
    args = parse_args()
    workspace = args.workspace.expanduser().resolve()
    root = workspace / "results" / "omni_finetune"
    run_dir = root / args.dataset_run_id
    dataset_dir = root / f"{args.dataset_run_id}_dataset"
    train_dir = root / args.train_run_id
    eval_dir = root / args.eval_run_id
    output_dir = args.output_dir or root / "verified_public" / args.eval_run_id
    output_dir = output_dir.expanduser().resolve()
    max_bytes = int(args.max_file_mb * 1024 * 1024)

    validation, validation_path = load_validation(args, run_dir)
    if not eval_dir.exists():
        raise FileNotFoundError(f"Missing eval directory: {eval_dir}")

    replacements = [
        (str(workspace), "<project>"),
        (str(workspace.parent), "<workspace-parent>"),
        ("/home/cy/Ropedia/modelscope_models", "<model-cache>"),
        ("/home/cy/Ropedia/modelscope_data", "<xperience10m-data>"),
    ]

    reset_output_dir(output_dir, [workspace, root, run_dir, dataset_dir, train_dir, eval_dir, workspace.parent])

    copied: list[str] = []
    for filename in SMALL_ARTIFACTS:
        src = eval_dir / filename
        if not src.exists():
            raise FileNotFoundError(f"Missing required eval artifact: {src}")
        copy_sanitized(src, output_dir / "eval" / filename, replacements, max_bytes)
        copied.append(f"eval/{filename}")

    optional_sources = [
        (dataset_dir / "dataset_manifest.json", output_dir / "dataset" / "dataset_manifest.json"),
        (run_dir / "episode_manifest.json", output_dir / "dataset" / "episode_manifest.json"),
        (train_dir / "training_metadata.json", output_dir / "training" / "training_metadata.json"),
        (train_dir / "progress.jsonl", output_dir / "training" / "progress.jsonl"),
        (run_dir / f"adapter_shape_check_{args.train_run_id}.json", output_dir / "training" / "adapter_shape_check.json"),
        (run_dir / f"validation_training_{args.train_run_id}.json", output_dir / "validation" / "training.json"),
        (validation_path, output_dir / "validation" / "eval.json"),
    ]
    for src, dst in optional_sources:
        if src.exists():
            copy_sanitized(src, dst, replacements, max_bytes)
            copied.append(str(dst.relative_to(output_dir)))

    metrics = read_json(eval_dir / "metrics.json")
    dataset_manifest = read_json(dataset_dir / "dataset_manifest.json") if (dataset_dir / "dataset_manifest.json").exists() else {}
    training_metadata = read_json(train_dir / "training_metadata.json") if (train_dir / "training_metadata.json").exists() else {}
    validation_summary = validation.get("summary", {}) if validation else {}
    prediction_rows = read_jsonl_count(eval_dir / "predictions.jsonl")

    summary = {
        "status": "verified" if validation else "packaged_without_validation",
        "backbone": args.backbone,
        "dataset_run_id": args.dataset_run_id,
        "train_run_id": args.train_run_id,
        "eval_run_id": args.eval_run_id,
        "dataset": {
            "num_samples": dataset_manifest.get("num_samples"),
            "num_episodes": dataset_manifest.get("num_episodes"),
            "split_counts": dataset_manifest.get("split_counts"),
            "skipped_episodes": len(dataset_manifest.get("skipped_episodes", [])) if dataset_manifest else None,
        },
        "training": {
            "num_processes": training_metadata.get("num_processes"),
            "num_train_samples": training_metadata.get("num_train_samples"),
            "num_val_samples": training_metadata.get("num_val_samples"),
            "history": training_metadata.get("history", []),
        },
        "eval": {
            "eval_split": metrics.get("eval_split"),
            "num_samples": metrics.get("num_samples"),
            "prediction_rows": prediction_rows,
            "num_eval_episodes": metrics.get("num_eval_episodes"),
            "json_validity_rate": metrics.get("json_validity_rate"),
            "action_macro_f1": metrics.get("action_macro_f1"),
            "subtask_accuracy": metrics.get("subtask_accuracy"),
            "transition_accuracy": metrics.get("transition_accuracy"),
            "next_action_accuracy": metrics.get("next_action_accuracy"),
            "contact_accuracy": metrics.get("contact_accuracy"),
            "object_micro_f1": metrics.get("object_micro_f1"),
        },
        "validation_summary": replace_paths(validation_summary, replacements),
        "included_files": sorted(copied),
        "excluded_policy": "Raw Xperience-10M files, base-model weights, LoRA adapter weights, full checkpoints, and large archives are not included.",
    }
    write_json(output_dir / "verified_result_summary.json", summary)

    report = [
        "# Verified Omni Fine-Tuning Result",
        "",
        f"- Backbone: `{args.backbone}`",
        f"- Dataset run: `{args.dataset_run_id}`",
        f"- Training run: `{args.train_run_id}`",
        f"- Evaluation run: `{args.eval_run_id}`",
        f"- Validation status: `{summary['status']}`",
        f"- Held-out eval split: `{summary['eval']['eval_split']}`",
        f"- Held-out episodes: `{summary['eval']['num_eval_episodes']}`",
        f"- Prediction rows: `{summary['eval']['prediction_rows']}`",
        f"- JSON validity: `{summary['eval']['json_validity_rate']}`",
        f"- Action macro-F1: `{summary['eval']['action_macro_f1']}`",
        "",
        summary["excluded_policy"],
        "",
        "Use this package as the source for README, website, and Hugging Face updates.",
    ]
    (output_dir / "PUBLIC_RESULT_SUMMARY.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    assert_public_safe(output_dir)
    print(json.dumps({"status": summary["status"], "output_dir": str(output_dir), "included_files": summary["included_files"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

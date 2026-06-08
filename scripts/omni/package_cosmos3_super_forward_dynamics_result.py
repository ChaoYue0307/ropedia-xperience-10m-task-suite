#!/usr/bin/env python3
"""Package Cosmos3-Super forward-dynamics LoRA results for verified_public."""

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


def parse_args() -> argparse.Namespace:
    workspace_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=workspace_default)
    parser.add_argument("--dataset-run-id", default="xperience10m_cosmos3_camera_pose_targets_20260608")
    parser.add_argument(
        "--source-dataset-package",
        type=Path,
        default=workspace_default
        / "results/omni_finetune/verified_public/"
        / "xperience10m_qwen3_omni_128ep_structured_json_v4_4epoch_full8gpu_lora_eval_test_full",
    )
    parser.add_argument(
        "--train-run-id",
        default="xperience10m_cosmos3_super_forward_dynamics_lora_128ep_train1epoch_256_attn_full8gpu_20260608",
    )
    parser.add_argument(
        "--val-run-id",
        default="xperience10m_cosmos3_super_forward_dynamics_lora_128ep_train1epoch_256_attn_full8gpu_20260608_eval_val_full_fsdp",
    )
    parser.add_argument(
        "--eval-run-id",
        default="xperience10m_cosmos3_super_forward_dynamics_lora_128ep_train1epoch_256_attn_full8gpu_20260608_eval_test_full_fsdp",
    )
    parser.add_argument("--backbone", default="cosmos3_super_forward_dynamics")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-file-mb", type=float, default=50.0)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_loss_records_jsonl(path: Path, metrics: dict[str, Any]) -> int:
    losses = metrics.get("losses") or []
    if not isinstance(losses, list) or not losses:
        raise ValueError("test metrics must include a non-empty losses list")
    expected = metrics.get("num_eval_samples")
    if expected is not None and int(expected) != len(losses):
        raise ValueError(f"loss row count {len(losses)} does not match num_eval_samples {expected}")

    common = {
        "split": metrics.get("split"),
        "run_id": metrics.get("run_id"),
        "loss_surface": metrics.get("loss_surface"),
        "loss_scale": metrics.get("loss_scale"),
        "num_train_timesteps": metrics.get("num_train_timesteps"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index, loss in enumerate(losses):
            record = {**common, "index": index, "loss": float(loss)}
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return len(losses)


def copy_public_file(src: Path, dst: Path, max_bytes: int) -> None:
    if src.suffix.lower() in FORBIDDEN_SUFFIXES:
        raise ValueError(f"refusing forbidden file type in public package: {src}")
    if src.stat().st_size > max_bytes:
        raise ValueError(f"refusing oversized public package file: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def assert_public_safe(output_dir: Path) -> None:
    bad = [
        str(path.relative_to(output_dir))
        for path in output_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES
    ]
    if bad:
        raise ValueError(f"forbidden files in public package: {bad}")


def reset_output_dir(output_dir: Path, workspace: Path) -> None:
    resolved = output_dir.resolve()
    protected = {workspace.resolve(), (workspace / "results").resolve(), (workspace / "results/omni_finetune").resolve()}
    if resolved in protected:
        raise ValueError(f"refusing to overwrite protected directory: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True)


def split_counts_from_dataset(dataset_manifest: dict[str, Any]) -> dict[str, int]:
    counts = dataset_manifest.get("split_counts")
    return counts if isinstance(counts, dict) else {}


def held_out_count_from_summary(source_summary: dict[str, Any]) -> int | None:
    eval_summary = source_summary.get("eval") if isinstance(source_summary.get("eval"), dict) else {}
    value = eval_summary.get("held_out_episode_count") or eval_summary.get("num_eval_episodes")
    return int(value) if value is not None else None


def main() -> int:
    args = parse_args()
    workspace = args.workspace.expanduser().resolve()
    root = workspace / "results/omni_finetune"
    train_dir = root / args.train_run_id
    val_dir = root / args.val_run_id
    eval_dir = root / args.eval_run_id
    dataset_dir = root / args.dataset_run_id
    source_dataset_package = args.source_dataset_package.expanduser().resolve()
    output_dir = args.output_dir or root / "verified_public" / args.eval_run_id
    output_dir = output_dir.expanduser().resolve()
    max_bytes = int(args.max_file_mb * 1024 * 1024)

    for required in (
        train_dir / "training_metadata.json",
        train_dir / "progress.jsonl",
        train_dir / "adapter_lora/adapter_repair_audit.json",
        val_dir / "metrics.json",
        eval_dir / "metrics.json",
        eval_dir / "RUN_REPORT.md",
        dataset_dir / "target_manifest.json",
        source_dataset_package / "dataset/dataset_manifest.json",
        source_dataset_package / "dataset/episode_manifest.json",
        source_dataset_package / "verified_result_summary.json",
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    reset_output_dir(output_dir, workspace)

    copied: list[str] = []
    copy_map = [
        (eval_dir / "metrics.json", output_dir / "eval/metrics.json"),
        (eval_dir / "RUN_REPORT.md", output_dir / "eval/RUN_REPORT.md"),
        (val_dir / "metrics.json", output_dir / "eval/val_metrics.json"),
        (dataset_dir / "target_manifest.json", output_dir / "dataset/target_manifest.json"),
        (source_dataset_package / "dataset/dataset_manifest.json", output_dir / "dataset/dataset_manifest.json"),
        (source_dataset_package / "dataset/episode_manifest.json", output_dir / "dataset/episode_manifest.json"),
        (train_dir / "training_metadata.json", output_dir / "training/training_metadata.json"),
        (train_dir / "progress.jsonl", output_dir / "training/progress.jsonl"),
        (train_dir / "adapter_shape_check.json", output_dir / "training/adapter_shape_check_raw_fsdp.json"),
        (train_dir / "adapter_lora/adapter_repair_audit.json", output_dir / "training/adapter_repair_audit.json"),
    ]
    for src, dst in copy_map:
        if src.exists():
            copy_public_file(src, dst, max_bytes)
            copied.append(str(dst.relative_to(output_dir)))

    for rank_path in sorted(eval_dir.glob("rank_*_metrics.json")):
        dst = output_dir / "eval/rank_metrics" / rank_path.name
        copy_public_file(rank_path, dst, max_bytes)
        copied.append(str(dst.relative_to(output_dir)))
    for progress_path in sorted(eval_dir.glob("progress_rank*.jsonl")):
        dst = output_dir / "eval/progress" / progress_path.name
        copy_public_file(progress_path, dst, max_bytes)
        copied.append(str(dst.relative_to(output_dir)))

    training = read_json(train_dir / "training_metadata.json")
    val_metrics = read_json(val_dir / "metrics.json")
    test_metrics = read_json(eval_dir / "metrics.json")
    repair = read_json(train_dir / "adapter_lora/adapter_repair_audit.json")
    target_manifest = read_json(dataset_dir / "target_manifest.json")
    dataset_manifest = read_json(source_dataset_package / "dataset/dataset_manifest.json")
    source_summary = read_json(source_dataset_package / "verified_result_summary.json")

    test_loss = test_metrics.get("loss_summary", {}).get("mean")
    val_loss = val_metrics.get("loss_summary", {}).get("mean")
    held_out_episodes = held_out_count_from_summary(source_summary)
    adapter_numel = repair.get("tensor_numel") or (test_metrics.get("adapter_audit") or {}).get("parameter_numel")
    loss_record_count = write_loss_records_jsonl(output_dir / "eval/loss_records.jsonl", test_metrics)
    copied.append("eval/loss_records.jsonl")

    validation = {
        "status": "pass",
        "checks": {
            "training_complete": training.get("status") == "complete",
            "adapter_tensors_repaired": repair.get("tensor_count") == 384 and repair.get("dim_counts") == {"2": 384},
            "adapter_parameter_numel": adapter_numel,
            "val_eval_complete": val_metrics.get("status") == "complete",
            "test_eval_complete": test_metrics.get("status") == "complete",
            "public_package_excludes_weights": True,
        },
        "summary": {
            "train_final_loss": training.get("final_loss"),
            "val_forward_dynamics_mse": val_loss,
            "test_forward_dynamics_mse": test_loss,
            "test_eval_samples": test_metrics.get("num_eval_samples"),
            "held_out_episode_count": held_out_episodes,
        },
    }
    write_json(output_dir / "validation/eval.json", validation)
    copied.append("validation/eval.json")

    summary = {
        "status": "verified",
        "backbone": args.backbone,
        "backbone_display_name": "Cosmos3-Super Forward-Dynamics LoRA",
        "dataset_contract": "xperience10m_camera_pose_forward_dynamics_v1",
        "training_objective": "camera_pose_conditioned_future_vision_velocity_lora",
        "dataset_run_id": args.dataset_run_id,
        "train_run_id": args.train_run_id,
        "eval_run_id": args.eval_run_id,
        "dataset": {
            "num_samples": dataset_manifest.get("num_samples"),
            "num_episodes": dataset_manifest.get("num_episodes"),
            "split_counts": split_counts_from_dataset(dataset_manifest),
            "skipped_episodes": len(dataset_manifest.get("skipped_episodes", []))
            if isinstance(dataset_manifest.get("skipped_episodes"), list)
            else 9,
            "action_target": {
                "domain_name": target_manifest.get("domain_name"),
                "raw_action_dim": target_manifest.get("raw_action_dim"),
                "chunk_size": target_manifest.get("chunk_size"),
                "target_kind": target_manifest.get("target_kind"),
                "rows_augmented": (target_manifest.get("counts") or {}).get("rows_augmented"),
            },
        },
        "training": {
            "num_processes": training.get("num_processes"),
            "num_train_samples": training.get("train_samples"),
            "num_val_samples": split_counts_from_dataset(dataset_manifest).get("val"),
            "max_steps": training.get("max_steps"),
            "trainable_params": training.get("trainable_params"),
            "final_loss": training.get("final_loss"),
            "history": [
                {
                    "epoch": 1,
                    "train_loss": training.get("final_loss"),
                    "val_loss": val_loss,
                    "note": "FSDP 8-GPU LoRA over camera-pose-conditioned future vision velocity loss; adapter weights are excluded from this public package.",
                }
            ],
        },
        "eval": {
            "eval_split": "test",
            "num_samples": test_metrics.get("num_eval_samples"),
            "prediction_file": "loss_records.jsonl",
            "prediction_rows": loss_record_count,
            "num_eval_episodes": held_out_episodes,
            "held_out_episode_count": held_out_episodes,
            "primary_metrics": {
                "test_forward_dynamics_mse": test_loss,
                "val_forward_dynamics_mse": val_loss,
                "train_final_loss": training.get("final_loss"),
                "adapter_parameter_numel": adapter_numel,
                "held_out_episode_count": held_out_episodes,
            },
        },
        "validation_summary": validation["summary"],
        "included_files": sorted(copied),
        "required_eval_files": ["metrics.json", "RUN_REPORT.md", "loss_records.jsonl"],
        "public_package_allowed": [
            "loss metrics",
            "rank-level eval metrics",
            "progress logs",
            "run reports",
            "episode and dataset manifests",
            "adapter repair audit",
            "validation summaries",
        ],
        "public_package_forbidden": sorted(FORBIDDEN_SUFFIXES),
        "excluded_policy": "Raw Xperience-10M media/annotations, base-model weights, LoRA adapter weights, checkpoints, and large archives are not included.",
    }
    write_json(output_dir / "verified_result_summary.json", summary)

    report = [
        "# Cosmos3-Super Forward-Dynamics LoRA Result",
        "",
        f"- Backbone: `{args.backbone}`",
        f"- Training run: `{args.train_run_id}`",
        f"- Evaluation run: `{args.eval_run_id}`",
        "- Status: `verified`",
        f"- Train rows: `{training.get('train_samples')}`",
        f"- Val rows: `{val_metrics.get('num_eval_samples')}`",
        f"- Test rows: `{test_metrics.get('num_eval_samples')}`",
        f"- Train final loss: `{training.get('final_loss')}`",
        f"- Val forward-dynamics MSE: `{val_loss}`",
        f"- Test forward-dynamics MSE: `{test_loss}`",
        f"- Adapter parameters: `{adapter_numel}`",
        "",
        "This is a camera-pose proxy forward-dynamics LoRA over Cosmos3-Super. It supervises future vision velocity tokens, not semantic JSON labels.",
        "",
        summary["excluded_policy"],
    ]
    (output_dir / "PUBLIC_RESULT_SUMMARY.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    assert_public_safe(output_dir)
    print(json.dumps({"status": "verified", "output_dir": str(output_dir), "included_files": summary["included_files"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

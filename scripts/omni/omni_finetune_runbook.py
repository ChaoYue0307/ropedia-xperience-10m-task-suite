#!/usr/bin/env python3
"""Write a staged Xperience-10M -> Qwen3-Omni pilot runbook and comparisons."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path


PRIMARY_METRICS = [
    "action_macro_f1",
    "subtask_accuracy",
    "transition_accuracy",
    "next_action_accuracy",
    "object_micro_f1",
    "json_validity_rate",
]


def parse_args() -> argparse.Namespace:
    workspace_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Create omni fine-tuning runbook and optional metric comparison.")
    parser.add_argument("--workspace", type=Path, default=workspace_default)
    parser.add_argument("--run-id", default="xperience10m_qwen3_omni_32ep")
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--next-scale-episodes", type=int, default=64)
    parser.add_argument("--manifest", type=Path, default=workspace_default / "results/omni_finetune/xperience10m_omni_dataset/dataset_manifest.json")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--metric-file", type=Path, action="append", help="metrics.json files to compare.")
    return parser.parse_args()


def command_output(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"


def preflight_snapshot() -> dict:
    return {
        "host": platform.node(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "nvidia_smi": command_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]),
        "cuda_visible_devices": command_output(["bash", "-lc", "printf %s \"${CUDA_VISIBLE_DEVICES:-unset}\""]),
        "ffmpeg": command_output(["ffmpeg", "-version"]).splitlines()[0],
        "disk_home": command_output(["df", "-h", "/home/cy"]),
    }


def stage_commands(run_id: str, manifest_path: Path) -> list[dict]:
    dataset_dir = f"results/omni_finetune/{run_id}_dataset"
    dataset_jsonl = f"{dataset_dir}/dataset.jsonl"
    return [
        {
            "phase": "phase_0_preflight",
            "goal": "Confirm H20 runtime, local Qwen weights, ModelScope access, ffmpeg, and HOMIE loader.",
            "commands": [
                "nvidia-smi",
                "ffmpeg -version",
                "python -c \"from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor; print('qwen imports ok')\"",
            ],
        },
        {
            "phase": "phase_1_one_episode_smoke",
            "goal": "Reproduce adapter smoke and validate JSONL/media generation.",
            "commands": [
                f"python scripts/omni/build_episode_manifest.py --data-root /home/cy/Ropedia/modelscope_data --max-episodes 1 --output {manifest_path}",
                f"python scripts/omni/export_qwen3_omni_action_dataset.py --manifest {manifest_path} --max-windows-per-episode 16 --run-id {run_id}_dataset",
                f"python scripts/omni/qwen3_omni_inference_smoke.py --dataset-jsonl {dataset_jsonl} --sample-limit 3 --run-id {run_id}_zero_shot",
            ],
        },
        {
            "phase": "phase_2_three_episode_overfit",
            "goal": "Train adapter-only and Qwen LoRA on 3 episodes; require decreasing loss and >=98% JSON validity.",
            "commands": [
                f"python scripts/omni/build_episode_manifest.py --data-root /home/cy/Ropedia/modelscope_data --max-episodes 3 --output {manifest_path}",
                f"python scripts/omni/export_qwen3_omni_action_dataset.py --manifest {manifest_path} --run-id {run_id}_3ep_dataset",
                f"python scripts/omni/train_qwen3_omni_lora.py --dataset-jsonl results/omni_finetune/{run_id}_3ep_dataset/dataset.jsonl --run-id {run_id}_3ep_lora --max-train-samples 256",
            ],
        },
        {
            "phase": "phase_3_32_episode_pilot",
            "goal": "Run adapter-only, frozen Qwen, Qwen LoRA video/audio/text, and Qwen LoRA plus sensor bridge.",
            "commands": [
                f"python scripts/omni/build_episode_manifest.py --data-root /home/cy/Ropedia/modelscope_data --max-episodes 32 --output {manifest_path}",
                f"python scripts/omni/export_qwen3_omni_action_dataset.py --manifest {manifest_path} --run-id {run_id}_dataset",
                f"python scripts/omni/train_qwen3_omni_lora.py --dataset-jsonl {dataset_jsonl} --run-id {run_id}_lora",
                f"python scripts/omni/eval_qwen3_omni_lora.py --dataset-jsonl {dataset_jsonl} --adapter-dir checkpoints/{run_id}_lora/adapter_lora --run-id {run_id}_eval",
            ],
        },
        {
            "phase": "phase_4_scale_decision",
            "goal": "Scale to 64 only after stability, disk headroom, and sensor bridge improvements are confirmed.",
            "commands": [
                f"python scripts/omni/omni_finetune_runbook.py --run-id {run_id} --metric-file results/omni_finetune/{run_id}_eval/metrics.json",
            ],
        },
    ]


def load_metrics(paths: list[Path] | None) -> list[dict]:
    rows = []
    for path in paths or []:
        payload = json.loads(path.read_text(encoding="utf-8"))
        row = {"path": str(path), "run": path.parent.name}
        for metric in PRIMARY_METRICS:
            row[metric] = payload.get(metric)
        rows.append(row)
    return rows


def main() -> int:
    args = parse_args()
    args.workspace = args.workspace.expanduser().resolve()
    if args.output_dir is None:
        args.output_dir = args.workspace / "results" / "omni_finetune" / args.run_id
    args.output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "run_id": args.run_id,
        "goal": "Fine-tune Qwen3-Omni-Instruct for Xperience-10M episode understanding JSON QA.",
        "default_scale": {
            "pilot_episodes": args.episodes,
            "next_scale_episodes": args.next_scale_episodes,
            "do_not_start_with": "10000 episodes",
        },
        "backbone": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "download_priority": ["ModelScope", "Hugging Face fallback"],
        "training_unit": "sampled window-centered clips",
        "split_unit": "held-out episodes",
        "primary_metrics": PRIMARY_METRICS,
        "preflight_snapshot": preflight_snapshot(),
        "stages": stage_commands(args.run_id, args.manifest),
        "comparisons": load_metrics(args.metric_file),
        "scale_acceptance": [
            "Full pipeline completes from downloaded subset to metrics.",
            "No train/test episode leakage.",
            "JSON validity >= 0.98.",
            "Sensor bridge beats video/audio/text-only LoRA on at least 3 primary metrics.",
            "Commands, model ID, dataset manifest, GPU info, and split file are recorded.",
        ],
    }
    (args.output_dir / "runbook.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        f"run_id: {args.run_id}",
        "objective: xperience10m_episode_understanding_json_qa",
        "backbone: Qwen/Qwen3-Omni-30B-A3B-Instruct",
        f"pilot_episodes: {args.episodes}",
        f"next_scale_episodes: {args.next_scale_episodes}",
        "download_priority: [ModelScope, Hugging Face fallback]",
        "full_parameter_finetune: false",
    ]
    (args.output_dir / "config.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Plan Xperience-10M episode counts for a fine-tuning run.

This is a storage and evaluation-design helper. It does not train a model and
does not invent results. Use it before downloading many episodes.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path


GB = 1024 ** 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate feasible Xperience-10M fine-tuning sample counts.")
    parser.add_argument("--storage-root", type=Path, default=Path("."), help="Disk root to inspect.")
    parser.add_argument("--free-gb", type=float, default=None, help="Override measured free space in GiB.")
    parser.add_argument("--target-free-after-download-gb", type=float, default=800.0)
    parser.add_argument("--model-cache-gb", type=float, default=250.0)
    parser.add_argument("--checkpoint-cache-gb", type=float, default=200.0)
    parser.add_argument("--log-cache-gb", type=float, default=50.0)
    parser.add_argument("--minimal-per-episode-gb", type=float, default=2.02)
    parser.add_argument("--all-training-per-episode-gb", type=float, default=2.40)
    parser.add_argument("--full-preview-per-episode-gb", type=float, default=5.10)
    parser.add_argument("--windows-per-episode", type=int, default=1161)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--output", type=Path, default=Path("outputs/omni_exploration/finetune_sample_budget.json"))
    return parser.parse_args()


def measured_free_gb(storage_root: Path, override: float | None) -> float:
    if override is not None:
        return float(override)
    if not storage_root.exists():
        raise FileNotFoundError(f"storage root does not exist: {storage_root}")
    return shutil.disk_usage(storage_root).free / GB


def max_episodes_for_budget(available_data_gb: float, per_episode_gb: float) -> int:
    if available_data_gb <= 0 or per_episode_gb <= 0:
        return 0
    return max(0, int(math.floor(available_data_gb / per_episode_gb)))


def split_windows(episodes: int, windows_per_episode: int, test_fraction: float) -> dict:
    if episodes <= 0:
        return {"train_episodes": 0, "test_episodes": 0, "train_windows": 0, "test_windows": 0}
    test_episodes = max(1, int(round(episodes * test_fraction))) if episodes > 1 else 1
    train_episodes = max(0, episodes - test_episodes)
    return {
        "train_episodes": train_episodes,
        "test_episodes": test_episodes,
        "train_windows": train_episodes * windows_per_episode,
        "test_windows": test_episodes * windows_per_episode,
    }


def phase_rows(max_all_training: int, windows_per_episode: int, test_fraction: float) -> list[dict]:
    phase_specs = [
        ("smoke", 1, "Verify loaders, alignment, and heads."),
        ("smoke_plus", 3, "Catch obvious multi-episode path issues."),
        ("pilot", 16, "First held-out-episode evaluation."),
        ("recommended_next", 32, "Default next run if download layout is clean."),
        ("useful_lora_small", 64, "Train sensor adapters plus selected LoRA layers."),
        ("useful_lora_medium", 128, "More useful LoRA run after pilot is stable."),
        ("storage_heavy", 256, "Only after checkpoint size and data layout are stable."),
    ]
    rows = []
    for name, episodes, purpose in phase_specs:
        split = split_windows(episodes, windows_per_episode, test_fraction)
        rows.append({
            "phase": name,
            "episodes": episodes,
            "feasible_under_all_training_budget": episodes <= max_all_training,
            "approx_windows": episodes * windows_per_episode,
            **split,
            "purpose": purpose,
        })
    return rows


def choose_recommendation(max_all_training: int) -> int:
    for candidate in (32, 16, 8, 3, 1):
        if max_all_training >= candidate:
            return candidate
    return 0


def main() -> int:
    args = parse_args()
    free_gb = measured_free_gb(args.storage_root.expanduser(), args.free_gb)
    reserved_gb = args.target_free_after_download_gb + args.model_cache_gb + args.checkpoint_cache_gb + args.log_cache_gb
    available_data_gb = max(0.0, free_gb - reserved_gb)

    modes = {
        "minimal_annotation_plus_one_video": args.minimal_per_episode_gb,
        "all_training_files_no_rrd": args.all_training_per_episode_gb,
        "full_preview_including_rrd": args.full_preview_per_episode_gb,
    }
    mode_summary = {
        name: {
            "per_episode_gb": per_episode_gb,
            "max_episodes": max_episodes_for_budget(available_data_gb, per_episode_gb),
        }
        for name, per_episode_gb in modes.items()
    }
    max_all_training = mode_summary["all_training_files_no_rrd"]["max_episodes"]
    recommended = choose_recommendation(max_all_training)

    payload = {
        "assumptions": {
            "storage_root": str(args.storage_root),
            "measured_or_overridden_free_gb": round(free_gb, 3),
            "target_free_after_download_gb": args.target_free_after_download_gb,
            "reserved_model_cache_gb": args.model_cache_gb,
            "reserved_checkpoint_cache_gb": args.checkpoint_cache_gb,
            "reserved_log_cache_gb": args.log_cache_gb,
            "available_for_episode_data_gb": round(available_data_gb, 3),
            "windows_per_episode": args.windows_per_episode,
            "test_fraction": args.test_fraction,
            "note": "Episode sizes are estimates until build_episode_manifest.py scans the actual downloaded folders.",
        },
        "modes": mode_summary,
        "recommended_next_episodes": recommended,
        "recommended_next_reason": (
            "Use 32 episodes first when feasible; otherwise use the largest smaller phase. "
            "Scale to 64 or 128 only after the pilot download and held-out-episode evaluation are stable."
        ),
        "phases": phase_rows(max_all_training, args.windows_per_episode, args.test_fraction),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["assumptions"], indent=2))
    print(json.dumps({"recommended_next_episodes": recommended, "modes": mode_summary}, indent=2))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

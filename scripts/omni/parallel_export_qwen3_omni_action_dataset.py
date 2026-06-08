#!/usr/bin/env python3
"""Parallel episode export for Qwen3-Omni train/validation datasets."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from qwen3_omni_dataset_utils import build_messages, label_counts, load_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    workspace_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Export Qwen3-Omni JSON-QA records with per-episode workers.")
    parser.add_argument("--workspace", type=Path, default=workspace_default)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-id", default="xperience10m_qwen3_parallel_export")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=workspace_default / "outputs/omni_exploration/feature_cache")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--window-frames", type=int, default=20)
    parser.add_argument("--stride-frames", type=int, default=20)
    parser.add_argument("--qwen-context-frames", type=int, default=120)
    parser.add_argument("--max-windows-per-episode", type=int, default=32)
    parser.add_argument("--max-video-frames", type=int, default=16)
    parser.add_argument("--audio-source", default="fisheye_cam0")
    parser.add_argument("--audio-sample-rate", type=int, default=16000)
    parser.add_argument("--audio-band-count", type=int, default=16)
    parser.add_argument("--render-media", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-rebuild-cache", action="store_true")
    return parser.parse_args()


def shard_episodes(episodes: list[dict], workers: int) -> list[list[dict]]:
    workers = max(1, min(workers, len(episodes)))
    shards = [[] for _ in range(workers)]
    for split in ("train", "val", "test", "unspecified"):
        split_eps = [ep for ep in episodes if ep.get("split", "unspecified") == split]
        for idx, episode in enumerate(split_eps):
            shards[idx % workers].append(episode)
    return [shard for shard in shards if shard]


def write_shard_manifest(base_payload: dict, episodes: list[dict], path: Path, shard_index: int) -> None:
    split_counts = Counter(ep.get("split", "unspecified") for ep in episodes)
    summary = dict(base_payload.get("summary", {}))
    summary.update({
        "parallel_shard_index": shard_index,
        "num_episodes": len(episodes),
        "split_counts": dict(split_counts),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"summary": summary, "episodes": episodes}, indent=2), encoding="utf-8")


def run_shard(args: argparse.Namespace, shard_manifest: Path, shard_output: Path, shard_index: int) -> dict:
    script = Path(__file__).with_name("export_qwen3_omni_action_dataset.py")
    cmd = [
        sys.executable,
        str(script),
        "--workspace",
        str(args.workspace),
        "--manifest",
        str(shard_manifest),
        "--run-id",
        f"{args.run_id}_shard_{shard_index:02d}",
        "--output-dir",
        str(shard_output),
        "--cache-dir",
        str(args.cache_dir),
        "--window-frames",
        str(args.window_frames),
        "--stride-frames",
        str(args.stride_frames),
        "--qwen-context-frames",
        str(args.qwen_context_frames),
        "--max-windows-per-episode",
        str(args.max_windows_per_episode),
        "--max-video-frames",
        str(args.max_video_frames),
        "--audio-source",
        args.audio_source,
        "--audio-sample-rate",
        str(args.audio_sample_rate),
        "--audio-band-count",
        str(args.audio_band_count),
        "--allow-empty",
    ]
    if not args.render_media:
        cmd.append("--no-render-media")
    if args.force_rebuild_cache:
        cmd.append("--force-rebuild-cache")

    log_path = shard_output / "export.log"
    shard_output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(" ".join(cmd) + "\n")
        log.flush()
        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
    return {
        "shard_index": shard_index,
        "manifest": str(shard_manifest),
        "output_dir": str(shard_output),
        "dataset_jsonl": str(shard_output / "dataset.jsonl"),
        "seconds": round(time.time() - started, 3),
    }


def merge_shards(args: argparse.Namespace, shard_results: list[dict], output_dir: Path) -> dict:
    records = []
    shard_manifests = []
    available_modalities = []
    feature_manifests = []
    skipped_episodes = []
    for shard in sorted(shard_results, key=lambda row: row["shard_index"]):
        shard_records = load_jsonl(Path(shard["dataset_jsonl"]))
        for record in shard_records:
            record["parallel_export_shard"] = shard["shard_index"]
        records.extend(shard_records)
        manifest_path = Path(shard["output_dir"]) / "dataset_manifest.json"
        if manifest_path.exists():
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            shard_manifests.append(payload)
            available_modalities.extend(payload.get("available_modalities", []))
            for skipped in payload.get("skipped_episodes", []):
                skipped_episodes.append({"shard_index": shard["shard_index"], **skipped})
            feature_manifests.append({
                "shard_index": shard["shard_index"],
                "feature_manifest": payload.get("feature_manifest", []),
            })

    action_options = sorted({record["answer_json"]["action"] for record in records if record["answer_json"]["action"] != "unknown"})
    subtask_options = sorted({record["answer_json"]["subtask"] for record in records if record["answer_json"]["subtask"] != "unknown"})
    for record in records:
        record["action_options"] = action_options
        record["subtask_options"] = subtask_options
        record["label_options"] = action_options
        record["messages"] = build_messages(record, action_options, include_answer=True)

    dataset_path = output_dir / "dataset.jsonl"
    write_jsonl(dataset_path, records)
    dataset_manifest = {
        "run_id": args.run_id,
        "dataset_path": str(dataset_path),
        "num_samples": len(records),
        "num_episodes": len({record["episode_id"] for record in records}),
        "split_counts": dict(Counter(record["split"] for record in records)),
        "label_counts": label_counts(records),
        "action_options": action_options,
        "subtask_options": subtask_options,
        "parallel_export": {
            "num_workers": args.num_workers,
            "shards": shard_results,
        },
        "clip_policy": {
            "window_frames": args.window_frames,
            "stride_frames": args.stride_frames,
            "qwen_context_frames": args.qwen_context_frames,
            "max_windows_per_episode": args.max_windows_per_episode,
            "max_video_frames": args.max_video_frames,
            "audio_span": "same_as_video_context",
            "mosaic": "2x3 multi-camera grid",
        },
        "feature_manifest": feature_manifests,
        "available_modalities": available_modalities,
        "skipped_episodes": skipped_episodes,
        "notes": [
            "Shard media and sensor-feature paths remain in shard output directories.",
            "Assistant answers are strict JSON for episode understanding, not robot-control policies.",
            "Merged label options are recomputed globally across all shards.",
            "Episodes with no labeled windows under the configured label rule are skipped and reported.",
        ],
    }
    (output_dir / "dataset_manifest.json").write_text(json.dumps(dataset_manifest, indent=2), encoding="utf-8")
    return dataset_manifest


def main() -> int:
    args = parse_args()
    args.workspace = args.workspace.expanduser().resolve()
    args.manifest = args.manifest.expanduser().resolve()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    if args.output_dir is None:
        args.output_dir = args.workspace / "results" / "omni_finetune" / args.run_id
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    episodes = payload.get("episodes", [])
    if not episodes:
        raise ValueError(f"No episodes found in manifest: {args.manifest}")

    shards = shard_episodes(episodes, args.num_workers)
    shard_root = args.output_dir / "shards"
    shard_jobs = []
    for shard_index, shard in enumerate(shards):
        shard_manifest = shard_root / f"manifest_shard_{shard_index:02d}.json"
        shard_output = shard_root / f"shard_{shard_index:02d}"
        write_shard_manifest(payload, shard, shard_manifest, shard_index)
        shard_jobs.append((shard_manifest, shard_output, shard_index))

    started = time.time()
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(shard_jobs)) as pool:
        futures = [
            pool.submit(run_shard, args, shard_manifest, shard_output, shard_index)
            for shard_manifest, shard_output, shard_index in shard_jobs
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps({"event": "shard_done", **result}, sort_keys=True), flush=True)

    dataset_manifest = merge_shards(args, results, args.output_dir)
    dataset_manifest["parallel_export"]["seconds"] = round(time.time() - started, 3)
    (args.output_dir / "dataset_manifest.json").write_text(json.dumps(dataset_manifest, indent=2), encoding="utf-8")
    print(json.dumps(dataset_manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

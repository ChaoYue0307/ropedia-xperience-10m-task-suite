#!/usr/bin/env python3
"""Export Xperience-10M windows as Qwen3-Omni JSON-QA fine-tuning records."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from qwen3_omni_dataset_utils import (
    build_messages,
    episode_dirs_from_sources,
    existing_videos,
    label_counts,
    primary_video_path,
    split_for_episode,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    workspace_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Export Qwen3-Omni JSON-QA SFT windows.")
    parser.add_argument("--workspace", type=Path, default=workspace_default)
    parser.add_argument("--episode-root", type=Path, action="append")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--split", choices=["all", "train", "val", "test"], default="all")
    parser.add_argument("--run-id", default="xperience10m_omni_dataset")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=workspace_default / "outputs/omni_exploration/feature_cache")
    parser.add_argument("--window-frames", type=int, default=20)
    parser.add_argument("--stride-frames", type=int, default=20)
    parser.add_argument("--qwen-context-frames", type=int, default=120)
    parser.add_argument("--max-video-frames", type=int, default=32)
    parser.add_argument("--min-label-fraction", type=float, default=0.6)
    parser.add_argument("--max-windows-per-episode", type=int, default=256)
    parser.add_argument("--video-image-size", type=int, default=32)
    parser.add_argument("--video-grid-size", type=int, default=8)
    parser.add_argument("--video-hist-bins", type=int, default=8)
    parser.add_argument("--depth-grid-size", type=int, default=8)
    parser.add_argument("--text-hash-dim", type=int, default=128)
    parser.add_argument("--audio-source", default="fisheye_cam0")
    parser.add_argument("--audio-sample-rate", type=int, default=16000)
    parser.add_argument("--audio-band-count", type=int, default=16)
    parser.add_argument("--mosaic-tile-width", type=int, default=320)
    parser.add_argument("--mosaic-tile-height", type=int, default=240)
    parser.add_argument("--mosaic-fps", type=float, default=8.0)
    parser.add_argument("--force-rebuild-cache", action="store_true")
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Write an empty dataset manifest instead of failing when every selected episode is skipped.",
    )
    parser.add_argument(
        "--with-handcrafted-video-features",
        action="store_true",
        help="Also decode MP4s into handcrafted sensor features. The default skips this because Qwen consumes rendered mosaic video directly.",
    )
    parser.add_argument("--render-media", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def add_repo_imports(args: argparse.Namespace) -> None:
    from qwen3_omni_adapter_smoke import add_repo_imports, add_toolkit_to_path

    add_repo_imports(args.workspace)
    add_toolkit_to_path(args.workspace)


def load_episode_dataset(args: argparse.Namespace, episode_dir: Path, target: str):
    from qwen3_omni_adapter_smoke import load_episode

    local_args = argparse.Namespace(**vars(args))
    local_args.target = target
    local_args.include_label_text = False
    local_args.max_windows_per_episode = 0
    local_args.skip_video_features = not args.with_handcrafted_video_features
    local_args.cache_dir = args.cache_dir / target
    local_args.output_dir = args.output_dir
    local_args.base_model_id = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    return load_episode(local_args, episode_dir)


def load_annotation(args: argparse.Namespace, episode_dir: Path) -> dict:
    from data_loader import load_from_annotation_hdf5

    return load_from_annotation_hdf5(episode_dir / "annotation.hdf5", 0, None, load_slam_point_cloud=True)


def frame_label(info: dict, target: str) -> str:
    key = "theme" if target == "subtask" else "action_label"
    label = str(info.get(key, "")).strip()
    return "" if not label or label.upper() == "N/A" else label


def majority_label(labels: list[str], min_fraction: float) -> str:
    labels = [label for label in labels if label]
    if not labels:
        return "unknown"
    label, count = Counter(labels).most_common(1)[0]
    return label if count / len(labels) >= min_fraction else "unknown"


def is_skippable_episode_error(exc: ValueError) -> bool:
    message = str(exc)
    skippable_markers = (
        "No labeled windows were created",
        "No caption_frame_info_map found in annotation",
    )
    return any(marker in message for marker in skippable_markers)


def collect_objects(frame_info: dict, start: int, end: int) -> list[str]:
    counts = Counter()
    for idx in range(start, end):
        objects = frame_info.get(idx, {}).get("objects", [])
        if isinstance(objects, str):
            objects = [objects]
        for obj in objects or []:
            value = str(obj).strip()
            if value:
                counts[value] += 1
    return [name for name, _count in counts.most_common()]


def contact_label(ann: dict, start: int, end: int) -> str:
    contacts = ann.get("contacts")
    if contacts is None or start >= len(contacts):
        return "unknown"
    window = np.asarray(contacts[start:min(end, len(contacts))])
    return "yes" if np.any(window > 0) else "no"


def transition_label(frame_info: dict, start: int, end: int) -> str:
    labels = [frame_label(frame_info.get(idx, {}), "action") for idx in range(start, end)]
    labels = [label for label in labels if label]
    return "yes" if len(set(labels)) > 1 else "no"


def context_span(start: int, end: int, n_frames: int, context_frames: int) -> tuple[int, int]:
    center = (start + end) // 2
    half = context_frames // 2
    ctx_start = max(0, center - half)
    ctx_end = min(n_frames - 1, ctx_start + context_frames - 1)
    ctx_start = max(0, ctx_end - context_frames + 1)
    return ctx_start, ctx_end


def video_fps(path: Path | None) -> float:
    if path is None or not path.exists():
        return 30.0
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0.0
    cap.release()
    return fps if fps and fps > 1e-3 else 30.0


def render_mosaic(video_paths: list[dict], output_path: Path, ctx_start: int, ctx_end: int, args: argparse.Namespace) -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    captures = []
    for item in video_paths:
        cap = cv2.VideoCapture(item["path"])
        captures.append(cap if cap.isOpened() else None)
    if not any(captures):
        for cap in captures:
            if cap is not None:
                cap.release()
        return False

    tile_w, tile_h = args.mosaic_tile_width, args.mosaic_tile_height
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.mosaic_fps,
        (tile_w * 3, tile_h * 2),
    )
    frame_indices = np.linspace(ctx_start, ctx_end, min(args.max_video_frames, ctx_end - ctx_start + 1), dtype=np.int64)
    black = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
    for frame_idx in frame_indices:
        tiles = []
        for cap in captures:
            if cap is None:
                tiles.append(black.copy())
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = cap.read()
            if not ok:
                tiles.append(black.copy())
            else:
                tiles.append(cv2.resize(frame, (tile_w, tile_h), interpolation=cv2.INTER_AREA))
        while len(tiles) < 6:
            tiles.append(black.copy())
        mosaic = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:6])])
        writer.write(mosaic)
    writer.release()
    for cap in captures:
        if cap is not None:
            cap.release()
    return output_path.exists()


def extract_audio(video_path: Path | None, output_path: Path, start_sec: float, duration_sec: float) -> bool:
    if video_path is None or not video_path.exists():
        return False
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        try:
            import imageio_ffmpeg

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-ss",
        f"{start_sec:.3f}",
        "-t",
        f"{duration_sec:.3f}",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
    return output_path.exists()


def stratified_cap(indices: list[int], labels: list[str], max_count: int) -> list[int]:
    if max_count <= 0 or len(indices) <= max_count:
        return indices
    by_label = defaultdict(list)
    for idx in indices:
        by_label[labels[idx]].append(idx)
    selected = []
    while len(selected) < max_count and any(by_label.values()):
        for label in sorted(by_label):
            if by_label[label] and len(selected) < max_count:
                selected.append(by_label[label].pop(0))
    return sorted(selected)


def build_answer(ann: dict, start: int, end: int, min_fraction: float) -> dict:
    frame_info = ann.get("caption_frame_info_map") or {}
    action = majority_label([frame_label(frame_info.get(idx, {}), "action") for idx in range(start, end)], min_fraction)
    subtask = majority_label([frame_label(frame_info.get(idx, {}), "subtask") for idx in range(start, end)], min_fraction)
    next_start = end
    next_end = min(end + (end - start), len(ann["img_names"]))
    next_action = "unknown"
    if next_start < next_end:
        next_action = majority_label([frame_label(frame_info.get(idx, {}), "action") for idx in range(next_start, next_end)], min_fraction)
    return {
        "action": action,
        "subtask": subtask,
        "objects": collect_objects(frame_info, start, end),
        "contact": contact_label(ann, start, end),
        "transition": transition_label(frame_info, start, end),
        "next_action": next_action,
        "evidence_window": {"start_frame": int(start), "end_frame": int(end - 1)},
    }


def export_episode(args: argparse.Namespace, episode_dir: Path, records: list[dict], summaries: dict) -> None:
    ann = load_annotation(args, episode_dir)
    action_ep = load_episode_dataset(args, episode_dir, "action")
    episode_key = f"{episode_dir.parent.name}__{episode_dir.name}"
    answers = [
        build_answer(ann, int(start), int(end) + 1, args.min_label_fraction)
        for start, end in zip(action_ep.starts, action_ep.ends)
    ]
    videos = existing_videos(episode_dir)
    primary = primary_video_path(videos)
    primary_path = Path(primary) if primary else None
    fps = video_fps(primary_path)
    feature_dir = args.output_dir / "sensor_features"
    media_dir = args.output_dir / "media" / episode_key
    feature_path = feature_dir / f"{episode_key}_sensor_features.npz"
    feature_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        feature_path,
        features=action_ep.X.astype(np.float32),
        action_labels=np.asarray([answer["action"] for answer in answers], dtype=object),
        subtask_labels=np.asarray([answer["subtask"] for answer in answers], dtype=object),
        starts=action_ep.starts,
        ends=action_ep.ends,
    )

    labels = [str(answer["action"]) for answer in answers]
    keep = stratified_cap(list(range(len(action_ep.labels))), labels, args.max_windows_per_episode)
    split = split_for_episode(episode_key, args.manifest, episode_dir)
    n_frames = len(ann["img_names"])

    for idx in keep:
        start = int(action_ep.starts[idx])
        end_inclusive = int(action_ep.ends[idx])
        end_exclusive = end_inclusive + 1
        ctx_start, ctx_end = context_span(start, end_inclusive, n_frames, args.qwen_context_frames)
        media = {
            "video_paths": videos,
            "context_start_frame": ctx_start,
            "context_end_frame": ctx_end,
            "max_video_frames": args.max_video_frames,
            "mosaic_video_path": None,
            "audio_path": None,
        }
        if args.render_media:
            stem = f"{episode_key}_w{idx:05d}_ctx{ctx_start}_{ctx_end}"
            mosaic_path = media_dir / f"{stem}_mosaic.mp4"
            audio_path = media_dir / f"{stem}_audio.wav"
            if render_mosaic(videos, mosaic_path, ctx_start, ctx_end, args):
                media["mosaic_video_path"] = str(mosaic_path)
            start_sec = ctx_start / fps
            duration_sec = max((ctx_end - ctx_start + 1) / fps, 0.01)
            if extract_audio(primary_path, audio_path, start_sec, duration_sec):
                media["audio_path"] = str(audio_path)

        answer = answers[idx]
        record = {
            "id": f"{episode_key}:qa:{idx}",
            "episode_id": episode_key,
            "source_episode_id": action_ep.episode_id,
            "episode_path": str(episode_dir),
            "split": split,
            "target": "episode_qa",
            "prompt_type": "json_episode_understanding",
            "center_window": {"start_frame": start, "end_frame": end_inclusive, "num_frames": args.window_frames},
            "media": media,
            "sensor_feature_path": str(feature_path),
            "sensor_feature_index": int(idx),
            "sensor_feature_dim": int(action_ep.X.shape[1]),
            "question": "Given the synchronized egocentric video/audio context and sensor window, identify the current embodied episode state.",
            "answer_json": answer,
            "label": answer["action"],
        }
        records.append(record)

    summaries["feature_manifest"] = action_ep.feature_manifest
    summaries["available_modalities"].append({"episode_id": episode_key, "modalities": action_ep.available_modalities})


def main() -> int:
    args = parse_args()
    args.workspace = args.workspace.expanduser().resolve()
    if args.output_dir is None:
        args.output_dir = args.workspace / "results" / "omni_finetune" / args.run_id
    args.output_dir.mkdir(parents=True, exist_ok=True)
    add_repo_imports(args)

    episode_dirs = episode_dirs_from_sources(args.episode_root, args.manifest, args.split)
    if not episode_dirs:
        default_episode = args.workspace / "data/sample/xperience-10m-sample"
        if default_episode.exists():
            episode_dirs = [default_episode.resolve()]
        else:
            raise ValueError("No episode directories found. Pass --episode-root or --manifest.")

    records: list[dict] = []
    summaries = {"feature_manifest": [], "available_modalities": [], "skipped_episodes": []}
    for episode_dir in episode_dirs:
        try:
            export_episode(args, episode_dir, records, summaries)
        except ValueError as exc:
            if not is_skippable_episode_error(exc):
                raise
            summaries["skipped_episodes"].append({
                "episode_path": str(episode_dir),
                "reason": str(exc),
            })

    if not records and not args.allow_empty:
        raise ValueError("No dataset records were exported from the selected episodes.")

    action_options = sorted({record["answer_json"]["action"] for record in records if record["answer_json"]["action"] != "unknown"})
    subtask_options = sorted({record["answer_json"]["subtask"] for record in records if record["answer_json"]["subtask"] != "unknown"})
    for record in records:
        record["action_options"] = action_options
        record["subtask_options"] = subtask_options
        record["label_options"] = action_options
        record["messages"] = build_messages(record, action_options, include_answer=True)

    dataset_path = args.output_dir / "dataset.jsonl"
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
        "clip_policy": {
            "label_window_frames": args.window_frames,
            "qwen_context_frames": args.qwen_context_frames,
            "max_video_frames": args.max_video_frames,
            "audio_span": "same_as_video_context",
            "mosaic": "2x3 multi-camera grid",
            "allow_empty": args.allow_empty,
        },
        "feature_manifest": summaries["feature_manifest"],
        "available_modalities": summaries["available_modalities"],
        "skipped_episodes": summaries["skipped_episodes"],
        "notes": [
            "Assistant answers are strict JSON for episode understanding, not robot-control policies.",
            "Sensor features are stored as NPZ pointers; raw annotation.hdf5 is not copied into the dataset records.",
            "Episodes with no labeled windows under the configured label rule are skipped and reported.",
        ],
    }
    (args.output_dir / "dataset_manifest.json").write_text(json.dumps(dataset_manifest, indent=2), encoding="utf-8")
    (args.output_dir / "config.yaml").write_text(
        "\n".join([
            f"run_id: {args.run_id}",
            "objective: episode_understanding_json_qa",
            "backbone: Qwen/Qwen3-Omni-30B-A3B-Instruct",
            f"max_windows_per_episode: {args.max_windows_per_episode}",
            f"qwen_context_frames: {args.qwen_context_frames}",
            f"max_video_frames: {args.max_video_frames}",
            f"render_media: {str(args.render_media).lower()}",
            f"with_handcrafted_video_features: {str(args.with_handcrafted_video_features).lower()}",
        ]) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(dataset_manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Extract raw Xperience-10M caption interaction text from annotation HDF5 files.

The full annotation files are large, so this utility is designed for
one-file-at-a-time extraction. It writes compact JSONL rows containing the raw
caption config, segment labels, action labels/descriptions, and per-frame
``interaction`` strings needed to build task-15 targets.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-file", type=Path, required=True, help="Text file of repo-relative annotation.hdf5 paths.")
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--local-root", type=Path, help="Optional local root containing repo-relative annotation paths.")
    parser.add_argument("--download", action="store_true", help="Download missing files from Hugging Face one at a time.")
    parser.add_argument("--repo-id", default="ropedia-ai/xperience-10m")
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument("--cache-dir", type=Path, help="Dedicated Hugging Face cache directory for downloaded HDF5 files.")
    parser.add_argument("--cleanup-cache-each-file", action="store_true")
    parser.add_argument("--allow-clean-cache-dir", action="store_true")
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    return parser.parse_args()


def load_h5py():
    try:
        import h5py  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment diagnostic.
        raise SystemExit("h5py is required to read annotation.hdf5 captions") from exc
    return h5py


def load_hf_download():
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment diagnostic.
        raise SystemExit("huggingface_hub is required when --download is set") from exc
    return hf_hub_download


def read_paths(paths_file: Path, max_files: int) -> list[str]:
    paths = [line.strip() for line in paths_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if max_files > 0:
        return paths[:max_files]
    return paths


def completed_paths(output_jsonl: Path) -> set[str]:
    if not output_jsonl.exists():
        return set()
    done: set[str] = set()
    with output_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            path = row.get("annotation_path")
            if isinstance(path, str):
                done.add(path)
    return done


def safe_cleanup_cache(cache_dir: Path, allow: bool) -> None:
    if not cache_dir.exists():
        return
    marker_ok = "xperience10m_annotation_caption_cache" in cache_dir.name
    if not allow and not marker_ok:
        raise SystemExit(
            f"Refusing to remove cache dir {cache_dir}; use a dedicated path named "
            "xperience10m_annotation_caption_cache or pass --allow-clean-cache-dir."
        )
    shutil.rmtree(cache_dir)


def resolve_annotation_path(args: argparse.Namespace, rel_path: str) -> tuple[Path, bool]:
    if args.local_root:
        candidate = args.local_root / rel_path
        if candidate.exists():
            return candidate, False
    if not args.download:
        raise FileNotFoundError(f"Missing local annotation and --download is not set: {rel_path}")
    if args.cache_dir is None:
        raise SystemExit("--cache-dir is required with --download so downloads stay isolated.")
    hf_hub_download = load_hf_download()
    path = hf_hub_download(
        repo_id=args.repo_id,
        filename=rel_path,
        repo_type=args.repo_type,
        cache_dir=str(args.cache_dir),
    )
    return Path(path), True


def as_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def sorted_interactions(interaction: Any) -> list[dict[str, Any]]:
    if not isinstance(interaction, dict):
        return []
    rows: list[dict[str, Any]] = []
    for key, value in interaction.items():
        text = as_text(value)
        if text is None:
            continue
        try:
            frame_sort = int(key)
        except (TypeError, ValueError):
            frame_sort = 0
        rows.append({"frame": str(key), "frame_sort": frame_sort, "text": text})
    return sorted(rows, key=lambda row: (row["frame_sort"], row["frame"]))


def compact_actions(segment: dict[str, Any]) -> list[dict[str, Any]]:
    actions = segment.get("Current Action")
    if not isinstance(actions, list):
        return []
    compact: list[dict[str, Any]] = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        compact.append(
            {
                "label": as_text(action.get("label")),
                "description": as_text(action.get("description")),
                "start_frame": action.get("start_frame"),
                "end_frame": action.get("end_frame"),
            }
        )
    return compact


def extract_caption_rows(annotation_path: Path, rel_path: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    h5py = load_h5py()
    with h5py.File(annotation_path, "r") as handle:
        if "caption" not in handle:
            raise KeyError("caption dataset is missing")
        raw = handle["caption"][()]
    text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
    caption = json.loads(text)
    if not isinstance(caption, dict):
        raise TypeError("caption JSON must be an object")

    config = caption.get("config") if isinstance(caption.get("config"), dict) else {}
    segments = caption.get("segments") if isinstance(caption.get("segments"), list) else []
    episode_id = str(Path(rel_path).parent)
    rows: list[dict[str, Any]] = []
    segment_count = 0
    action_count = 0
    interaction_count = 0

    for segment in segments:
        if not isinstance(segment, dict):
            continue
        segment_count += 1
        actions = compact_actions(segment)
        action_count += len(actions)
        interactions = sorted_interactions(segment.get("interaction"))
        interaction_count += len(interactions)
        base = {
            "annotation_path": rel_path,
            "episode_id": episode_id,
            "main_task": as_text(config.get("Main Task")),
            "segment_id": segment.get("segment_id"),
            "segment_start_frame": segment.get("start_frame"),
            "segment_end_frame": segment.get("end_frame"),
            "sub_task": as_text(segment.get("Sub Task")),
            "actions": actions,
            "action_labels": [a["label"] for a in actions if a.get("label")],
            "action_descriptions": [a["description"] for a in actions if a.get("description")],
            "objects": segment.get("objects") if isinstance(segment.get("objects"), list) else [],
        }
        for interaction in interactions:
            rows.append(
                {
                    **base,
                    "interaction_frame": interaction["frame"],
                    "interaction_frame_sort": interaction["frame_sort"],
                    "interaction_text": interaction["text"],
                }
            )

    summary = {
        "annotation_path": rel_path,
        "episode_id": episode_id,
        "main_task": as_text(config.get("Main Task")),
        "caption_total_frames": config.get("total_frames"),
        "caption_total_tokens": config.get("total_tokens"),
        "segment_count": segment_count,
        "action_count": action_count,
        "interaction_count": interaction_count,
        "output_row_count": len(rows),
    }
    return summary, rows


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    paths = read_paths(args.paths_file, args.max_files)
    done = completed_paths(args.output_jsonl) if args.resume else set()
    summaries: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    started = time.time()

    for index, rel_path in enumerate(paths, start=1):
        if rel_path in done:
            summaries.append({"annotation_path": rel_path, "status": "skipped_existing"})
            continue
        downloaded = False
        try:
            local_path, downloaded = resolve_annotation_path(args, rel_path)
            summary, rows = extract_caption_rows(local_path, rel_path)
            append_rows(args.output_jsonl, rows)
            summaries.append({**summary, "status": "extracted", "index": index, "downloaded": downloaded})
            print(
                json.dumps(
                    {
                        "event": "extracted",
                        "index": index,
                        "total": len(paths),
                        "path": rel_path,
                        "rows": len(rows),
                        "downloaded": downloaded,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - keep batch extraction moving.
            error = {"annotation_path": rel_path, "status": "error", "error": f"{type(exc).__name__}: {exc}", "index": index}
            errors.append(error)
            summaries.append(error)
            print(json.dumps({"event": "error", **error}, sort_keys=True), flush=True)
        finally:
            if downloaded and args.cleanup_cache_each_file and args.cache_dir is not None:
                safe_cleanup_cache(args.cache_dir, args.allow_clean_cache_dir)
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

        write_json(
            args.manifest_json,
            {
                "status": "running",
                "requested_file_count": len(paths),
                "processed_file_count": len(summaries),
                "error_count": len(errors),
                "output_jsonl": str(args.output_jsonl),
                "elapsed_seconds": round(time.time() - started, 3),
                "files": summaries,
            },
        )

    write_json(
        args.manifest_json,
        {
            "status": "pass" if not errors else "partial",
            "requested_file_count": len(paths),
            "processed_file_count": len(summaries),
            "extracted_file_count": sum(1 for row in summaries if row.get("status") == "extracted"),
            "skipped_existing_count": sum(1 for row in summaries if row.get("status") == "skipped_existing"),
            "error_count": len(errors),
            "output_jsonl": str(args.output_jsonl),
            "output_row_count": sum(int(row.get("output_row_count") or 0) for row in summaries),
            "elapsed_seconds": round(time.time() - started, 3),
            "files": summaries,
            "errors": errors,
        },
    )


if __name__ == "__main__":
    main()

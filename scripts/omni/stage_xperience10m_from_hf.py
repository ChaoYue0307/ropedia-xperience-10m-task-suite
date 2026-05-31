#!/usr/bin/env python3
"""Stage a bounded Xperience-10M episode subset from Hugging Face.

This downloads leaf episode folders such as:

    <session_uuid>/ep1/{annotation.hdf5,fisheye_cam0.mp4,...}

It intentionally excludes visualization.rrd and writes a manifest that can be
used before transferring data to the H20 training server.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


REQUIRED_FILES = [
    "annotation.hdf5",
    "fisheye_cam0.mp4",
    "fisheye_cam1.mp4",
    "fisheye_cam2.mp4",
    "fisheye_cam3.mp4",
    "stereo_left.mp4",
    "stereo_right.mp4",
]


@dataclass
class Episode:
    episode_id: str
    prefix: str
    files: dict[str, int]

    @property
    def session_id(self) -> str:
        return self.prefix.split("/", 1)[0]

    @property
    def leaf_episode(self) -> str:
        parts = self.prefix.split("/", 1)
        return parts[1] if len(parts) > 1 else "."

    @property
    def missing(self) -> list[str]:
        return [name for name in REQUIRED_FILES if name not in self.files]

    @property
    def is_complete(self) -> bool:
        return not self.missing

    @property
    def is_degraded_valid(self) -> bool:
        return "annotation.hdf5" in self.files and "fisheye_cam0.mp4" in self.files

    @property
    def bytes(self) -> int:
        return sum(self.files.values())

    def as_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "prefix": self.prefix,
            "session_id": self.session_id,
            "leaf_episode": self.leaf_episode,
            "files": self.files,
            "missing": self.missing,
            "is_complete": self.is_complete,
            "is_degraded_valid": self.is_degraded_valid,
            "bytes": self.bytes,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="ropedia-ai/xperience-10m")
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--target-episodes", type=int, default=32)
    parser.add_argument("--max-top-level", type=int, default=64)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--reserve-gb", type=float, default=100.0)
    parser.add_argument("--prefer-complete", action="store_true", default=True)
    parser.add_argument("--allow-degraded", action="store_true")
    parser.add_argument("--min-episode-gb", type=float, default=0.25)
    parser.add_argument(
        "--selection-strategy",
        choices=["stratified", "first"],
        default="stratified",
        help="stratified spreads episodes across top-level session UUIDs.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifest-name", default="stage_manifest.json")
    return parser.parse_args()


def file_size(item) -> int:
    value = getattr(item, "size", None)
    return int(value) if value is not None else 0


def natural_episode_key(episode: Episode) -> tuple[str, int, str]:
    match = re.fullmatch(r"ep(\d+)", episode.leaf_episode)
    numeric = int(match.group(1)) if match else 10**9
    return episode.session_id, numeric, episode.leaf_episode


def collect_candidates(api: HfApi, repo_id: str, max_top_level: int) -> list[Episode]:
    candidates: list[Episode] = []
    top_count = 0
    for top in api.list_repo_tree(repo_id, repo_type="dataset", recursive=False):
        top_path = getattr(top, "path", "")
        if not top_path:
            continue
        top_count += 1
        grouped: dict[str, dict[str, int]] = {}
        for item in api.list_repo_tree(repo_id, repo_type="dataset", path_in_repo=top_path, recursive=True):
            path = getattr(item, "path", "")
            name = Path(path).name
            if name not in REQUIRED_FILES:
                continue
            prefix = Path(path).parent.as_posix()
            grouped.setdefault(prefix, {})[name] = file_size(item)

        for prefix, files in sorted(grouped.items()):
            episode_id = prefix.replace("/", "__")
            episode = Episode(episode_id=episode_id, prefix=prefix, files=files)
            if episode.is_degraded_valid:
                candidates.append(episode)

        if top_count >= max_top_level:
            break
    return sorted(candidates, key=natural_episode_key)


def round_robin_by_session(episodes: list[Episode], target: int) -> list[Episode]:
    grouped: dict[str, list[Episode]] = {}
    for episode in sorted(episodes, key=natural_episode_key):
        grouped.setdefault(episode.session_id, []).append(episode)

    selected: list[Episode] = []
    session_ids = sorted(grouped)
    depth = 0
    while len(selected) < target:
        added = False
        for session_id in session_ids:
            bucket = grouped[session_id]
            if depth < len(bucket):
                selected.append(bucket[depth])
                added = True
                if len(selected) >= target:
                    break
        if not added:
            break
        depth += 1
    return selected


def select_episodes(
    candidates: list[Episode],
    target: int,
    prefer_complete: bool,
    allow_degraded: bool,
    min_episode_bytes: int,
    selection_strategy: str,
) -> list[Episode]:
    eligible = [ep for ep in candidates if ep.bytes >= min_episode_bytes]
    if prefer_complete and not allow_degraded:
        complete = [ep for ep in eligible if ep.is_complete]
        if len(complete) >= target:
            pool = complete
        else:
            pool = [ep for ep in eligible if ep.is_degraded_valid]
    else:
        pool = [ep for ep in eligible if ep.is_degraded_valid]

    if selection_strategy == "first":
        return pool[:target]
    return round_robin_by_session(pool, target)


def local_file(local_dir: Path, filename: str) -> Path:
    return local_dir / filename


def download_one(repo_id: str, local_dir: Path, filename: str, token: str | None) -> dict:
    path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=filename,
        local_dir=str(local_dir),
        token=token,
    )
    stat = Path(path).stat()
    return {"path": filename, "local_path": path, "bytes": stat.st_size}


def validate_selected(local_dir: Path, selected: list[Episode]) -> list[dict]:
    records = []
    for episode in selected:
        files = {}
        for name in REQUIRED_FILES:
            path = local_file(local_dir, f"{episode.prefix}/{name}")
            files[name] = {
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else 0,
            }
        records.append(
            {
                **episode.as_dict(),
                "local_files": files,
                "local_complete": all(item["exists"] for item in files.values()),
                "local_degraded_valid": files["annotation.hdf5"]["exists"]
                and files["fisheye_cam0.mp4"]["exists"],
            }
        )
    return records


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    local_dir = args.local_dir.expanduser().resolve()
    local_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi(token=token)
    candidates = collect_candidates(api, args.repo_id, args.max_top_level)
    min_episode_bytes = int(args.min_episode_gb * 1024**3)
    selected = select_episodes(
        candidates,
        args.target_episodes,
        args.prefer_complete,
        args.allow_degraded,
        min_episode_bytes,
        args.selection_strategy,
    )
    required_bytes = sum(ep.bytes for ep in selected)
    free_bytes = shutil.disk_usage(local_dir).free
    reserve_bytes = int(args.reserve_gb * 1024**3)

    payload = {
        "repo_id": args.repo_id,
        "local_dir": str(local_dir),
        "target_episodes": args.target_episodes,
        "max_top_level": args.max_top_level,
        "selection_strategy": args.selection_strategy,
        "min_episode_bytes": min_episode_bytes,
        "allow_degraded": args.allow_degraded,
        "num_candidates": len(candidates),
        "num_selected": len(selected),
        "num_selected_sessions": len({ep.session_id for ep in selected}),
        "required_bytes": required_bytes,
        "free_bytes_before": free_bytes,
        "reserve_bytes": reserve_bytes,
        "dry_run": args.dry_run,
        "selected": [ep.as_dict() for ep in selected],
    }
    write_manifest(local_dir / args.manifest_name, payload)

    if len(selected) < args.target_episodes:
        raise SystemExit(f"only found {len(selected)} valid episodes, target is {args.target_episodes}")
    if free_bytes - required_bytes < reserve_bytes:
        raise SystemExit(
            f"not enough free space: need {required_bytes} bytes plus reserve {reserve_bytes}, "
            f"free {free_bytes}"
        )
    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return 0

    filenames = [f"{ep.prefix}/{name}" for ep in selected for name in REQUIRED_FILES if name in ep.files]
    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(download_one, args.repo_id, local_dir, filename, token) for filename in filenames]
        for idx, future in enumerate(as_completed(futures), start=1):
            item = future.result()
            results.append(item)
            print(f"[{idx}/{len(futures)}] {item['path']} {item['bytes']}")

    final_payload = {
        **payload,
        "downloaded_files": sorted(results, key=lambda item: item["path"]),
        "validated": validate_selected(local_dir, selected),
        "free_bytes_after": shutil.disk_usage(local_dir).free,
    }
    write_manifest(local_dir / args.manifest_name, final_payload)
    print(f"Wrote {local_dir / args.manifest_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

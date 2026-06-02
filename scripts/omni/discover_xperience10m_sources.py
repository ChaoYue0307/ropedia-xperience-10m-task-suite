#!/usr/bin/env python3
"""Discover available Xperience-10M episodes and generate a readiness gate report."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


VIDEO_FILES = [
    "annotation.hdf5",
    "fisheye_cam0.mp4",
    "fisheye_cam1.mp4",
    "fisheye_cam2.mp4",
    "fisheye_cam3.mp4",
    "stereo_left.mp4",
    "stereo_right.mp4",
]


@dataclass
class EpisodeRecord:
    source: str
    episode_id: str
    episode_path: str
    has_annotation: bool
    has_fisheye_cam0: bool
    has_all_videos: bool
    has_any_video: bool
    missing_views: list[str]

    @property
    def is_degraded_valid(self) -> bool:
        return self.has_annotation and self.has_fisheye_cam0

    @property
    def is_complete(self) -> bool:
        return self.is_degraded_valid and self.has_all_videos

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "episode_id": self.episode_id,
            "episode_path": self.episode_path,
            "has_annotation": self.has_annotation,
            "has_fisheye_cam0": self.has_fisheye_cam0,
            "has_all_videos": self.has_all_videos,
            "has_any_video": self.has_any_video,
            "missing_views": self.missing_views,
            "is_degraded_valid": self.is_degraded_valid,
            "is_complete": self.is_complete,
        }


def parse_args() -> argparse.Namespace:
    workspace_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Discover Xperience-10M episode availability.")
    parser.add_argument("--workspace", type=Path, default=workspace_default)
    parser.add_argument("--data-root", type=Path, default=Path("modelscope_data"))
    parser.add_argument("--output", type=Path, default=Path("results/omni_finetune/source_discovery.json"))
    parser.add_argument("--report-output", type=Path, default=Path("results/omni_finetune/DATA_BLOCKER_REPORT.md"))
    parser.add_argument("--target-episodes", type=int, default=32)
    parser.add_argument(
        "--modelscope-repo-id",
        action="append",
        default=["ropedia-ai/xperience-10m", "ropedia-ai/xperience-10m-sample"],
        help="ModelScope dataset repo ids to probe.",
    )
    parser.add_argument(
        "--hf-repo-id",
        action="append",
        default=["ropedia-ai/xperience-10m", "ropedia-ai/xperience-10m-sample"],
        help="Hugging Face dataset repo ids to probe.",
    )
    parser.add_argument("--skip-modelscope", action="store_true")
    parser.add_argument("--skip-huggingface", action="store_true")
    return parser.parse_args()


def _coerce_files(payload) -> list[str]:
    if payload is None:
        return []
    if isinstance(payload, dict):
        for key in ("data", "files", "FilePaths", "File"):
            if key in payload and isinstance(payload[key], list):
                payload = payload[key]
                break
    if not isinstance(payload, list):
        payload = [payload]

    output = []
    for item in payload:
        if isinstance(item, str):
            output.append(item)
            continue
        if isinstance(item, dict):
            for key in ("path", "rfilename", "name", "Path", "uri"):
                value = item.get(key)
                if isinstance(value, str) and value:
                    output.append(value)
                    break
    return [i for i in output if i]


def _call_provider_api(callers: list[Callable[[], object]]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    for idx, fn in enumerate(callers):
        try:
            payload = fn()
            files = _coerce_files(payload)
        except Exception as exc:
            errors.append(f"call {idx} failed: {exc}")
            continue
        if files:
            return files, []
    return [], errors


def scan_local_episodes(data_root: Path) -> list[EpisodeRecord]:
    if not data_root.exists():
        return []

    out: dict[str, EpisodeRecord] = {}
    for annotation in sorted(data_root.rglob("annotation.hdf5")):
        episode_dir = annotation.parent
        present = {name: (episode_dir / name).exists() for name in VIDEO_FILES}
        missing = [name for name in VIDEO_FILES[1:] if not present[name]]
        out[str(episode_dir)] = EpisodeRecord(
            source="local",
            episode_id=episode_dir.name,
            episode_path=str(episode_dir),
            has_annotation=present["annotation.hdf5"],
            has_fisheye_cam0=present["fisheye_cam0.mp4"],
            has_all_videos=all(present[name] for name in VIDEO_FILES[1:]),
            has_any_video=any(present[name] for name in VIDEO_FILES[1:]),
            missing_views=missing,
        )
    return sorted(out.values(), key=lambda ep: ep.episode_id)


def collect_remote_records(source: str, repo_id: str, files: list[str]) -> list[EpisodeRecord]:
    grouped: dict[str, dict[str, bool]] = {}
    for raw_path in files:
        norm = str(raw_path).replace("\\", "/").strip("/")
        if not norm:
            continue
        name = Path(norm).name
        if name not in VIDEO_FILES:
            continue

        parent = Path(norm).parent.as_posix()
        if not parent:
            episode_key = Path(repo_id).name
            episode_path = f"{source}:{repo_id}"
            bucket_key = f"{source}:{repo_id}:."
        else:
            episode_key = Path(parent).name
            episode_path = f"{source}:{repo_id}/{parent}"
            bucket_key = f"{source}:{repo_id}:{parent}"

        bucket = grouped.setdefault(
            bucket_key,
            {
                "episode_id": episode_key,
                "episode_path": episode_path,
                "annotation.hdf5": False,
                "fisheye_cam0.mp4": False,
                "fisheye_cam1.mp4": False,
                "fisheye_cam2.mp4": False,
                "fisheye_cam3.mp4": False,
                "stereo_left.mp4": False,
                "stereo_right.mp4": False,
            },
        )
        bucket[name] = True

    episodes = []
    for bucket in grouped.values():
        episodes.append(
            EpisodeRecord(
                source=source,
                episode_id=bucket["episode_id"],
                episode_path=bucket["episode_path"],
                has_annotation=bucket["annotation.hdf5"],
                has_fisheye_cam0=bucket["fisheye_cam0.mp4"],
                has_all_videos=all(bucket[n] for n in VIDEO_FILES[1:]),
                has_any_video=any(bucket[n] for n in VIDEO_FILES[1:]),
                missing_views=[name for name in VIDEO_FILES[1:] if not bucket[name]],
            )
        )
    return sorted(episodes, key=lambda ep: ep.episode_id)


def summarize_episodes(episodes: list[EpisodeRecord], errors: list[str], name: str) -> dict:
    return {
        "source": name,
        "num_episodes": len(episodes),
        "num_degraded_valid_episodes": sum(ep.is_degraded_valid for ep in episodes),
        "num_complete_episodes": sum(ep.is_complete for ep in episodes),
        "errors": errors,
        "episodes": [ep.as_dict() for ep in episodes],
    }


def build_modelscope_records(repo_id: str) -> tuple[list[EpisodeRecord], list[str]]:
    try:
        from modelscope.hub.api import HubApi
    except Exception as exc:
        return [], [f"modelscope import failed: {exc}"]

    try:
        api = HubApi()
    except Exception as exc:
        return [], [f"modelscope HubApi init failed: {exc}"]

    callers = [
        lambda: api.get_dataset_files(repo_id),
        lambda: api.get_dataset_files(repo_id=repo_id),
        lambda: api.get_dataset_files(repo_id=repo_id, revision="master"),
        lambda: api.list_repo_files(repo_id=repo_id, repo_type="dataset"),
        lambda: api.get_repo_files(repo_id, repo_type="dataset"),
    ]
    files, errs = _call_provider_api(callers)
    if not files:
        return [], errs or ["modelscope returned no files"]
    return collect_remote_records("modelscope", repo_id, files), []


def build_huggingface_records(repo_id: str) -> tuple[list[EpisodeRecord], list[str]]:
    try:
        from huggingface_hub import HfApi
    except Exception as exc:
        return [], [f"huggingface_hub import failed: {exc}"]

    api = HfApi()
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    except Exception as exc:
        return [], [f"huggingface list_repo_files failed: {exc}"]

    records = _coerce_files(files)
    if not records:
        return [], ["huggingface returned no files"]
    return collect_remote_records("huggingface", repo_id, records), []


def pick_source(local: dict, modelscope: dict, huggingface: dict, target: int) -> tuple[str, list[str]]:
    if local["num_degraded_valid_episodes"] >= target:
        return "local", []
    if modelscope["num_degraded_valid_episodes"] >= target:
        return "modelscope", []
    if huggingface["num_degraded_valid_episodes"] >= target:
        return "huggingface", []

    blockers = [
        f"Not enough degraded-valid episodes for a 32-episode pilot. Need {target}, local has {local['num_degraded_valid_episodes']}.",
        "Current local path remains one-episode proof-of-stack only.",
    ]
    if local["num_episodes"] == 0:
        blockers.append(f"No local annotation.hdf5 found under {local.get('data_root', 'configured data root')}")
    if not modelscope["episodes"]:
        blockers.append("ModelScope probe unavailable or reported no matching episode files.")
    if not huggingface["episodes"]:
        blockers.append("Hugging Face probe unavailable or reported no matching episode files.")
    return "none", blockers


def write_blocker_report(payload: dict, path: Path) -> None:
    lines = [
        "# Xperience-10M Fine-Tune Readiness",
        "",
        f"Target episodes: {payload['target_episodes']}",
        f"Ready for 32-episode pilot: {payload['ready_for_32_episode_pilot']}",
        f"Selected source: {payload['selected_source']}",
        "",
        "## Source counts",
        f"- local (degraded-valid): {payload['local']['num_degraded_valid_episodes']} / {payload['local']['num_episodes']}",
        f"- modelscope (degraded-valid): {payload['modelscope']['num_degraded_valid_episodes']} / {payload['modelscope']['num_episodes']}",
        f"- huggingface (degraded-valid): {payload['huggingface']['num_degraded_valid_episodes']} / {payload['huggingface']['num_episodes']}",
        "",
        "## Blockers",
    ]
    if payload["blockers"]:
        lines.extend([f"- {item}" for item in payload["blockers"]])
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Interpretation",
            "- Degraded-valid means: annotation.hdf5 and fisheye_cam0.mp4 both exist.",
            "- Complete means all six MP4 views are present with annotation.",
            "- A 32-episode pilot moves to full execution only after this script selects a source with 32+ degraded-valid episodes.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    workspace = args.workspace.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()

    local_episodes = scan_local_episodes(data_root)
    local_summary = summarize_episodes(local_episodes, [], "local")
    local_summary["data_root"] = str(data_root)

    modelscope_episodes = []
    modelscope_errors: list[str] = []
    if not args.skip_modelscope:
        for repo in args.modelscope_repo_id:
            ep, errs = build_modelscope_records(repo)
            modelscope_episodes.extend(ep)
            modelscope_errors.extend([f"{repo}: {x}" for x in errs])
    modelscope_summary = summarize_episodes(modelscope_episodes, modelscope_errors, "modelscope")

    hf_episodes = []
    hf_errors: list[str] = []
    if not args.skip_huggingface:
        for repo in args.hf_repo_id:
            ep, errs = build_huggingface_records(repo)
            hf_episodes.extend(ep)
            hf_errors.extend([f"{repo}: {x}" for x in errs])
    huggingface_summary = summarize_episodes(hf_episodes, hf_errors, "huggingface")

    selected, blockers = pick_source(local_summary, modelscope_summary, huggingface_summary, args.target_episodes)
    ready = selected != "none"

    payload = {
        "target_episodes": args.target_episodes,
        "workspace": str(workspace),
        "data_root": str(data_root),
        "ready_for_32_episode_pilot": ready,
        "selected_source": selected,
        "local": local_summary,
        "modelscope": modelscope_summary,
        "huggingface": huggingface_summary,
        "blockers": blockers,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    write_blocker_report(payload, args.report_output)

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

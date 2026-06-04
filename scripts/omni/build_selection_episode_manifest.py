#!/usr/bin/env python3
"""Build a manifest from a fixed selected-episode list.

This is used for progressive train/validation runs while the remaining held-out
test episodes are still staging. Splits come from the selection file, not from a
fresh random split, so the final test set stays sealed.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from build_episode_manifest import add_toolkit_to_path, inspect_episode
from qwen3_omni_dataset_utils import VIDEO_NAMES


def parse_args() -> argparse.Namespace:
    workspace_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Build a manifest from selected Xperience-10M episodes.")
    parser.add_argument("--workspace", type=Path, default=workspace_default)
    parser.add_argument("--data-root", type=Path, action="append", required=True)
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--include-split", choices=["train", "val", "test"], action="append")
    parser.add_argument("--require-all-videos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-train-episodes", type=int, default=1)
    parser.add_argument("--min-val-episodes", type=int, default=1)
    parser.add_argument("--window-frames", type=int, default=20)
    parser.add_argument("--stride-frames", type=int, default=20)
    parser.add_argument("--min-label-fraction", type=float, default=0.6)
    return parser.parse_args()


def is_complete_episode(path: Path, require_all_videos: bool) -> tuple[bool, list[str]]:
    missing = []
    if not (path / "annotation.hdf5").is_file():
        missing.append("annotation.hdf5")
    if require_all_videos:
        for name in VIDEO_NAMES:
            if not (path / name).is_file():
                missing.append(name)
    elif not any((path / name).is_file() for name in VIDEO_NAMES):
        missing.append("any_mp4")
    return not missing, missing


def selected_episode_path(data_roots: list[Path], episode_path: str) -> Path | None:
    for root in data_roots:
        candidate = root / episode_path
        if candidate.exists():
            return candidate.resolve()
    return None


def write_report(path: Path, summary: dict) -> None:
    lines = [
        "# Progressive Train/Validation Manifest",
        "",
        f"- Selected episodes: `{summary['selected_episode_count']}`",
        f"- Included splits: `{', '.join(summary['included_splits'])}`",
        f"- Included complete episodes: `{summary['included_episode_count']}`",
        f"- Included split counts: `{summary['included_split_counts']}`",
        f"- Complete selected episodes by split: `{summary['complete_selected_split_counts']}`",
        f"- Available test episodes kept sealed: `{summary['sealed_test_episodes_available']}`",
        f"- Require all six videos: `{summary['require_all_videos']}`",
        "",
        "The manifest uses the split labels from the fixed 128-episode selection file.",
        "Episodes assigned to the held-out test split are reported but not included unless explicitly requested.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.workspace = args.workspace.expanduser().resolve()
    data_roots = [path.expanduser().resolve() for path in args.data_root]
    include_splits = args.include_split or ["train", "val"]
    add_toolkit_to_path(args.workspace)

    selection = json.loads(args.selection_json.expanduser().read_text(encoding="utf-8"))
    selected = selection.get("selected_episodes", [])
    if not selected:
        raise ValueError(f"No selected_episodes found in {args.selection_json}")

    episodes = []
    missing_or_incomplete = []
    complete_counts = Counter()
    selected_counts = Counter(str(item.get("split", "unspecified")) for item in selected)

    for item in selected:
        split = str(item.get("split", "unspecified"))
        rel_path = str(item.get("episode_path", ""))
        episode_dir = selected_episode_path(data_roots, rel_path)
        if episode_dir is None:
            missing_or_incomplete.append({"episode_path": rel_path, "split": split, "missing": ["episode_dir"]})
            continue
        complete, missing = is_complete_episode(episode_dir, args.require_all_videos)
        if not complete:
            missing_or_incomplete.append({"episode_path": rel_path, "split": split, "missing": missing})
            continue
        complete_counts[split] += 1
        if split not in include_splits:
            continue

        episode = inspect_episode(episode_dir / "annotation.hdf5", args)
        unique_id = rel_path.replace("/", "__")
        episode.update({
            "episode_id": unique_id,
            "source_episode_id": episode_dir.name,
            "episode_path": rel_path,
            "split": split,
            "selection_rank": item.get("selection_rank"),
            "selection_score": item.get("selection_score"),
            "size_band": item.get("size_band"),
            "top_level_session": item.get("top_level_session"),
        })
        episodes.append(episode)

    included_counts = Counter(ep["split"] for ep in episodes)
    if included_counts.get("train", 0) < args.min_train_episodes:
        raise SystemExit(f"Only {included_counts.get('train', 0)} train episodes available; need {args.min_train_episodes}.")
    if "val" in include_splits and included_counts.get("val", 0) < args.min_val_episodes:
        raise SystemExit(f"Only {included_counts.get('val', 0)} val episodes available; need {args.min_val_episodes}.")

    summary = {
        "selection_json": str(args.selection_json),
        "selected_episode_count": len(selected),
        "selected_split_counts": dict(selected_counts),
        "complete_selected_split_counts": dict(complete_counts),
        "included_splits": include_splits,
        "included_episode_count": len(episodes),
        "included_split_counts": dict(included_counts),
        "sealed_test_episodes_available": complete_counts.get("test", 0) if "test" not in include_splits else 0,
        "require_all_videos": args.require_all_videos,
        "train_minimal_bytes": sum(ep["train_minimal_bytes"] for ep in episodes),
        "total_bytes": sum(ep["total_bytes"] for ep in episodes),
        "windowing": {
            "window_frames": args.window_frames,
            "stride_frames": args.stride_frames,
            "min_label_fraction": args.min_label_fraction,
        },
        "notes": [
            "Splits are inherited from the fixed selected-episode file.",
            "Held-out test episodes are excluded by default for progressive train/validation runs.",
            "Episode ids are session-qualified to avoid collisions between repeated ep1/ep2 folder names.",
        ],
    }
    payload = {"summary": summary, "episodes": episodes, "missing_or_incomplete": missing_or_incomplete}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.report_output:
        write_report(args.report_output.expanduser(), summary)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

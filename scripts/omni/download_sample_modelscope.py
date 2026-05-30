#!/usr/bin/env python3
"""Download the public Xperience-10M sample from ModelScope.

This is the preferred path for servers inside mainland China. It downloads
only model-training files by default and skips visualization.rrd.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_PATTERNS = [
    "README.md",
    "annotation.hdf5",
    "fisheye_cam0.mp4",
]

ALL_TRAINING_PATTERNS = [
    "README.md",
    "annotation.hdf5",
    "fisheye_cam0.mp4",
    "fisheye_cam1.mp4",
    "fisheye_cam2.mp4",
    "fisheye_cam3.mp4",
    "stereo_left.mp4",
    "stereo_right.mp4",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Ropedia sample data from ModelScope.")
    parser.add_argument("--repo-id", default="ropedia-ai/xperience-10m-sample")
    parser.add_argument("--output-dir", type=Path, default=Path("data/sample/xperience-10m-sample"))
    parser.add_argument(
        "--mode",
        choices=["minimal", "all-training", "all"],
        default="minimal",
        help="minimal downloads annotation + one video; all-training adds all MP4s; all also allows visualization.rrd.",
    )
    parser.add_argument("--max-workers", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from modelscope.hub.snapshot_download import snapshot_download

    if args.mode == "minimal":
        allow_patterns = DEFAULT_PATTERNS
    elif args.mode == "all-training":
        allow_patterns = ALL_TRAINING_PATTERNS
    else:
        allow_patterns = None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(args.output_dir),
        allow_patterns=allow_patterns,
        max_workers=args.max_workers,
    )
    summary = {
        "repo_id": args.repo_id,
        "output_dir": str(args.output_dir),
        "mode": args.mode,
        "allow_patterns": allow_patterns,
        "download_path": path,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

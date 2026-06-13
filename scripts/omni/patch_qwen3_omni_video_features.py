#!/usr/bin/env python3
"""Patch/check the Qwen3-Omni video-feature compatibility issue.

Some Transformers 5.0.0 Qwen3-Omni builds unpack ``pooler_output`` as a pair in
the video branch. The source used for the verified v5/v6 runs reads
``pooler_output`` and ``deepstack_features`` separately. This helper keeps a
private staged GPU environment reproducible without replacing the whole
installed package by hand.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import sysconfig
import time
from pathlib import Path


EXPECTED_COMPAT_SHA256 = "da5feea4afc11767db3ca7eedb85ac129c66605643dadc6272c4288b03be7d25"
KNOWN_BAD_SHA256 = "2aa5752c32965dbaeee230a016afbbbb30d459a46a12c88c1d6f712e12ba95ad"

BAD_PATTERN = """video_embeds, video_embeds_multiscale = self.get_video_features(
                pixel_values_videos, video_grid_thw, return_dict=True
            ).pooler_output"""

GOOD_PATTERN = """video_outputs = self.get_video_features(pixel_values_videos, video_grid_thw, return_dict=True)
            video_embeds = video_outputs.pooler_output
            video_embeds_multiscale = video_outputs.deepstack_features"""

RELATIVE_MODELING_PATH = Path("transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locate_modeling_file(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()

    candidates = []
    paths = sysconfig.get_paths()
    for key in ("purelib", "platlib"):
        base = paths.get(key)
        if base:
            candidates.append(Path(base) / RELATIVE_MODELING_PATH)

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise SystemExit(
        "Could not find installed Transformers Qwen3-Omni modeling file. "
        "Pass --modeling-file explicitly."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modeling-file", type=Path, help="explicit modeling_qwen3_omni_moe.py path")
    parser.add_argument("--apply", action="store_true", help="apply the narrow video-feature patch if needed")
    parser.add_argument(
        "--strict-hash",
        action="store_true",
        help="fail unless the final file hash matches the verified compatible source hash",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = locate_modeling_file(args.modeling_file)
    if not path.is_file():
        raise SystemExit(f"Missing modeling file: {path}")

    before_hash = sha256(path)
    if before_hash == EXPECTED_COMPAT_SHA256:
        print(f"Qwen3-Omni modeling file already source-compatible: {path}")
        print(f"sha256={before_hash}")
        return 0

    text = path.read_text(encoding="utf-8")
    if BAD_PATTERN not in text:
        message = (
            f"Qwen3-Omni modeling file does not contain the known bad video-feature pattern: {path}\n"
            f"sha256={before_hash}"
        )
        if args.strict_hash:
            print(message, file=sys.stderr)
            return 1
        print(message)
        return 0

    if not args.apply:
        print(f"Known incompatible Qwen3-Omni video-feature pattern found: {path}", file=sys.stderr)
        print(f"sha256={before_hash}", file=sys.stderr)
        if before_hash == KNOWN_BAD_SHA256:
            print("This matches the observed pre-patch hash.", file=sys.stderr)
        return 1

    backup = path.with_name(f"{path.name}.video_features_prepatch_{int(time.time())}.bak")
    shutil.copy2(path, backup)
    path.write_text(text.replace(BAD_PATTERN, GOOD_PATTERN, 1), encoding="utf-8")
    after_hash = sha256(path)
    print(f"Patched Qwen3-Omni video-feature source: {path}")
    print(f"backup={backup}")
    print(f"sha256_before={before_hash}")
    print(f"sha256_after={after_hash}")

    if args.strict_hash and after_hash != EXPECTED_COMPAT_SHA256:
        print(
            "Patched file does not match the verified compatible source hash; inspect before running eval.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

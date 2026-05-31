#!/usr/bin/env python3
"""Upload the prepared Qwen3-Omni LoRA pilot adapter to Hugging Face."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    proc = subprocess.run(cmd, check=True, env=env)
    if proc.returncode:
        raise SystemExit(f"command failed: {' '.join(cmd)}")


def build_repo(repo_id: str, token: str, private: bool) -> None:
    # `repo create` returns non-zero when repo already exists.
    name = repo_id.split("/", 1)[1] if "/" in repo_id else repo_id
    namespace = repo_id.split("/", 2)[0] if "/" in repo_id else None
    cmd = [
        "huggingface-cli",
        "repo",
        "create",
        name,
        "--type",
        "model",
        "--yes",
    ]
    if private:
        cmd.append("--private")
    if namespace:
        cmd.extend(["--organization", namespace])
    cmd.extend(["--token", token])
    subprocess.run(cmd, check=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="Target repo id, e.g. cy0307/xxx")
    parser.add_argument(
        "--source-dir",
        default="results/omni_finetune/hf_upload",
        help="Directory that contains adapter files and README",
    )
    parser.add_argument(
        "--token",
        required=False,
        default=None,
        help="HF token; otherwise reads HF_TOKEN env var",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create repo as private if not exists",
    )
    parser.add_argument(
        "--message",
        default="Upload Qwen3-Omni LoRA pilot adapter",
        help="Commit message",
    )

    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN", "")
    if not token:
        raise SystemExit("HF token missing; pass --token or set HF_TOKEN")

    source = Path(args.source_dir).expanduser().resolve()
    if not source.is_dir():
        raise SystemExit(f"source directory does not exist: {source}")

    build_repo(args.repo_id, token, args.private)
    run(
        [
            "huggingface-cli",
            "upload",
            args.repo_id,
            str(source),
            ".",
            "--token",
            token,
            "--commit-message",
            args.message,
            "--repo-type",
            "model",
        ]
    )
    print(f"Uploaded {source} -> https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

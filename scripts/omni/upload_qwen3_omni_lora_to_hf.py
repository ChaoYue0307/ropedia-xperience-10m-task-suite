#!/usr/bin/env python3
"""Upload the prepared Qwen3-Omni LoRA pilot adapter to Hugging Face."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def upload_folder(repo_id: str, source: Path, token: str, private: bool, message: str) -> str:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("huggingface_hub is required for upload; install it in this environment") from exc

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    result = api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(source),
        path_in_repo=".",
        commit_message=message,
    )
    return str(result)


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

    result = upload_folder(args.repo_id, source, token, args.private, args.message)
    print(f"Uploaded {source} -> https://huggingface.co/{args.repo_id}")
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

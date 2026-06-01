#!/usr/bin/env python3
"""Publish prepared Hugging Face bundles for the Xperience-10M task suite.

The repo itself is the source of truth for code, docs, validators, and website
assets. The prepared Hugging Face folders live outside the repo by default:

    ../hf_publish/space
    ../hf_publish/artifacts
    ../hf_publish/model

This script uploads those prepared folders and handles model binaries as an
explicit second model-repo batch so `.npz` weights and `.pt` checkpoints cannot
silently drift behind the model card.
"""

from __future__ import annotations

import argparse
import getpass
import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HF_ROOT = ROOT.parent / "hf_publish"
DEFAULT_NAMESPACE = "cy0307"
DEFAULT_SPACE_REPO = "ropedia-xperience-10m-task-suite"
DEFAULT_ARTIFACT_REPO = "ropedia-xperience-10m-task-suite-artifacts"
DEFAULT_MODEL_REPO = "ropedia-xperience-10m-task-baselines"
COLLECTION_TITLE = "Ropedia Xperience-10M Task Suite"

COMMON_IGNORE = [
    ".DS_Store",
    "__pycache__/*",
    "**/__pycache__/*",
    "*.pyc",
    ".git/*",
]

STALE_ARTIFACT_REMOTE_FILES = [
    "results/omni_finetune/adapter_lora/tokenizer.json",
    "results/omni_finetune/hf_upload/tokenizer.json",
    "REVIEWER_SCORECARD.md",
    "docs/data/reviewer_packet.json",
    "docs/data/reviewer_scorecard.json",
]

STALE_SPACE_REMOTE_FILES = [
    "REVIEWER_SCORECARD.md",
    "data/reviewer_packet.json",
    "data/reviewer_scorecard.json",
]

STALE_MODEL_REMOTE_FILES = [
    "REVIEWER_SCORECARD.md",
    "metrics/reviewer_packet.json",
    "metrics/reviewer_scorecard.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-root", type=Path, default=DEFAULT_HF_ROOT)
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--space-repo", default=DEFAULT_SPACE_REPO)
    parser.add_argument("--artifact-repo", default=DEFAULT_ARTIFACT_REPO)
    parser.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN", "").strip())
    return parser.parse_args()


def full_repo(namespace: str, repo_name: str) -> str:
    return repo_name if "/" in repo_name else f"{namespace}/{repo_name}"


def prune_generated_artifacts(root: Path) -> None:
    for cache_dir in sorted(root.rglob("__pycache__"), reverse=True):
        shutil.rmtree(cache_dir, ignore_errors=True)
    for cache_file in root.rglob("*.pyc"):
        cache_file.unlink(missing_ok=True)
    for junk_file in root.rglob(".DS_Store"):
        junk_file.unlink(missing_ok=True)


def prune_artifact_bundle(hf_root: Path) -> None:
    artifact_root = hf_root / "artifacts"
    for relative_path in STALE_ARTIFACT_REMOTE_FILES:
        (artifact_root / relative_path).unlink(missing_ok=True)


def upload_folder(
    api: HfApi,
    token: str,
    repo_id: str,
    repo_type: str | None,
    folder: Path,
    message: str,
    *,
    allow_patterns: list[str] | None = None,
    ignore_patterns: list[str] | None = None,
):
    print(f"Uploading {folder} -> {repo_id}")
    return api.upload_folder(
        repo_id=repo_id,
        repo_type=repo_type,
        folder_path=str(folder),
        commit_message=message,
        token=token,
        allow_patterns=allow_patterns,
        ignore_patterns=COMMON_IGNORE + (ignore_patterns or []),
    )


def delete_remote_file_if_present(
    api: HfApi,
    token: str,
    repo_id: str,
    repo_type: str,
    path_in_repo: str,
) -> None:
    try:
        api.delete_file(
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type=repo_type,
            token=token,
            commit_message=f"Remove stale {path_in_repo}",
        )
        print(f"Deleted stale remote file: {repo_id}/{path_in_repo}")
    except Exception as exc:
        message = str(exc)
        if "404" in message or "Entry Not Found" in message or "not found" in message.lower():
            print(f"Remote file already absent: {repo_id}/{path_in_repo}")
            return
        print(f"Remote stale-file cleanup skipped for {repo_id}/{path_in_repo}: {exc}")


def main() -> int:
    args = parse_args()
    hf_root = args.hf_root.resolve()
    prune_generated_artifacts(hf_root)
    prune_artifact_bundle(hf_root)

    token = args.token or getpass.getpass("HF token: ").strip()
    if not token:
        raise SystemExit("No token provided.")

    api = HfApi(token=token)
    me = api.whoami(token=token)
    username = me.get("name")
    if username != args.namespace:
        raise SystemExit(f"Authenticated as {username!r}, expected {args.namespace!r}.")

    space_repo = full_repo(args.namespace, args.space_repo)
    artifact_repo = full_repo(args.namespace, args.artifact_repo)
    model_repo = full_repo(args.namespace, args.model_repo)

    api.create_repo(space_repo, repo_type="space", space_sdk="static", exist_ok=True, token=token)
    api.create_repo(artifact_repo, repo_type="dataset", exist_ok=True, token=token)
    api.create_repo(model_repo, repo_type=None, exist_ok=True, token=token)

    upload_folder(
        api,
        token,
        space_repo,
        "space",
        hf_root / "space",
        "Publish Ropedia Xperience-10M task-suite Space",
    )
    for path_in_repo in STALE_SPACE_REMOTE_FILES:
        delete_remote_file_if_present(api, token, space_repo, "space", path_in_repo)
    upload_folder(
        api,
        token,
        artifact_repo,
        "dataset",
        hf_root / "artifacts",
        "Publish Ropedia Xperience-10M derived artifacts",
        ignore_patterns=["**/*.pt", "**/*.npz"],
    )
    for path_in_repo in STALE_ARTIFACT_REMOTE_FILES:
        delete_remote_file_if_present(api, token, artifact_repo, "dataset", path_in_repo)
    upload_folder(
        api,
        token,
        model_repo,
        None,
        hf_root / "model",
        "Publish Ropedia Xperience-10M task baseline cards",
        ignore_patterns=["**/*.pt", "**/*.npz"],
    )
    for path_in_repo in STALE_MODEL_REMOTE_FILES:
        delete_remote_file_if_present(api, token, model_repo, "model", path_in_repo)
    upload_folder(
        api,
        token,
        model_repo,
        None,
        hf_root / "model",
        "Publish Ropedia Xperience-10M model binaries",
        allow_patterns=["**/*.npz", "**/*.pt"],
    )

    try:
        collection = api.create_collection(
            COLLECTION_TITLE,
            namespace=args.namespace,
            description=(
                "Space, artifact dataset, and minimal plus neural baseline model repos "
                "for the Ropedia Xperience-10M single-episode task suite."
            ),
            private=False,
            exists_ok=True,
            token=token,
        )
        api.add_collection_item(collection.slug, space_repo, "space", note="Interactive/static dashboard.", exists_ok=True, token=token)
        api.add_collection_item(collection.slug, artifact_repo, "dataset", note="Derived metrics, predictions, scripts, and diagrams.", exists_ok=True, token=token)
        api.add_collection_item(collection.slug, model_repo, "model", note="Minimal numpy weights plus neural MLP checkpoints.", exists_ok=True, token=token)
        print(f"Collection: https://huggingface.co/collections/{collection.slug}")
    except Exception as exc:
        print(f"Collection update skipped: {exc}")

    print("Done")
    print(f"Space: https://huggingface.co/spaces/{space_repo}")
    print(f"Artifacts: https://huggingface.co/datasets/{artifact_repo}")
    print(f"Models: https://huggingface.co/{model_repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

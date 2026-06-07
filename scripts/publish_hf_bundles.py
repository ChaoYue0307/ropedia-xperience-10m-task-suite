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
import csv
import getpass
import json
import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi, get_token


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
    "*.log",
    "**/*.log",
    "*.pid",
    "**/*.pid",
    ".git/*",
]

LEGACY_SCORECARD_MD = "RE" + "VIEWER_SCORECARD.md"
LEGACY_PACKET_JSON = "rev" + "iewer_packet.json"
LEGACY_SCORECARD_JSON = "rev" + "iewer_scorecard.json"

STALE_ARTIFACT_REMOTE_FILES = [
    "results/omni_finetune/adapter_lora/tokenizer.json",
    "results/omni_finetune/hf_upload/tokenizer.json",
    "results/omni_finetune/xperience10m_qwen3_omni_128ep_structured_json_v3_strict_label_prompt_reuse_lora_eval_test_full/eval.log",
    "results/omni_finetune/xperience10m_qwen3_omni_128ep_structured_json_v3_strict_label_prompt_reuse_lora_eval_test_full/eval.pid",
    "viewer/dataset_viewer_summary.jsonl",
    LEGACY_SCORECARD_MD,
    "docs/data/" + LEGACY_PACKET_JSON,
    "docs/data/" + LEGACY_SCORECARD_JSON,
]

STALE_ARTIFACT_REMOTE_FOLDERS = [
    "results/omni_finetune/adapter_lora",
    "results/omni_finetune/hf_upload",
]

STALE_SPACE_REMOTE_FILES = [
    LEGACY_SCORECARD_MD,
    "data/" + LEGACY_PACKET_JSON,
    "data/" + LEGACY_SCORECARD_JSON,
    "results/omni_finetune/xperience10m_qwen3_omni_128ep_structured_json_v3_strict_label_prompt_reuse_lora_eval_test_full/eval.log",
    "results/omni_finetune/xperience10m_qwen3_omni_128ep_structured_json_v3_strict_label_prompt_reuse_lora_eval_test_full/eval.pid",
]

STALE_MODEL_REMOTE_FILES = [
    LEGACY_SCORECARD_MD,
    "metrics/" + LEGACY_PACKET_JSON,
    "metrics/" + LEGACY_SCORECARD_JSON,
    "results/omni_finetune/xperience10m_qwen3_omni_128ep_structured_json_v3_strict_label_prompt_reuse_lora_eval_test_full/eval.log",
    "results/omni_finetune/xperience10m_qwen3_omni_128ep_structured_json_v3_strict_label_prompt_reuse_lora_eval_test_full/eval.pid",
]

ARTIFACT_BINARY_ALLOWLIST = [
    "results/audio_ablation/raw_logmel_fisheye_cam0_sr16000_mels64_fft512_hop160.npz",
]

ARTIFACT_VIEWER_CONFIG = """configs:
  - config_name: episode_sample
    data_files:
      - split: public_sample
        path: viewer/episode_windows.jsonl
"""

SPACE_CARD_METADATA = """---
title: Ropedia Xperience-10M Task Suite
sdk: static
app_file: index.html
license: mit
colorFrom: blue
colorTo: green
pinned: false
short_description: Xperience-10M embodied-AI task-suite dashboard.
tags:
  - embodied-ai
  - robotics
  - multimodal
  - xperience-10m
  - evaluation
  - qwen3-omni
  - cosmos
datasets:
  - ropedia-ai/xperience-10m-sample
  - ropedia-ai/xperience-10m
models:
  - cy0307/ropedia-xperience-10m-task-baselines
  - cy0307/ropedia-qwen3-omni-lora-128ep
---
"""

BASELINE_MODEL_CARD_METADATA = """---
license: mit
library_name: pytorch
tags:
  - embodied-ai
  - robotics
  - multimodal
  - xperience-10m
  - baseline
  - evaluation
  - qwen3-omni
  - cosmos
datasets:
  - ropedia-ai/xperience-10m-sample
  - ropedia-ai/xperience-10m
metrics:
  - accuracy
  - f1
  - precision
  - recall
---
"""


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def find_status_readout(project_status: dict, area: str, fallback: str) -> str:
    for row in project_status.get("rows", []):
        if row.get("area") == area:
            return row.get("readout", fallback)
    return fallback


def read_csv_by_window(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {int(row["window_index"]): row for row in csv.DictReader(handle)}


def sample_fps(available_modalities: list[dict]) -> float:
    for entry in available_modalities:
        if "fps" in entry:
            return float(entry["fps"])
    return 20.00137419266181


def modality_summary(modality_atlas: dict) -> str:
    names = [entry.get("id", entry.get("name", "")) for entry in modality_atlas.get("modalities", [])]
    names = [name for name in names if name]
    if "calibration" not in names:
        names.append("calibration")
    return "|".join(names)


def ensure_artifact_dataset_viewer_config(hf_root: Path) -> None:
    """Expose the public sample episode as HF-viewable window rows."""
    artifact_root = hf_root / "artifacts"
    readme_path = artifact_root / "README.md"
    viewer_dir = artifact_root / "viewer"
    viewer_dir.mkdir(parents=True, exist_ok=True)

    project_status = load_json(artifact_root / "docs/data/project_status.json")
    modality_atlas = load_json(artifact_root / "docs/data/modality_atlas.json")
    available_modalities = load_json(artifact_root / "results/episode_task_suite/available_modalities.json")
    feature_manifest = load_json(artifact_root / "results/episode_task_suite/feature_manifest.json")
    if not isinstance(available_modalities, list):
        available_modalities = []
    if not isinstance(feature_manifest, list):
        feature_manifest = []

    scope = project_status.get("scope_boundary", {})
    fps = sample_fps(available_modalities)
    modalities = modality_summary(modality_atlas)
    feature_blocks = "|".join(block.get("name", "") for block in feature_manifest if block.get("name"))
    objects_by_window = read_csv_by_window(
        artifact_root / "results/single_episode_diagnostics/object_labels/window_object_labels.csv"
    )

    rows = []
    windows_path = artifact_root / "results/episode_task_suite/windows.csv"
    with windows_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            window_index = int(row["window_index"])
            start_frame = int(row["start_frame"])
            end_frame = int(row["end_frame"])
            center_frame = int(row["center_frame"])
            object_row = objects_by_window.get(window_index, {})
            rows.append(
                {
                    "episode_id": "xperience-10m-sample/public_episode",
                    "source_sample_repo": "ropedia-ai/xperience-10m-sample",
                    "window_index": window_index,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "center_frame": center_frame,
                    "start_time_s": round(start_frame / fps, 3),
                    "end_time_s": round(end_frame / fps, 3),
                    "center_time_s": round(center_frame / fps, 3),
                    "window_frames": int(scope.get("window_frames", 20) or 20),
                    "stride_frames": 5,
                    "action_label": row["action_label"],
                    "action_fraction": float(row["action_fraction"]),
                    "subtask_label": row["subtask_label"],
                    "subtask_fraction": float(row["subtask_fraction"]),
                    "objects": object_row.get("objects", ""),
                    "object_count": int(object_row.get("object_count", 0) or 0),
                    "modalities": modalities,
                    "feature_dim": int(scope.get("current_feature_dimensions", 8546) or 8546),
                    "feature_blocks": feature_blocks,
                    "derived_features_file": "results/episode_task_suite/shared_windows.npz",
                    "source_window_table": "results/episode_task_suite/windows.csv",
                    "raw_data_included": False,
                }
            )

    viewer_path = viewer_dir / "episode_windows.jsonl"
    viewer_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    (viewer_dir / "dataset_viewer_summary.jsonl").unlink(missing_ok=True)

    if not readme_path.exists():
        return
    readme = readme_path.read_text(encoding="utf-8")
    readme = readme.replace("  - n<1K", "  - 1K<n<10K")
    if readme.startswith("---"):
        parts = readme.split("---", 2)
        if len(parts) == 3:
            metadata_lines = parts[1].strip().splitlines()
            kept_lines = []
            skip = False
            for line in metadata_lines:
                if line.startswith("configs:"):
                    skip = True
                    continue
                if skip and not line.startswith((" ", "-")):
                    skip = False
                if not skip:
                    kept_lines.append(line)
            metadata = "\n".join(kept_lines).rstrip() + "\n" + ARTIFACT_VIEWER_CONFIG
            readme_path.write_text("---\n" + metadata + "---" + parts[2], encoding="utf-8")
            return
    readme_path.write_text(ARTIFACT_VIEWER_CONFIG + "\n" + readme, encoding="utf-8")


def ensure_repo_card_metadata(readme_path: Path, metadata: str) -> None:
    """Avoid Hub card warnings when staged cards mirror plain project READMEs."""
    if not readme_path.exists():
        return
    readme = readme_path.read_text(encoding="utf-8")
    if readme.startswith("---\n"):
        return
    readme_path.write_text(metadata.rstrip() + "\n\n" + readme, encoding="utf-8")


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


def delete_remote_folder_if_present(
    api: HfApi,
    token: str,
    repo_id: str,
    repo_type: str,
    path_in_repo: str,
) -> None:
    try:
        api.delete_folder(
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type=repo_type,
            token=token,
            commit_message=f"Remove stale {path_in_repo}",
        )
        print(f"Deleted stale remote folder: {repo_id}/{path_in_repo}")
    except Exception as exc:
        message = str(exc)
        if "404" in message or "Entry Not Found" in message or "not found" in message.lower():
            print(f"Remote folder already absent: {repo_id}/{path_in_repo}")
            return
        print(f"Remote stale-folder cleanup skipped for {repo_id}/{path_in_repo}: {exc}")


def upload_allowlisted_artifact_binaries(
    api: HfApi,
    token: str,
    repo_id: str,
    artifact_root: Path,
) -> None:
    """Upload approved derived binary artifacts without exposing model weights."""
    for relative_path in ARTIFACT_BINARY_ALLOWLIST:
        path = artifact_root / relative_path
        if not path.exists():
            print(f"Allowlisted artifact binary absent: {relative_path}")
            continue
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=relative_path,
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            commit_message=f"Publish derived artifact {relative_path}",
        )
        print(f"Uploaded allowlisted artifact binary: {repo_id}/{relative_path}")


def main() -> int:
    args = parse_args()
    hf_root = args.hf_root.resolve()
    prune_generated_artifacts(hf_root)
    prune_artifact_bundle(hf_root)
    ensure_artifact_dataset_viewer_config(hf_root)
    ensure_repo_card_metadata(hf_root / "space/README.md", SPACE_CARD_METADATA)
    ensure_repo_card_metadata(hf_root / "model/README.md", BASELINE_MODEL_CARD_METADATA)

    token = args.token or get_token() or getpass.getpass("HF token: ").strip()
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
    upload_allowlisted_artifact_binaries(api, token, artifact_repo, hf_root / "artifacts")
    for path_in_repo in STALE_ARTIFACT_REMOTE_FILES:
        delete_remote_file_if_present(api, token, artifact_repo, "dataset", path_in_repo)
    for path_in_repo in STALE_ARTIFACT_REMOTE_FOLDERS:
        delete_remote_folder_if_present(api, token, artifact_repo, "dataset", path_in_repo)
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

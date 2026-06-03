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
import json
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

LEGACY_SCORECARD_MD = "RE" + "VIEWER_SCORECARD.md"
LEGACY_PACKET_JSON = "rev" + "iewer_packet.json"
LEGACY_SCORECARD_JSON = "rev" + "iewer_scorecard.json"

STALE_ARTIFACT_REMOTE_FILES = [
    "results/omni_finetune/adapter_lora/tokenizer.json",
    "results/omni_finetune/hf_upload/tokenizer.json",
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
]

STALE_MODEL_REMOTE_FILES = [
    LEGACY_SCORECARD_MD,
    "metrics/" + LEGACY_PACKET_JSON,
    "metrics/" + LEGACY_SCORECARD_JSON,
]

ARTIFACT_BINARY_ALLOWLIST = [
    "results/audio_ablation/raw_logmel_fisheye_cam0_sr16000_mels64_fft512_hop160.npz",
]

ARTIFACT_VIEWER_CONFIG = """configs:
  - config_name: viewer_summary
    data_files:
      - split: train
        path: viewer/dataset_viewer_summary.jsonl
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


def ensure_artifact_dataset_viewer_config(hf_root: Path) -> None:
    """Keep HF Dataset Viewer on one JSONL config despite mixed artifact files."""
    artifact_root = hf_root / "artifacts"
    readme_path = artifact_root / "README.md"
    viewer_dir = artifact_root / "viewer"
    viewer_dir.mkdir(parents=True, exist_ok=True)

    project_status = load_json(artifact_root / "docs/data/project_status.json")
    summary_metrics = load_json(artifact_root / "docs/data/summary_metrics.json")
    alignment = load_json(artifact_root / "docs/data/xperience10m_dataset_card_alignment.json")

    scope = project_status.get("scope_boundary", {})
    official = alignment.get("hf_repo_metadata_observed", {})
    live_size = official.get("live_hf_page_observed", {}).get("total_file_size_display", "unknown")
    listing = official.get("api_file_listing_observed", {})
    relay = summary_metrics.get("omni_relay", {})

    rows = [
        {
            "row_type": "project_scope",
            "title": "Public sample pipeline",
            "value": (
                f"{scope.get('validated_episode_count', 1)} public Xperience-10M sample episode; "
                f"{scope.get('aligned_frames', 5821):,} frames; "
                f"{scope.get('sliding_windows', 1161):,} aligned 20-frame windows; "
                f"{scope.get('current_feature_dimensions', 8546):,}-dimensional multimodal representation"
            ),
            "source_file": "docs/data/project_status.json",
            "notes": "This artifact repo stores derived outputs only and does not redistribute raw Xperience-10M MP4/HDF5/RRD data or full Qwen weights.",
        },
        {
            "row_type": "modality_contract",
            "title": "Modalities represented",
            "value": "video, audio, depth, camera pose/SLAM, motion capture, inertial sensing, calibration metadata, and language annotations",
            "source_file": "docs/data/xperience10m_dataset_card_alignment.json",
            "notes": "Audio is featurized in the current public-sample task representation; raw synchronized data remains governed by the official upstream dataset terms.",
        },
        {
            "row_type": "task_suite",
            "title": "Embodied task suite",
            "value": (
                f"{scope.get('core_task_count', 12)} human-readable task contracts with minimal baselines "
                f"and {scope.get('neural_head_count', 12)} compact neural MLP heads over the same chronological split"
            ),
            "source_file": "results/episode_task_suite/summary_report.json",
            "notes": "These are single-episode public-sample results, not held-out multi-episode model-quality claims.",
        },
        {
            "row_type": "audio_study",
            "title": "Audio contribution study",
            "value": find_status_readout(
                project_status,
                "Audio contribution study",
                "Audio contribution results are recorded in results/audio_ablation/.",
            ),
            "source_file": "docs/data/project_status.json",
            "notes": "See results/audio_ablation/ for the per-task audio/no-audio comparison and raw-logmel feature artifact.",
        },
        {
            "row_type": "neural_baselines",
            "title": "Compact neural heads",
            "value": (
                f"{scope.get('neural_head_count', 12)} PyTorch MLP task heads plus "
                f"{scope.get('direction_extension_probe_count', 4)} extension probes are included as derived artifacts"
            ),
            "source_file": "docs/data/project_status.json",
            "notes": "The neural heads are local task baselines over derived features, separate from the pending Qwen3-Omni multi-episode pilot.",
        },
        {
            "row_type": "official_upstream",
            "title": "Official Xperience-10M dataset",
            "value": (
                f"Official gated HF metadata observed: {listing.get('annotation_hdf5_count', 12103):,} annotation files, "
                f"{listing.get('episode_folder_count', 12103) - 1:,} complete visible episodes, "
                f"{listing.get('sibling_count', 85258):,}+ listed files, and {live_size} live HF-hosted file-size display"
            ),
            "source_file": "docs/data/xperience10m_dataset_card_alignment.json",
            "notes": "The official dataset card also describes about 10M experience units, about 10,000 recording hours, and about 1 PB full-scale data; this artifact repo is not the raw dataset mirror.",
        },
        {
            "row_type": "multi_episode_plan",
            "title": "Selected multi-episode relay",
            "value": (
                f"{relay.get('target_episodes', 128)} metadata-balanced episodes selected for future held-out training "
                "with a 96/16/16 train/val/test target split"
            ),
            "source_file": "results/omni_finetune/DATA_ACCESS_STATUS.md",
            "notes": "Staging, manifest construction, training, and held-out evaluation must complete before reporting full multi-episode metrics.",
        },
        {
            "row_type": "foundation_model_plan",
            "title": "Foundation-model branches",
            "value": "Qwen3-Omni is the first trainable LoRA pilot; Cosmos 3 is planned as the world-model/action-generation branch; policy candidates include OpenVLA, openpi, and GR00T after action targets are explicit",
            "source_file": "docs/data/project_status.json",
            "notes": "The current package documents the plan and public-sample baselines; it does not claim a completed full-dataset fine-tune.",
        },
        {
            "row_type": "viewer_config",
            "title": "Dataset Viewer configuration",
            "value": "This viewer_summary config points HF Dataset Viewer to one JSONL file so mixed artifact formats are not interpreted as incompatible splits",
            "source_file": "viewer/dataset_viewer_summary.jsonl",
            "notes": "The rest of the repo intentionally keeps CSV, JSON, JSONL, Markdown, HTML, NPZ, PT, PNG, and SVG artifacts for reproducibility and presentation.",
        },
    ]

    viewer_path = viewer_dir / "dataset_viewer_summary.jsonl"
    viewer_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n",
        encoding="utf-8",
    )

    if not readme_path.exists():
        return
    readme = readme_path.read_text(encoding="utf-8")
    if "viewer/dataset_viewer_summary.jsonl" in readme:
        return
    if readme.startswith("---"):
        parts = readme.split("---", 2)
        if len(parts) == 3:
            metadata = parts[1].rstrip() + "\n" + ARTIFACT_VIEWER_CONFIG
            readme_path.write_text("---" + metadata + "---" + parts[2], encoding="utf-8")
            return
    readme_path.write_text(ARTIFACT_VIEWER_CONFIG + "\n" + readme, encoding="utf-8")


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

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
DEFAULT_WEIGHTS_RESULTS_REPO = "ropedia-xperience-10m-weights-results"
DEFAULT_QWEN3_LORA_REPO = "ropedia-qwen3-omni-lora-128ep"
DEFAULT_COSMOS3_SUPER_LORA_REPO = "ropedia-cosmos3-super-forward-dynamics-lora-128ep"
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
    "README_GRADIO_RUNTIME.md",
    "README_SPACE_RUNTIME.md",
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
        path: viewer/episode_windows.parquet
  - config_name: selected_128_windows
    data_files:
      - split: selected_128
        path: viewer/selected128_windows.parquet
"""
ENHANCEMENT_MARKER = "docs/data/task_suite_enhancement_128.json"
ENHANCEMENT_CARD_BLOCK = """
## 128-Episode Enhancement Pack

The no-new-episode suite push is recorded in `TASK_SUITE_ENHANCEMENT_128.md`
and `docs/data/task_suite_enhancement_128.json`. It recommends
`multiscale_20s10_40s20_80s40`, hierarchical action/subtask targets,
label-normalized scoring, and compact raw-feature shards before adding more
episodes.
"""

SPACE_CARD_METADATA = """---
title: Ropedia Xperience-10M Task Suite
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
license: mit
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
  - cy0307/ropedia-xperience-10m-weights-results
  - cy0307/ropedia-qwen3-omni-lora-128ep
  - cy0307/ropedia-cosmos3-super-forward-dynamics-lora-128ep
---
"""

SPACE_REQUIREMENTS = """gradio>=4.44.0
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


def parse_multiscale_window_id(window_id: str) -> dict[str, int | str]:
    prefix = window_id.split(":", 1)[0]
    parts = prefix.split("_")
    parsed: dict[str, int | str] = {
        "window_scale": parts[0] if parts else "",
        "window_frames": 0,
        "stride_frames": 0,
    }
    for part in parts:
        if part.endswith("f") and part[:-1].isdigit():
            parsed["window_frames"] = int(part[:-1])
        elif part.startswith("stride") and part.removeprefix("stride").isdigit():
            parsed["stride_frames"] = int(part.removeprefix("stride"))
    return parsed


def selected128_episode_key_to_path(episode_key: str) -> str:
    if "__" not in episode_key:
        return episode_key.replace("__", "/")
    session_id, ep_id = episode_key.split("__", 1)
    return f"{session_id}/{ep_id}"


def write_selected128_viewer_table(artifact_root: Path, viewer_dir: Path) -> None:
    """Expose selected-128 exported windows as a separate HF dataset config."""
    windows_path = artifact_root / "results/omni_finetune/a100_128_metadata_task_baselines_20260616_v2/windows.csv"
    feature_index_path = artifact_root / "docs/data/xperience10m_128_episode_feature_index.json"
    feature_index = load_json(feature_index_path)
    selection_summary = feature_index.get("selection_summary", {})
    processed_summary = feature_index.get("processed_summary", {})
    qwen_export = processed_summary.get("qwen_v6_multiscale_export", {})
    selected_source_episode_count = int(selection_summary.get("selected_episode_count", 128) or 128)
    expected_rows = int(qwen_export.get("num_samples", 34269) or 34269)
    expected_window_episode_count = int(qwen_export.get("num_episodes", 119) or 119)

    rows = []
    with windows_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            parsed = parse_multiscale_window_id(row["id"])
            start_frame = int(row["start_frame"])
            end_frame = int(row["end_frame"])
            row_id = row["id"]
            episode_id = row["episode_id"]
            rows.append(
                {
                    "evidence_line": "selected_128_episodes",
                    "source_dataset_repo": "ropedia-ai/xperience-10m",
                    "source_access": "gated_upstream_not_redistributed",
                    "source_episode_id": episode_id,
                    "official_episode_path": selected128_episode_key_to_path(episode_id),
                    "window_id": row_id,
                    "window_scale": parsed["window_scale"],
                    "window_frames": parsed["window_frames"],
                    "stride_frames": parsed["stride_frames"],
                    "split": row["split"],
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "center_frame": (start_frame + end_frame) // 2,
                    "main_task": row["main_task"],
                    "selected_source_episode_count": selected_source_episode_count,
                    "exported_window_episode_count": expected_window_episode_count,
                    "exported_window_count": expected_rows,
                    "split_policy": "selected 96/16/16 episode split",
                    "feature_index": "docs/data/xperience10m_128_episode_feature_index.json",
                    "source_window_table": windows_path.relative_to(artifact_root).as_posix(),
                    "raw_data_included": False,
                }
            )

    if len(rows) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} selected-128 rows, found {len(rows)} in {windows_path}")
    episode_count = len({row["source_episode_id"] for row in rows})
    if episode_count != expected_window_episode_count:
        raise RuntimeError(f"Expected {expected_window_episode_count} exported-window episodes, found {episode_count}")
    if selected_source_episode_count < episode_count:
        raise RuntimeError(
            f"Selected source episode count {selected_source_episode_count} is smaller than exported episode count {episode_count}"
        )

    jsonl_path = viewer_dir / "selected128_windows.jsonl"
    jsonl_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    try:
        import pandas as pd

        parquet_path = viewer_dir / "selected128_windows.parquet"
        pd.DataFrame(rows).to_parquet(parquet_path, index=False)
    except ImportError:
        print("pandas/pyarrow unavailable; wrote selected-128 JSONL viewer fallback only")


def ensure_artifact_dataset_viewer_config(hf_root: Path) -> None:
    """Expose public sample and selected-128 windows as separate HF-viewable tables."""
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
    try:
        import pandas as pd

        parquet_path = viewer_dir / "episode_windows.parquet"
        pd.DataFrame(rows).to_parquet(parquet_path, index=False)
    except ImportError:
        print("pandas/pyarrow unavailable; wrote JSONL viewer fallback only")
    write_selected128_viewer_table(artifact_root, viewer_dir)
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
    normalized_metadata = metadata.rstrip() + "\n\n"
    if readme.startswith("---\n"):
        parts = readme.split("---", 2)
        if len(parts) == 3:
            new_readme = normalized_metadata + parts[2].lstrip("\n")
            if new_readme != readme:
                readme_path.write_text(new_readme, encoding="utf-8")
            return
    new_readme = normalized_metadata + readme.lstrip("\n")
    if new_readme != readme:
        readme_path.write_text(new_readme, encoding="utf-8")


def ensure_space_runtime_files(hf_root: Path) -> None:
    """Keep the Hub Space runtime small and explicit."""
    requirements_path = hf_root / "space/requirements.txt"
    requirements_path.write_text(SPACE_REQUIREMENTS, encoding="utf-8")


def ensure_enhancement_card_links(hf_root: Path) -> None:
    for relative_path in ("artifacts/README.md", "model/README.md"):
        path = hf_root / relative_path
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        if ENHANCEMENT_MARKER in text:
            continue
        insert_before = "\n## Dataset Boundary" if relative_path.startswith("artifacts/") else "\n## Start Here"
        if insert_before in text:
            text = text.replace(insert_before, ENHANCEMENT_CARD_BLOCK + insert_before, 1)
        else:
            text = text.rstrip() + "\n" + ENHANCEMENT_CARD_BLOCK
        path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-root", type=Path, default=DEFAULT_HF_ROOT)
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--space-repo", default=DEFAULT_SPACE_REPO)
    parser.add_argument("--artifact-repo", default=DEFAULT_ARTIFACT_REPO)
    parser.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    parser.add_argument("--weights-results-repo", default=DEFAULT_WEIGHTS_RESULTS_REPO)
    parser.add_argument("--qwen3-lora-repo", default=DEFAULT_QWEN3_LORA_REPO)
    parser.add_argument("--cosmos3-super-lora-repo", default=DEFAULT_COSMOS3_SUPER_LORA_REPO)
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN", "").strip())
    parser.add_argument("--skip-space", action="store_true")
    parser.add_argument("--skip-artifacts", action="store_true")
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--skip-weights-results", action="store_true")
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
    effective_repo_type = repo_type or "model"
    effective_ignore_patterns = COMMON_IGNORE + (ignore_patterns or [])
    if effective_repo_type != "space" and hasattr(api, "upload_large_folder"):
        return api.upload_large_folder(
            repo_id=repo_id,
            repo_type=effective_repo_type,
            folder_path=str(folder),
            allow_patterns=allow_patterns,
            ignore_patterns=effective_ignore_patterns,
            num_workers=8,
            print_report=True,
            print_report_every=60,
        )
    return api.upload_folder(
        repo_id=repo_id,
        repo_type=repo_type,
        folder_path=str(folder),
        commit_message=message,
        token=token,
        allow_patterns=allow_patterns,
        ignore_patterns=effective_ignore_patterns,
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


def upsert_collection_item_notes(
    api: HfApi,
    token: str,
    collection_slug: str,
    notes_by_repo: dict[str, str],
) -> None:
    collection = api.get_collection(collection_slug, token=token)
    for item in collection.items:
        note = notes_by_repo.get(item.item_id)
        if note is None or item.note == note:
            continue
        api.update_collection_item(
            collection_slug,
            item.item_object_id,
            note=note,
            token=token,
        )


def main() -> int:
    args = parse_args()
    hf_root = args.hf_root.resolve()
    prune_generated_artifacts(hf_root)
    prune_artifact_bundle(hf_root)
    ensure_artifact_dataset_viewer_config(hf_root)
    ensure_space_runtime_files(hf_root)
    ensure_repo_card_metadata(hf_root / "space/README.md", SPACE_CARD_METADATA)
    ensure_repo_card_metadata(hf_root / "model/README.md", BASELINE_MODEL_CARD_METADATA)
    ensure_enhancement_card_links(hf_root)

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
    weights_results_repo = full_repo(args.namespace, args.weights_results_repo)
    qwen3_lora_repo = full_repo(args.namespace, args.qwen3_lora_repo)
    cosmos3_super_lora_repo = full_repo(args.namespace, args.cosmos3_super_lora_repo)

    api.create_repo(space_repo, repo_type="space", space_sdk="gradio", exist_ok=True, token=token)
    api.create_repo(artifact_repo, repo_type="dataset", exist_ok=True, token=token)
    api.create_repo(model_repo, repo_type=None, exist_ok=True, token=token)
    api.create_repo(weights_results_repo, repo_type=None, exist_ok=True, token=token)

    if not args.skip_space:
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
    if not args.skip_artifacts:
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
    if not args.skip_model:
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
    if not args.skip_weights_results:
        upload_folder(
            api,
            token,
            weights_results_repo,
            None,
            hf_root / "weights_results",
            "Publish consolidated Ropedia Xperience-10M weights/results bundle",
        )

    try:
        collection_description = (
            "Ropedia Xperience-10M dashboard, public artifacts, baselines, "
            "Qwen3-Omni v6, and Cosmos3-Super/Nano results."
        )
        collection_notes = {
            space_repo: "Interactive/static dashboard with raw public-sample previews and task-suite analysis.",
            artifact_repo: "Public-safe metrics, predictions, docs, scripts, diagrams, and verified_public result packages.",
            model_repo: "Minimal numpy weights plus aligned neural MLP checkpoints and task-head metrics.",
            weights_results_repo: "Consolidated baseline weights, Qwen3-Omni v6 LoRA, Cosmos3-Super forward-dynamics LoRA, verified results, and analysis manifest.",
            qwen3_lora_repo: "Verified v6 rank64 Qwen3-Omni LoRA adapter for the selected 128-episode diagnostic row.",
            cosmos3_super_lora_repo: "Verified Cosmos3-Super forward-dynamics LoRA adapter over camera-pose proxy targets.",
        }
        collection = api.create_collection(
            COLLECTION_TITLE,
            namespace=args.namespace,
            description=collection_description,
            private=False,
            exists_ok=True,
            token=token,
        )
        api.update_collection_metadata(
            collection.slug,
            description=collection_description,
            private=False,
            token=token,
        )
        api.add_collection_item(collection.slug, space_repo, "space", note=collection_notes[space_repo], exists_ok=True, token=token)
        api.add_collection_item(collection.slug, artifact_repo, "dataset", note=collection_notes[artifact_repo], exists_ok=True, token=token)
        api.add_collection_item(collection.slug, model_repo, "model", note=collection_notes[model_repo], exists_ok=True, token=token)
        api.add_collection_item(
            collection.slug,
            weights_results_repo,
            "model",
            note=collection_notes[weights_results_repo],
            exists_ok=True,
            token=token,
        )
        api.add_collection_item(
            collection.slug,
            qwen3_lora_repo,
            "model",
            note=collection_notes[qwen3_lora_repo],
            exists_ok=True,
            token=token,
        )
        api.add_collection_item(
            collection.slug,
            cosmos3_super_lora_repo,
            "model",
            note=collection_notes[cosmos3_super_lora_repo],
            exists_ok=True,
            token=token,
        )
        upsert_collection_item_notes(api, token, collection.slug, collection_notes)
        print(f"Collection: https://huggingface.co/collections/{collection.slug}")
    except Exception as exc:
        print(f"Collection update skipped: {exc}")

    print("Done")
    print(f"Space: https://huggingface.co/spaces/{space_repo}")
    print(f"Artifacts: https://huggingface.co/datasets/{artifact_repo}")
    print(f"Models: https://huggingface.co/{model_repo}")
    print(f"Weights/results: https://huggingface.co/{weights_results_repo}")
    print(f"Qwen3-Omni LoRA: https://huggingface.co/{qwen3_lora_repo}")
    print(f"Cosmos3-Super LoRA: https://huggingface.co/{cosmos3_super_lora_repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

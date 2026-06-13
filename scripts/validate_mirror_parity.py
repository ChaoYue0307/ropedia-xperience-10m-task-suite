#!/usr/bin/env python3
"""Validate parity between the repo and prepared Hugging Face mirrors.

This is a publisher-side check. It compares critical website data, figures, and
validator scripts across the local repo, prepared HF Space bundle, prepared HF
artifact dataset bundle, and prepared HF model bundle before upload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HF_ROOT = ROOT.parent / "hf_publish"
DEFAULT_OUTPUT = ROOT / "docs/data/mirror_parity.json"

DATA_FILES = [
    "additional_development_directions.json",
    "audio_ablation_summary.json",
    "artifact_index.json",
    "brand_assets.json",
    "evidence_contract.json",
    "evaluation_protocol.json",
    "figure_index.json",
    "foundation_model_plan.json",
    "live_publication_status.json",
    "modality_atlas.json",
    "omni_finetune_verified_result.json",
    "omni_model_comparison.json",
    "project_brief.json",
    "project_manifest.json",
    "project_packet.json",
    "project_status.json",
    "publication_audit.json",
    "public_surface_qa.json",
    "qwen3_full_parameter_gates.json",
    "qwen3_v5_v6_comparison.json",
    "quality_gates.json",
    "rendered_site_check.json",
    "reproducibility_matrix.json",
    "research_roadmap.json",
    "research_roadmap_interactive.json",
    "research_takeaways.json",
    "research_direction_extensions.json",
    "research_directions.json",
    "scope_claims_audit.json",
    "single_episode_explorer.json",
    "source_alignment_audit.json",
    "summary_metrics.json",
    "task_suite_enhancement_128.json",
    "task_surface_integrity.json",
    "task_walkthroughs.json",
    "website_integrity.json",
    "xperience10m_dataset_card_alignment.json",
]

ASSET_FILES = [
    "charts/audio_ablation_delta.svg",
    "brand/xperience10m-logo-apple-touch.png",
    "brand/xperience10m-logo-favicon-32.png",
    "brand/xperience10m-logo-favicon-64.png",
    "brand/xperience10m-logo-mark.png",
    "brand/xperience10m-logo-mark-192.png",
    "brand/xperience10m-logo-mark-512.png",
    "brand/xperience10m-logo-social-card.png",
    "task_suite_infographic.png",
    "pipeline_diagram.png",
    "task_architectures.png",
    "modalities/audio.png",
    "modalities/depth.jpg",
    "modalities/inertial.png",
    "modalities/language.png",
    "modalities/motion_capture.png",
    "modalities/pose_slam.png",
    "modalities/video.jpg",
]

SCRIPT_FILES = [
    "omni/analyze_qwen3_omni_errors.py",
    "omni/audit_cosmos3_super_training_contract.py",
    "omni/build_omni_model_comparison.py",
    "omni/build_qwen3_full_parameter_gate_summary.py",
    "omni/collect_qwen3_v4_release_artifacts.py",
    "omni/defer_cosmos3_super_after_qwen_v4.sh",
    "omni/defer_qwen3_fullparam_after_verified_qwen.sh",
    "omni/export_cosmos3_camera_pose_targets.py",
    "omni/pack_cosmos3_super_action_batch.py",
    "omni/prepare_cosmos3_super_lora_hf_package.py",
    "omni/prepare_qwen3_lora_hf_package.py",
    "omni/patch_qwen3_omni_video_features.py",
    "omni/probe_cosmos3_super_training_readiness.py",
    "omni/run_private_gpu_qwen3_v6_repro_smoke.sh",
    "omni/run_128_task_baselines.py",
    "omni/build_task_suite_enhancement_128.py",
    "omni/run_cosmos3_super_forward_dynamics_lora.sh",
    "omni/train_cosmos3_super_forward_dynamics_lora.py",
    "audio_ablation_and_raw_upgrade.py",
    "build_artifact_index.py",
    "build_brand_assets.py",
    "build_evaluation_protocol.py",
    "build_figure_index.py",
    "build_quality_gates.py",
    "build_public_surface_qa.py",
    "build_rendered_site_check.py",
    "build_interactive_research_roadmap.py",
    "build_single_episode_explorer.py",
    "build_research_takeaways.py",
    "single_episode_diagnostics.py",
    "verify_live_publication.py",
    "validate_mirror_parity.py",
    "validate_publication_package.py",
    "validate_scope_claims.py",
    "validate_source_alignment.py",
    "validate_task_surface.py",
    "validate_website_integrity.py",
    "sync_hf_publish_mirrors.py",
    "publish_hf_bundles.py",
]

WEBSITE_FILES = [
    "apple-touch-icon.png",
    "favicon.png",
    "index.html",
    "research_roadmap.html",
    "single_episode_explorer.html",
    "site.webmanifest",
]

RESULT_FILES = [
    "audio_ablation/AUDIO_ABLATION_SUMMARY.md",
    "audio_ablation/audio_ablation_metrics.csv",
    "audio_ablation/audio_ablation_summary.json",
    "audio_ablation/audio_delta_summary.csv",
    "audio_ablation/raw_logmel_fisheye_cam0_sr16000_mels64_fft512_hop160.npz",
    "single_episode_diagnostics/provenance.json",
    "single_episode_diagnostics/README.md",
    "single_episode_diagnostics/modality_ablation/ablation_metrics.csv",
    "single_episode_diagnostics/modality_ablation/ablation_summary.json",
    "single_episode_diagnostics/object_labels/object_vocab.json",
    "single_episode_diagnostics/object_labels/window_object_labels.csv",
    "single_episode_diagnostics/timeline_overlay/timeline_overlay.csv",
    "single_episode_diagnostics/alignment_stress/alignment_shift_metrics.csv",
    "single_episode_diagnostics/alignment_stress/alignment_stress_summary.json",
    "omni_finetune/verified_public/xperience10m_qwen3_omni_128ep_96train_16val_16test_valmon_20260605_eval/analysis/ERROR_ANALYSIS.md",
    "omni_finetune/verified_public/xperience10m_qwen3_omni_128ep_96train_16val_16test_valmon_20260605_eval/analysis/error_analysis_summary.json",
    "omni_finetune/verified_public/xperience10m_qwen3_omni_128ep_96train_16val_16test_valmon_20260605_eval/analysis/episode_error_analysis.csv",
    "omni_finetune/verified_public/xperience10m_qwen3_omni_128ep_96train_16val_16test_valmon_20260605_eval/analysis/action_family_error_analysis.csv",
    "omni_finetune/verified_public/xperience10m_qwen3_omni_128ep_96train_16val_16test_valmon_20260605_eval/analysis/train_seen_error_analysis.csv",
    "omni_finetune/verified_public/xperience10m_qwen3_omni_128ep_96train_16val_16test_valmon_20260605_eval/analysis/missing_modality_error_analysis.csv",
    "omni_finetune/verified_public/xperience10m_qwen3_omni_128ep_96train_16val_16test_valmon_20260605_eval/analysis/object_category_error_analysis.csv",
    "omni_finetune/multi_episode_128_task_baselines/BASELINE_ALIGNMENT_REPORT.md",
    "omni_finetune/multi_episode_128_task_baselines/summary_report.json",
    "omni_finetune/multi_episode_128_task_baselines/task_metrics.csv",
    "omni_finetune/task_suite_enhancement_128_v1_20260608/ENHANCEMENT_REPORT.md",
    "omni_finetune/task_suite_enhancement_128_v1_20260608/dense_window_scenarios.csv",
    "omni_finetune/task_suite_enhancement_128_v1_20260608/enhancement_plan.json",
    "omni_finetune/task_suite_enhancement_128_v1_20260608/experiment_backlog.json",
    "omni_finetune/task_suite_enhancement_128_v1_20260608/hierarchical_target_contract.json",
    "omni_finetune/task_suite_enhancement_128_v1_20260608/qwen_action_family_error_summary.csv",
    "omni_finetune/task_suite_enhancement_128_v1_20260608/task_bottlenecks.csv",
    "omni_finetune/OMNI_MODEL_COMPARISON.md",
    "omni_finetune/QWEN3_FULL_PARAMETER_GATES_20260609.md",
    "omni_finetune/QWEN3_V5_V6_COMPARISON_20260614.md",
    "omni_finetune/xperience10m_qwen3_omni_128ep_fullparam_smoke_preemptible_8gpu_20260609/fullparam_feasibility_summary.json",
    "omni_finetune/xperience10m_qwen3_omni_128ep_fullparam_smoke_preemptible_8gpu_20260609/progress.jsonl",
    "omni_finetune/xperience10m_qwen3_omni_128ep_fullparam_smoke_preemptible_8gpu_20260609/training_metadata.json",
    "omni_finetune/xperience10m_qwen3_omni_128ep_fullparam_smoke_preemptible_8gpu_20260609/RUN_REPORT.md",
    "omni_finetune/xperience10m_qwen3_omni_128ep_fullparam_shorttrain8_preemptible_8gpu_20260609/fullparam_shorttrain8_summary.json",
    "omni_finetune/xperience10m_qwen3_omni_128ep_fullparam_shorttrain8_preemptible_8gpu_20260609/progress.jsonl",
    "omni_finetune/xperience10m_qwen3_omni_128ep_fullparam_shorttrain8_preemptible_8gpu_20260609/launch_status.jsonl",
    "omni_finetune/xperience10m_qwen3_omni_128ep_fullparam_pilot32_preemptible_8gpu_20260609/fullparam_pilot32_summary.json",
    "omni_finetune/xperience10m_qwen3_omni_128ep_fullparam_pilot32_preemptible_8gpu_20260609/progress.jsonl",
    "omni_finetune/xperience10m_qwen3_omni_128ep_fullparam_pilot32_preemptible_8gpu_20260609/launch_status.jsonl",
    "omni_finetune/xperience10m_qwen3_omni_128ep_fullparam_pilot64_preemptible_8gpu_20260609/fullparam_pilot64_summary.json",
    "omni_finetune/xperience10m_qwen3_omni_128ep_fullparam_pilot64_preemptible_8gpu_20260609/progress.jsonl",
    "omni_finetune/xperience10m_qwen3_omni_128ep_fullparam_pilot64_preemptible_8gpu_20260609/training_metadata.json",
    "omni_finetune/xperience10m_qwen3_omni_128ep_fullparam_pilot64_preemptible_8gpu_20260609/RUN_REPORT.md",
    "omni_finetune/xperience10m_qwen3_omni_128ep_fullparam_pilot64_preemptible_8gpu_20260609/config.yaml",
    "omni_finetune/xperience10m_qwen3_omni_128ep_fullparam_pilot64_preemptible_8gpu_20260609/launch_status.jsonl",
    "omni_finetune/xperience10m_qwen3_omni_128ep_fullparam_pilot128_preemptible_8gpu_20260609/fullparam_pilot128_summary.json",
    "omni_finetune/xperience10m_qwen3_omni_128ep_fullparam_pilot128_preemptible_8gpu_20260609/progress.jsonl",
    "omni_finetune/xperience10m_qwen3_omni_128ep_fullparam_pilot128_preemptible_8gpu_20260609/launch_status.jsonl",
    "omni_finetune/xperience10m_qwen3_omni_128ep_fullparam_pilot128_after_qwen_v5_preemptible_8gpu_20260609/progress.jsonl",
    "omni_finetune/xperience10m_qwen3_omni_128ep_fullparam_pilot128_after_qwen_v5_preemptible_8gpu_20260609/config.yaml",
    "omni_finetune/xperience10m_qwen3_omni_128ep_fullparam_pilot128_after_qwen_v5_preemptible_8gpu_20260609/training_metadata.json",
    "omni_finetune/xperience10m_qwen3_omni_128ep_fullparam_pilot128_after_qwen_v5_preemptible_8gpu_20260609/RUN_REPORT.md",
    "omni_finetune/xperience10m_qwen3_omni_128ep_fullparam_pilot128_after_qwen_v5_preemptible_8gpu_20260609/launch_status.jsonl",
    "omni_finetune/HF_UPLOAD.md",
    "omni_finetune/xperience10m_cosmos3_super_training_readiness_20260607/training_readiness.json",
    "omni_finetune/xperience10m_cosmos3_super_training_readiness_20260607/RUN_REPORT.md",
    "omni_finetune/xperience10m_cosmos3_super_training_readiness_metadata_a100_20260609/training_readiness.json",
    "omni_finetune/xperience10m_cosmos3_super_training_readiness_metadata_a100_20260609/RUN_REPORT.md",
    "omni_finetune/xperience10m_cosmos3_super_training_contract_audit_camera_pose_20260608/training_contract_audit.json",
    "omni_finetune/xperience10m_cosmos3_super_training_contract_audit_camera_pose_20260608/RUN_REPORT.md",
    "omni_finetune/xperience10m_cosmos3_super_action_packer_schema_smoke_20260608/packer_summary.json",
    "omni_finetune/xperience10m_cosmos3_super_action_packer_schema_smoke_20260608/RUN_REPORT.md",
    "omni_finetune/verified_public/xperience10m_cosmos3_nano_128ep_future_window_h5_compat_adapter_eval_test_full/verified_result_summary.json",
    "omni_finetune/verified_public/xperience10m_cosmos3_nano_128ep_future_window_h5_compat_adapter_eval_test_full/PUBLIC_RESULT_SUMMARY.md",
    "omni_finetune/verified_public/xperience10m_cosmos3_nano_128ep_future_window_h5_compat_adapter_eval_test_full/eval/metrics.json",
    "omni_finetune/verified_public/xperience10m_cosmos3_nano_128ep_future_window_h5_compat_adapter_eval_test_full/eval/RUN_REPORT.md",
]

DOC_FILES = [
    "ARTIFACT_GUIDE.md",
    "OMNI_MODEL_EXTENSION_CONTRACT.md",
    "QUALITY_GATES.md",
    "EVALUATION_PROTOCOL.md",
    "FIGURE_INDEX.md",
    "FOUNDATION_MODEL_PLAN.md",
    "ADDITIONAL_DEVELOPMENT_DIRECTIONS.md",
    "PROJECT_BRIEF.md",
    "RENDERED_SITE_CHECK.md",
    "RESEARCH_ROADMAP.md",
    "PROJECT_STATUS.md",
    "REPRODUCIBILITY.md",
    "TASK_SUITE_ENHANCEMENT_128.md",
    "PUBLIC_SURFACE_QA.md",
    "RESEARCH_TAKEAWAYS.md",
    "SOURCE_ALIGNMENT_AUDIT.md",
    "XPERIENCE10M_DATASET_CARD_ALIGNMENT.md",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path, hf_root: Path) -> str:
    resolved = path.resolve()
    bases = [
        ("hf_space", hf_root / "space"),
        ("hf_artifacts", hf_root / "artifacts"),
        ("hf_model", hf_root / "model"),
        ("repo", ROOT),
        ("hf_publish", hf_root),
    ]
    for label, base in bases:
        try:
            return f"{label}:{resolved.relative_to(base.resolve()).as_posix()}"
        except ValueError:
            continue
    return path.name


def file_record(path: Path, hf_root: Path) -> dict:
    record = {
        "path": display_path(path, hf_root),
        "exists": path.exists(),
    }
    if path.exists() and path.is_file():
        record["bytes"] = path.stat().st_size
        record["sha256"] = sha256(path)
    else:
        record["bytes"] = 0
        record["sha256"] = None
    return record


def verified_public_result_files() -> list[str]:
    """Return every public-safe file from verified model packages.

    Verified packages are already sanitized by the package audit; mirroring the
    whole package prevents final model results from being silently omitted when a
    new run lands under results/omni_finetune/verified_public.
    """

    verified_root = ROOT / "results/omni_finetune/verified_public"
    if not verified_root.exists():
        return []
    files: list[str] = []
    for path in verified_root.rglob("*"):
        if not path.is_file():
            continue
        files.append(path.relative_to(ROOT / "results").as_posix())
    return sorted(files)


def parity_group(name: str, local_path: Path, mirrors: dict[str, Path], hf_root: Path) -> dict:
    local = file_record(local_path, hf_root)
    mirror_records = {surface: file_record(path, hf_root) for surface, path in mirrors.items()}
    failures = []
    if not local["exists"]:
        failures.append({"surface": "repo", "kind": "missing", "path": local["path"]})
    for surface, record in mirror_records.items():
        if not record["exists"]:
            failures.append({"surface": surface, "kind": "missing", "path": record["path"]})
            continue
        if local["exists"] and record["sha256"] != local["sha256"]:
            failures.append(
                {
                    "surface": surface,
                    "kind": "hash_mismatch",
                    "path": record["path"],
                    "expected_sha256": local["sha256"],
                    "actual_sha256": record["sha256"],
                }
            )
    return {
        "name": name,
        "status": "pass" if not failures else "fail",
        "local": local,
        "mirrors": mirror_records,
        "failures": failures,
    }


def build_report(hf_root: Path) -> dict:
    groups = []

    for filename in DATA_FILES:
        groups.append(
            parity_group(
                f"data/{filename}",
                ROOT / "docs/data" / filename,
                {
                    "hf_space": hf_root / "space/data" / filename,
                    "hf_artifacts_data": hf_root / "artifacts/data" / filename,
                    "hf_artifacts": hf_root / "artifacts/docs/data" / filename,
                    "hf_model_data": hf_root / "model/data" / filename,
                    "hf_model_docs_data": hf_root / "model/docs/data" / filename,
                    "hf_model": hf_root / "model/metrics" / filename,
                },
                hf_root,
            )
        )

    for filename in ASSET_FILES:
        groups.append(
            parity_group(
                f"assets/{filename}",
                ROOT / "docs/assets" / filename,
                {
                    "hf_space": hf_root / "space/assets" / filename,
                    "hf_artifacts_docs": hf_root / "artifacts/docs/assets" / filename,
                    "hf_artifacts_card": hf_root / "artifacts/assets" / filename,
                    "hf_model": hf_root / "model/assets" / filename,
                },
                hf_root,
            )
        )

    for filename in SCRIPT_FILES:
        groups.append(
            parity_group(
                f"scripts/{filename}",
                ROOT / "scripts" / filename,
                {
                    "hf_artifacts": hf_root / "artifacts/scripts" / filename,
                    "hf_model": hf_root / "model/scripts" / filename,
                },
                hf_root,
            )
        )

    for filename in WEBSITE_FILES:
        groups.append(
            parity_group(
                f"website/{filename}",
                ROOT / "docs" / filename,
                {
                    "hf_space": hf_root / "space" / filename,
                    "hf_artifacts_docs": hf_root / "artifacts/docs" / filename,
                },
                hf_root,
            )
        )

    result_files = sorted(set(RESULT_FILES) | set(verified_public_result_files()))
    for filename in result_files:
        groups.append(
            parity_group(
                f"results/{filename}",
                ROOT / "results" / filename,
                {
                    "hf_space": hf_root / "space/results" / filename,
                    "hf_artifacts": hf_root / "artifacts/results" / filename,
                    "hf_model": hf_root / "model/results" / filename,
                },
                hf_root,
            )
        )

    for filename in DOC_FILES:
        groups.append(
            parity_group(
                f"docs/{filename}",
                ROOT / filename,
                {
                    "hf_space": hf_root / "space" / filename,
                    "hf_artifacts": hf_root / "artifacts" / filename,
                    "hf_model": hf_root / "model" / filename,
                },
                hf_root,
            )
        )

    failures = [
        {"group": group["name"], **failure}
        for group in groups
        for failure in group["failures"]
    ]
    by_surface: dict[str, int] = {}
    for failure in failures:
        by_surface[failure["surface"]] = by_surface.get(failure["surface"], 0) + 1

    return {
        "status": "pass" if not failures else "fail",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hf_root": "hf_publish",
        "summary": {
            "group_count": len(groups),
            "failure_count": len(failures),
            "failures_by_surface": by_surface,
        },
        "checks": [
            {
                "name": "repo_hf_space_artifact_model_data_parity",
                "status": "pass"
                if not any(failure["group"].startswith("data/") for failure in failures)
                else "fail",
            },
            {
                "name": "repo_hf_visual_asset_parity",
                "status": "pass"
                if not any(failure["group"].startswith("assets/") for failure in failures)
                else "fail",
            },
            {
                "name": "repo_hf_validator_script_parity",
                "status": "pass"
                if not any(failure["group"].startswith("scripts/") for failure in failures)
                else "fail",
            },
            {
                "name": "repo_hf_website_html_parity",
                "status": "pass"
                if not any(failure["group"].startswith("website/") for failure in failures)
                else "fail",
            },
            {
                "name": "repo_hf_diagnostic_result_parity",
                "status": "pass"
                if not any(failure["group"].startswith("results/") for failure in failures)
                else "fail",
            },
            {
                "name": "repo_hf_quality_doc_parity",
                "status": "pass"
                if not any(failure["group"].startswith("docs/") for failure in failures)
                else "fail",
            },
        ],
        "groups": groups,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-root", type=Path, default=DEFAULT_HF_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    report = build_report(args.hf_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{report['status'].upper()}: wrote {args.output}")
    if report["status"] != "pass":
        for failure in report["failures"][:40]:
            print(f"- {failure['group']}: {failure['surface']} {failure['kind']} {failure['path']}")
        if len(report["failures"]) > 40:
            print(f"- ... {len(report['failures']) - 40} more failures")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

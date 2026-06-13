#!/usr/bin/env python3
"""Build a compact source-of-truth artifact index for the research project.

The index is intentionally selective. It lists the files behind the public
project readouts, not every prediction array or checkpoint in the repository.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/data/artifact_index.json"

ARTIFACTS = [
    {
        "id": "project_brief",
        "title": "Project brief",
        "path": "PROJECT_BRIEF.md",
        "kind": "project_path",
        "surface": "repo_hf",
        "shows": "Gives first-pass readers a concise project shape before the detailed artifact trail.",
    },
    {
        "id": "project_brief_json",
        "title": "Project brief JSON",
        "path": "docs/data/project_brief.json",
        "kind": "project_path",
        "surface": "website_hf",
        "shows": "Machine-readable first-reader project brief for the website and Hugging Face mirrors.",
    },
    {
        "id": "project_status",
        "title": "Project status",
        "path": "PROJECT_STATUS.md",
        "kind": "project_path",
        "surface": "repo_hf",
        "shows": "Gives a compact current-state table for first-pass readers.",
    },
    {
        "id": "project_status_json",
        "title": "Project status JSON",
        "path": "docs/data/project_status.json",
        "kind": "project_path",
        "surface": "website_hf",
        "shows": "Machine-readable copy of the current project status for website and HF mirrors.",
    },
    {
        "id": "research_roadmap",
        "title": "Research roadmap",
        "path": "RESEARCH_ROADMAP.md",
        "kind": "project_path",
        "surface": "repo_hf",
        "shows": "Defines the path from public-sample task development to multi-episode held-out evaluation and larger omni-model extensions.",
    },
    {
        "id": "research_roadmap_json",
        "title": "Research roadmap JSON",
        "path": "docs/data/research_roadmap.json",
        "kind": "project_path",
        "surface": "website_hf",
        "shows": "Machine-readable research roadmap for the website and Hugging Face mirrors.",
    },
    {
        "id": "foundation_model_plan",
        "title": "Foundation model plan",
        "path": "FOUNDATION_MODEL_PLAN.md",
        "kind": "project_path",
        "surface": "repo_hf",
        "shows": "Defines the post-data-gate backbone choices: Qwen3-Omni first, Cosmos 3 for world modeling, and VLA/policy models after action-target conversion.",
    },
    {
        "id": "foundation_model_plan_json",
        "title": "Foundation model plan JSON",
        "path": "docs/data/foundation_model_plan.json",
        "kind": "project_path",
        "surface": "website_hf",
        "shows": "Machine-readable foundation-model selection matrix with source links, entry conditions, and evaluation additions.",
    },
    {
        "id": "omni_model_extension_contract",
        "title": "Omni model extension contract",
        "path": "OMNI_MODEL_EXTENSION_CONTRACT.md",
        "kind": "scaleup_contract",
        "surface": "repo_hf",
        "shows": "Defines the shared manifest, episode split, held-out evaluation, packaging, and public-safety rules for Qwen3-Omni, Cosmos-style, and VLA/policy model branches.",
    },
    {
        "id": "omni_backbone_registry_configs",
        "title": "Omni backbone registry configs",
        "path": "configs/omni_backbones",
        "kind": "scaleup_contract",
        "surface": "repo_hf",
        "shows": "Stores the implemented Qwen3-Omni LoRA contract and planned Cosmos-style world-model and VLA/policy branch contracts.",
    },
    {
        "id": "omni_backbone_registry_validator",
        "title": "Omni backbone registry validator",
        "path": "scripts/omni/backbone_registry.py",
        "kind": "scaleup_contract",
        "surface": "repo_hf",
        "shows": "Validates backbone ids, split defaults, leakage guards, required metrics, required files, and forbidden public package categories.",
    },
    {
        "id": "omni_model_neutral_window_index_exporter",
        "title": "Model-neutral window index exporter",
        "path": "scripts/omni/export_model_neutral_window_index.py",
        "kind": "scaleup_contract",
        "surface": "repo_hf",
        "shows": "Converts Qwen JSONL records into a model-neutral window index with Qwen, Cosmos-style, and policy/VLA adapter views.",
    },
    {
        "id": "omni_backbone_scaffolder",
        "title": "Omni backbone scaffolder",
        "path": "scripts/omni/scaffold_omni_backbone.py",
        "kind": "scaleup_contract",
        "surface": "repo_hf",
        "shows": "Creates a validated planned-backbone config from an existing contract template so new model families inherit the shared rules.",
    },
    {
        "id": "omni_backbone_packaging_smoke",
        "title": "Omni backbone packaging smoke test",
        "path": "scripts/omni/smoke_test_backbone_packaging.py",
        "kind": "scaleup_contract",
        "surface": "repo_hf",
        "shows": "Builds synthetic verified packages for every configured backbone and audits them against the public-safe package contract.",
    },
    {
        "id": "qwen3_omni_error_analysis_script",
        "title": "Qwen3-Omni held-out error-analysis script",
        "path": "scripts/omni/analyze_qwen3_omni_errors.py",
        "kind": "scaleup_contract",
        "surface": "repo_hf",
        "shows": "Computes public-safe held-out error-analysis tables by episode, action family, train-seen status, required-modality state, and object category.",
    },
    {
        "id": "multi_episode_128_baseline_script",
        "title": "128-episode aligned baseline runner",
        "path": "scripts/omni/run_128_task_baselines.py",
        "kind": "scaleup_status",
        "surface": "repo_hf",
        "shows": "Runs simple metadata and neural MLP baselines on the same selected 96/16/16 episode split used by the Qwen3-Omni diagnostic pilot.",
    },
    {
        "id": "task_suite_enhancement_128",
        "title": "128-episode task-suite enhancement pack",
        "path": "TASK_SUITE_ENHANCEMENT_128.md",
        "kind": "scaleup_status",
        "surface": "repo_hf",
        "shows": "Records the no-new-episode dense-window, hierarchical-target, bottleneck, and experiment-backlog plan for pushing the current 128-episode suite harder without overwriting prior results.",
    },
    {
        "id": "task_suite_enhancement_128_json",
        "title": "128-episode task-suite enhancement JSON",
        "path": "docs/data/task_suite_enhancement_128.json",
        "kind": "scaleup_status",
        "surface": "website_hf",
        "shows": "Machine-readable enhancement pack for the website and Hugging Face mirrors.",
    },
    {
        "id": "task_suite_enhancement_128_result",
        "title": "128-episode task-suite enhancement result package",
        "path": "results/omni_finetune/task_suite_enhancement_128_v1_20260608/enhancement_plan.json",
        "kind": "scaleup_status",
        "surface": "repo_hf",
        "shows": "Versioned result directory with dense-window estimates, hierarchical target contract, task bottlenecks, Qwen action-family error summary, and experiment cards.",
    },
    {
        "id": "task_suite_enhancement_128_builder",
        "title": "128-episode task-suite enhancement builder",
        "path": "scripts/omni/build_task_suite_enhancement_128.py",
        "kind": "scaleup_status",
        "surface": "repo_hf",
        "shows": "Regenerates the enhancement pack from committed 128-episode windows, baseline summaries, verified Qwen predictions, and Cosmos reference metrics.",
    },
    {
        "id": "qwen3_full_parameter_gates",
        "title": "Qwen3-Omni full-parameter feasibility gates",
        "path": "results/omni_finetune/QWEN3_FULL_PARAMETER_GATES_20260609.md",
        "kind": "scaleup_status",
        "surface": "repo_hf",
        "shows": "Summarizes the 2026-06-09 full-parameter FSDP feasibility gates: 1/8/32/64-step guarded runs passed, the 128-step opportunistic pilot was preempted for Qwen v5 handoff, and no full checkpoints or weights are published.",
    },
    {
        "id": "qwen3_full_parameter_gates_json",
        "title": "Qwen3-Omni full-parameter feasibility gates JSON",
        "path": "docs/data/qwen3_full_parameter_gates.json",
        "kind": "scaleup_status",
        "surface": "website_hf",
        "shows": "Machine-readable summary of full-parameter feasibility evidence and publication policy for website and Hugging Face mirrors.",
    },
    {
        "id": "qwen3_v5_v6_comparison",
        "title": "Qwen3-Omni v5/v6 comparison",
        "path": "results/omni_finetune/QWEN3_V5_V6_COMPARISON_20260614.md",
        "kind": "scaleup_status",
        "surface": "repo_hf",
        "shows": "Reader-facing comparison of the verified Qwen3 v5 release row and the latest verified v6 row, including metric deltas and release-tag policy.",
    },
    {
        "id": "qwen3_v5_v6_comparison_json",
        "title": "Qwen3-Omni v5/v6 comparison JSON",
        "path": "docs/data/qwen3_v5_v6_comparison.json",
        "kind": "scaleup_status",
        "surface": "website_hf",
        "shows": "Machine-readable v5/v6 metric deltas and publication recommendation for website and Hugging Face mirrors.",
    },
    {
        "id": "qwen3_full_parameter_gates_builder",
        "title": "Qwen3-Omni full-parameter gate summary builder",
        "path": "scripts/omni/build_qwen3_full_parameter_gate_summary.py",
        "kind": "scaleup_status",
        "surface": "repo_hf",
        "shows": "Regenerates the full-parameter feasibility-gate Markdown and JSON summaries from the run-local evidence files.",
    },
    {
        "id": "qwen3_full_parameter_post_verified_deferrer",
        "title": "Qwen3-Omni post-verified full-parameter deferrer",
        "path": "scripts/omni/defer_qwen3_fullparam_after_verified_qwen.sh",
        "kind": "scaleup_status",
        "surface": "repo_hf",
        "shows": "Waits for a verified Qwen held-out package, then launches a bounded 128-step full-parameter feasibility pilot on the same multiscale v5 dataset with no checkpoints or weights saved.",
    },
    {
        "id": "qwen3_lora_hf_package_builder",
        "title": "Qwen3 LoRA HF package builder",
        "path": "scripts/omni/prepare_qwen3_lora_hf_package.py",
        "kind": "publication_workflow",
        "surface": "repo_hf",
        "shows": "Builds the upload-ready Hugging Face adapter folder from a verified Qwen3 LoRA result summary and adapter directory.",
    },
    {
        "id": "additional_development_directions",
        "title": "Additional development directions",
        "path": "ADDITIONAL_DEVELOPMENT_DIRECTIONS.md",
        "kind": "project_path",
        "surface": "repo_hf",
        "shows": "Records concrete non-backbone Xperience-10M development tracks: taxonomy, benchmark protocol, representation learning, skill graphs, affordances, 3D/4D memory, QA, and policy transfer.",
    },
    {
        "id": "additional_development_directions_json",
        "title": "Additional development directions JSON",
        "path": "docs/data/additional_development_directions.json",
        "kind": "project_path",
        "surface": "website_hf",
        "shows": "Machine-readable additional development directions for the website and Hugging Face mirrors.",
    },
    {
        "id": "xperience_embodied_foundation_pretraining",
        "title": "Xperience Embodied Foundation Model pretraining goal",
        "path": "XPERIENCE_EMBODIED_FOUNDATION_MODEL_PRETRAINING.md",
        "kind": "project_path",
        "surface": "repo_hf",
        "shows": "Describes the future full-corpus Xperience-native pretraining goal, target modules, objectives, staged scale-up, hardware ranges, and evaluation protocol.",
    },
    {
        "id": "evidence_contract",
        "title": "Evidence contract",
        "path": "EVIDENCE_CONTRACT.md",
        "kind": "project_scope",
        "surface": "repo",
        "shows": "Defines the implemented scope, setup-stage items, and multi-episode prerequisites.",
    },
    {
        "id": "project_packet",
        "title": "Project packet",
        "path": "docs/data/project_packet.json",
        "kind": "project_path",
        "surface": "website_hf",
        "shows": "Gives a short project path with scope status and public surfaces.",
    },
    {
        "id": "artifact_guide",
        "title": "Artifact guide",
        "path": "ARTIFACT_GUIDE.md",
        "kind": "project_path",
        "surface": "repo_hf",
        "shows": "Gives the human-readable map from project scope to data, tasks, platform mirrors, and scale-up status.",
    },
    {
        "id": "official_dataset_card_alignment",
        "title": "Official Xperience-10M dataset-card alignment",
        "path": "XPERIENCE10M_DATASET_CARD_ALIGNMENT.md",
        "kind": "source_alignment",
        "surface": "repo_hf",
        "shows": "Aligns public dataset wording with the official gated Xperience-10M card, public sample card, HF API metadata, and current project coverage.",
    },
    {
        "id": "official_dataset_card_alignment_json",
        "title": "Official Xperience-10M dataset-card alignment JSON",
        "path": "docs/data/xperience10m_dataset_card_alignment.json",
        "kind": "source_alignment",
        "surface": "website_hf",
        "shows": "Machine-readable upstream dataset-card, sample-card, and HF API alignment facts for website and HF mirrors.",
    },
    {
        "id": "source_alignment",
        "title": "Source alignment",
        "path": "SOURCE_ALIGNMENT_AUDIT.md",
        "kind": "source_alignment",
        "surface": "repo_hf",
        "shows": "Summarizes the pass/fail check for full-dataset facts, sample-card facts, API-listing notes, and project coverage.",
    },
    {
        "id": "source_alignment_json",
        "title": "Source alignment JSON",
        "path": "docs/data/source_alignment_audit.json",
        "kind": "source_alignment",
        "surface": "website_hf",
        "shows": "Machine-readable source-alignment pass/fail check for repo, website, and HF surfaces.",
    },
    {
        "id": "source_alignment_validator",
        "title": "Source alignment validator",
        "path": "scripts/validate_source_alignment.py",
        "kind": "source_alignment",
        "surface": "repo_hf",
        "shows": "Regenerates the source-alignment report from committed facts and public card text.",
    },
    {
        "id": "hf_publisher",
        "title": "Hugging Face publisher",
        "path": "scripts/publish_hf_bundles.py",
        "kind": "publication_workflow",
        "surface": "repo_hf",
        "shows": "Publishes prepared Space, artifact dataset, and model bundles, including an explicit model-binary upload batch.",
    },
    {
        "id": "github_package_dockerfile",
        "title": "GitHub package Dockerfile",
        "path": "Dockerfile",
        "kind": "publication_workflow",
        "surface": "repo",
        "shows": "Builds the static-dashboard container package for GitHub Container Registry.",
    },
    {
        "id": "github_package_workflow",
        "title": "GitHub package workflow",
        "path": ".github/workflows/publish-ghcr.yml",
        "kind": "publication_workflow",
        "surface": "repo",
        "shows": "Publishes the static-dashboard image to GitHub Container Registry on main or manual dispatch.",
    },
    {
        "id": "evaluation_protocol",
        "title": "Evaluation protocol",
        "path": "EVALUATION_PROTOCOL.md",
        "kind": "evaluation_protocol",
        "surface": "repo_hf",
        "shows": "Defines the window unit, chronological split, task metrics, leakage controls, and current limitations.",
    },
    {
        "id": "evaluation_protocol_json",
        "title": "Evaluation protocol JSON",
        "path": "docs/data/evaluation_protocol.json",
        "kind": "evaluation_protocol",
        "surface": "website_hf",
        "shows": "Machine-readable protocol generated from committed task metrics for website and HF mirrors.",
    },
    {
        "id": "evaluation_protocol_builder",
        "title": "Evaluation protocol builder",
        "path": "scripts/build_evaluation_protocol.py",
        "kind": "evaluation_protocol",
        "surface": "repo_hf",
        "shows": "Regenerates the protocol from committed summary metrics and task artifacts.",
    },
    {
        "id": "research_takeaways",
        "title": "Research takeaways",
        "path": "RESEARCH_TAKEAWAYS.md",
        "kind": "result_interpretation",
        "surface": "repo_hf",
        "shows": "Summarizes the main research lessons from committed metrics and identifies which experiments need held-out episodes.",
    },
    {
        "id": "research_takeaways_json",
        "title": "Research takeaways JSON",
        "path": "docs/data/research_takeaways.json",
        "kind": "result_interpretation",
        "surface": "website_hf",
        "shows": "Machine-readable result interpretation for the website, HF cards, and mirror checks.",
    },
    {
        "id": "research_takeaways_builder",
        "title": "Research takeaways builder",
        "path": "scripts/build_research_takeaways.py",
        "kind": "result_interpretation",
        "surface": "repo_hf",
        "shows": "Regenerates the research takeaways from committed summary metrics and task result artifacts.",
    },
    {
        "id": "audio_ablation_script",
        "title": "Audio contribution script",
        "path": "scripts/audio_ablation_and_raw_upgrade.py",
        "kind": "result_interpretation",
        "surface": "repo_hf",
        "shows": "Measures audio contribution variants across all 12 task contracts.",
    },
    {
        "id": "audio_ablation_summary",
        "title": "Audio ablation summary",
        "path": "results/audio_ablation/audio_ablation_summary.json",
        "kind": "metrics_source",
        "surface": "repo_hf",
        "shows": "Stores per-task audio deltas for all current features, no-audio, audio-only, alternate-audio-only, replacement, and all-plus-alternate variants.",
    },
    {
        "id": "audio_ablation_summary_md",
        "title": "Audio ablation summary report",
        "path": "results/audio_ablation/AUDIO_ABLATION_SUMMARY.md",
        "kind": "result_interpretation",
        "surface": "repo_hf",
        "shows": "Human-readable table showing the measured audio contribution and alternate-representation delta for every task.",
    },
    {
        "id": "audio_ablation_website_json",
        "title": "Audio ablation website JSON",
        "path": "docs/data/audio_ablation_summary.json",
        "kind": "website_data",
        "surface": "website_hf",
        "shows": "Machine-readable audio ablation summary mirrored into the static website and Hugging Face bundles.",
    },
    {
        "id": "audio_ablation_delta_chart",
        "title": "Audio ablation delta chart",
        "path": "docs/assets/charts/audio_ablation_delta.svg",
        "kind": "visual_evidence",
        "surface": "website_hf",
        "shows": "Bar chart of measured current-audio primary-metric deltas across the 12 tasks.",
    },
    {
        "id": "figure_index",
        "title": "Figure index",
        "path": "FIGURE_INDEX.md",
        "kind": "visual_evidence",
        "surface": "repo_hf",
        "shows": "Catalogs public figures, charts, modality thumbnails, dimensions, hashes, roles, and source scripts.",
    },
    {
        "id": "figure_index_json",
        "title": "Figure index JSON",
        "path": "docs/data/figure_index.json",
        "kind": "visual_evidence",
        "surface": "website_hf",
        "shows": "Machine-readable visual asset index for website and Hugging Face mirrors.",
    },
    {
        "id": "figure_index_builder",
        "title": "Figure index builder",
        "path": "scripts/build_figure_index.py",
        "kind": "visual_evidence",
        "surface": "repo_hf",
        "shows": "Regenerates visual-asset hashes, dimensions, and source-script provenance.",
    },
    {
        "id": "brand_assets_json",
        "title": "Brand assets manifest",
        "path": "docs/data/brand_assets.json",
        "kind": "visual_evidence",
        "surface": "website_hf",
        "shows": "Machine-readable manifest for the generated logo system, favicon, social card, dimensions, hashes, and usage roles.",
    },
    {
        "id": "brand_logo_social_card",
        "title": "Brand logo social card",
        "path": "docs/assets/brand/xperience10m-logo-social-card.png",
        "kind": "visual_evidence",
        "surface": "website_hf",
        "shows": "Provides the project logo card used in README, Hugging Face cards, and social previews.",
    },
    {
        "id": "brand_asset_builder",
        "title": "Brand asset builder",
        "path": "scripts/build_brand_assets.py",
        "kind": "visual_evidence",
        "surface": "repo_hf",
        "shows": "Regenerates logo derivatives, favicon variants, app icons, and the Open Graph social card from the generated logo mark.",
    },
    {
        "id": "quality_gates",
        "title": "Release checks",
        "path": "QUALITY_GATES.md",
        "kind": "quality_gate",
        "surface": "repo_hf",
        "shows": "Lists the automated and post-publish checks used to keep the release current.",
    },
    {
        "id": "quality_gate_manifest",
        "title": "Release-check manifest",
        "path": "docs/data/quality_gates.json",
        "kind": "quality_gate",
        "surface": "website_hf",
        "shows": "Machine-readable release-check summary for validators, mirrors, and public project surfaces.",
    },
    {
        "id": "public_surface_qa",
        "title": "Public project surface",
        "path": "PUBLIC_SURFACE_QA.md",
        "kind": "quality_gate",
        "surface": "repo_hf",
        "shows": "Keeps the repo, website, and Hugging Face cards aligned as one cohesive research project surface.",
    },
    {
        "id": "public_surface_qa_json",
        "title": "Public project surface JSON",
        "path": "docs/data/public_surface_qa.json",
        "kind": "quality_gate",
        "surface": "website_hf",
        "volatile": True,
        "shows": "Machine-readable report for SEO/social metadata, accessible tab semantics, public links, project links, and clear project presentation.",
    },
    {
        "id": "public_surface_qa_builder",
        "title": "Public project surface builder",
        "path": "scripts/build_public_surface_qa.py",
        "kind": "quality_gate",
        "surface": "repo_hf",
        "shows": "Regenerates the public presentation report before release.",
    },
    {
        "id": "task_surface_integrity",
        "title": "Task-surface integrity report",
        "path": "docs/data/task_surface_integrity.json",
        "kind": "quality_gate",
        "surface": "website_hf",
        "volatile": True,
        "shows": "Confirms the public 12-task cards use human-readable research names, representative modality thumbnails, and the interactive walkthrough/player JSON contract.",
    },
    {
        "id": "rendered_site_check",
        "title": "Rendered website check",
        "path": "RENDERED_SITE_CHECK.md",
        "kind": "quality_gate",
        "surface": "repo_hf",
        "volatile": True,
        "shows": "Records the latest browser-level load, tab, walkthrough deep-link, control-click, and console-health check.",
    },
    {
        "id": "rendered_site_check_json",
        "title": "Rendered website check JSON",
        "path": "docs/data/rendered_site_check.json",
        "kind": "quality_gate",
        "surface": "website_hf",
        "volatile": True,
        "shows": "Machine-readable browser-level website check for the public static site.",
    },
    {
        "id": "rendered_site_check_builder",
        "title": "Rendered website check builder",
        "path": "scripts/build_rendered_site_check.py",
        "kind": "quality_gate",
        "surface": "repo_hf",
        "shows": "Builds the rendered website check from browser observations.",
    },
    {
        "id": "task_surface_validator",
        "title": "Task-surface integrity validator",
        "path": "scripts/validate_task_surface.py",
        "kind": "quality_gate",
        "surface": "repo_hf",
        "shows": "Regenerates the task-surface integrity report and fails if task cards expose raw artifact ids or lose the interactive player wiring.",
    },
    {
        "id": "live_publication_status",
        "title": "Live publication status",
        "path": "docs/data/live_publication_status.json",
        "kind": "quality_gate",
        "surface": "website_hf",
        "volatile": True,
        "shows": "Records the last live GitHub/HF URL verification after upload.",
    },
    {
        "id": "live_publication_verifier",
        "title": "Live publication verifier",
        "path": "scripts/verify_live_publication.py",
        "kind": "quality_gate",
        "surface": "repo",
        "shows": "Fetches the published GitHub/HF URLs and compares live hashes and public-card markers against the release assets.",
    },
    {
        "id": "reproducibility_contract",
        "title": "Reproducibility contract",
        "path": "REPRODUCIBILITY.md",
        "kind": "reproducibility",
        "surface": "repo_hf",
        "shows": "Defines public reproduction commands, expected outputs, and non-reproducible scale-up boundaries.",
    },
    {
        "id": "reproducibility_matrix",
        "title": "Reproducibility matrix",
        "path": "docs/data/reproducibility_matrix.json",
        "kind": "reproducibility",
        "surface": "website_hf",
        "shows": "Machine-readable reproduction steps with expected artifacts and public boundaries.",
    },
    {
        "id": "artifact_index_builder",
        "title": "Artifact index builder",
        "path": "scripts/build_artifact_index.py",
        "kind": "project_path",
        "surface": "repo_hf",
        "shows": "Generates the selective artifact catalog from local files.",
    },
    {
        "id": "publication_audit",
        "title": "Public bundle contents",
        "path": "docs/data/publication_audit.json",
        "kind": "publication_package_check",
        "surface": "website_hf",
        "volatile": True,
        "shows": "Confirms public bundles exclude raw data, caches, heavy archives, and credential text.",
    },
    {
        "id": "scale_up_status_check",
        "title": "Multi-episode pilot status",
        "path": "docs/data/scope_claims_audit.json",
        "kind": "scale_up_status",
        "surface": "website_hf",
        "volatile": True,
        "shows": "Separates setup paths from completed held-out-episode results.",
    },
    {
        "id": "mirror_parity",
        "title": "Prepared mirror parity report",
        "path": "docs/data/mirror_parity.json",
        "kind": "mirror_parity",
        "surface": "website_hf",
        "volatile": True,
        "shows": "Confirms prepared GitHub/HF Space/artifact/model mirrors share the same critical data, figure, website HTML, and validator files.",
    },
    {
        "id": "website_integrity",
        "title": "Website integrity report",
        "path": "docs/data/website_integrity.json",
        "kind": "integrity_report",
        "surface": "website_hf",
        "volatile": True,
        "shows": "Confirms local website links, anchors, JSON data files, and referenced images resolve.",
    },
    {
        "id": "project_manifest",
        "title": "Project manifest",
        "path": "docs/data/project_manifest.json",
        "kind": "metadata",
        "surface": "website_hf",
        "shows": "Lists public URLs, upstream sources, and machine-readable project metadata.",
    },
    {
        "id": "task_summary",
        "title": "12-task summary report",
        "path": "results/episode_task_suite/summary_report.json",
        "kind": "metrics_source",
        "surface": "repo_hf",
        "shows": "Stores the task definitions, splits, feature dimension, and minimal/neural metrics.",
    },
    {
        "id": "website_metrics_bundle",
        "title": "Website metrics bundle",
        "path": "docs/data/summary_metrics.json",
        "kind": "website_data",
        "surface": "website_hf",
        "shows": "Mirrors task metrics for the static dashboard.",
    },
    {
        "id": "feature_manifest",
        "title": "Feature manifest",
        "path": "results/episode_task_suite/feature_manifest.json",
        "kind": "data_contract",
        "surface": "repo_hf",
        "shows": "Maps the current window vector back to source feature blocks.",
    },
    {
        "id": "available_modalities",
        "title": "Available modalities",
        "path": "results/episode_task_suite/available_modalities.json",
        "kind": "data_contract",
        "surface": "repo_hf",
        "shows": "Documents which sample modalities entered the current extracted feature contract.",
    },
    {
        "id": "windows_table",
        "title": "Aligned windows table",
        "path": "results/episode_task_suite/windows.csv",
        "kind": "data_contract",
        "surface": "repo_hf",
        "shows": "Lists the 1,161 aligned windows and their frame/action/subtask labels.",
    },
    {
        "id": "neural_mlp_directory",
        "title": "Neural MLP task-head results",
        "path": "results/episode_task_suite/neural_mlp",
        "kind": "result_directory",
        "surface": "repo_hf_model",
        "shows": "Stores matching PyTorch MLP results for the 12 task contracts.",
    },
    {
        "id": "research_direction_taxonomy",
        "title": "Research direction taxonomy",
        "path": "results/episode_task_suite/research_directions/research_direction_taxonomy.json",
        "kind": "taxonomy",
        "surface": "repo_hf",
        "shows": "Maps the 12 tasks to the four Ropedia research directions as direct/proxy/diagnostic.",
    },
    {
        "id": "research_direction_extensions",
        "title": "Research direction extension probes",
        "path": "results/episode_task_suite/research_direction_extensions/research_direction_extension_results.json",
        "kind": "metrics_source",
        "surface": "repo_hf",
        "shows": "Stores one coded extension probe per research direction with minimal and neural metrics.",
    },
    {
        "id": "task_walkthroughs",
        "title": "Task walkthroughs",
        "path": "results/episode_task_suite/task_walkthroughs/TASK_WALKTHROUGHS.md",
        "kind": "onboarding_doc",
        "surface": "repo_hf",
        "shows": "Explains every task with case study, input, process modules, output, and limitation.",
    },
    {
        "id": "task_suite_infographic",
        "title": "12-task suite infographic",
        "path": "docs/assets/task_suite_infographic.png",
        "kind": "generated_figure",
        "surface": "website_hf",
        "shows": "Presents the task suite and sample modality thumbnails with metrics generated from committed files.",
    },
    {
        "id": "modality_atlas",
        "title": "Responsive modality atlas",
        "path": "docs/data/modality_atlas.json",
        "kind": "website_data",
        "surface": "website_hf",
        "shows": "Documents the seven public-sample modality cards and their derived thumbnail assets.",
    },
    {
        "id": "modality_thumbnails",
        "title": "Standalone modality thumbnails",
        "path": "docs/assets/modalities",
        "kind": "generated_figure_assets",
        "surface": "website_hf",
        "shows": "Stores small derived thumbnails for readable website modality cards without raw data redistribution.",
    },
    {
        "id": "pipeline_figure",
        "title": "Pipeline figure",
        "path": "docs/assets/pipeline_diagram.png",
        "kind": "generated_figure",
        "surface": "website_hf",
        "shows": "Shows the raw-episode to artifact pipeline with verified labels.",
    },
    {
        "id": "architecture_figure",
        "title": "Architecture figure",
        "path": "docs/assets/task_architectures.png",
        "kind": "generated_figure",
        "surface": "website_hf",
        "shows": "Shows the shared feature pipeline and minimal/neural head families.",
    },
    {
        "id": "qwen_data_access_status",
        "title": "Qwen3-Omni data access status",
        "path": "results/omni_finetune/DATA_ACCESS_STATUS.md",
        "kind": "scaleup_status",
        "surface": "repo_hf",
        "shows": "Summarizes the data-readiness checks required before a held-out Qwen3-Omni pilot can report metrics.",
    },
    {
        "id": "qwen3_lora_hf_upload_note",
        "title": "Qwen3 LoRA HF upload note",
        "path": "results/omni_finetune/HF_UPLOAD.md",
        "kind": "publication_workflow",
        "surface": "repo_hf",
        "shows": "Documents the final 128-episode LoRA adapter upload path, target model repo, package builder, and forbidden files.",
    },
    {
        "id": "multi_episode_access_status",
        "title": "Multi-episode access status",
        "path": "results/omni_finetune/MULTI_EPISODE_ACCESS_STATUS.md",
        "kind": "scaleup_status",
        "surface": "repo_hf",
        "shows": "Documents the public multi-episode access status and 32-episode pilot selection.",
    },
    {
        "id": "qwen3_omni_error_analysis_report",
        "title": "Qwen3-Omni held-out error-analysis report",
        "path": "results/omni_finetune/verified_public/xperience10m_qwen3_omni_128ep_96train_16val_16test_valmon_20260605_eval/analysis/ERROR_ANALYSIS.md",
        "kind": "scaleup_status",
        "surface": "repo_hf",
        "shows": "Summarizes the earlier validation-aware Qwen3-Omni held-out failures by episode, action family, train-seen status, required-modality state, and object category.",
    },
    {
        "id": "qwen3_omni_error_analysis_json",
        "title": "Qwen3-Omni held-out error-analysis JSON",
        "path": "results/omni_finetune/verified_public/xperience10m_qwen3_omni_128ep_96train_16val_16test_valmon_20260605_eval/analysis/error_analysis_summary.json",
        "kind": "scaleup_status",
        "surface": "repo_hf",
        "shows": "Machine-readable Qwen3-Omni held-out error analysis with grouped metrics and sanitized failure examples.",
    },
    {
        "id": "multi_episode_128_baseline_report",
        "title": "128-episode aligned baseline report",
        "path": "results/omni_finetune/multi_episode_128_task_baselines/BASELINE_ALIGNMENT_REPORT.md",
        "kind": "scaleup_status",
        "surface": "repo_hf",
        "shows": "Summarizes same-split simple and neural metadata baselines for the 12 task ids, with unsupported markers for tasks that need missing raw 128 feature blocks.",
    },
    {
        "id": "multi_episode_128_baseline_summary",
        "title": "128-episode aligned baseline summary",
        "path": "results/omni_finetune/multi_episode_128_task_baselines/summary_report.json",
        "kind": "metrics_source",
        "surface": "repo_hf",
        "shows": "Machine-readable 96/16/16 split counts, run configuration, per-task simple metrics, neural metrics, and raw-feature unsupported statuses.",
    },
    {
        "id": "omni_model_comparison_report",
        "title": "Omni model comparison report",
        "path": "results/omni_finetune/OMNI_MODEL_COMPARISON.md",
        "kind": "scaleup_status",
        "surface": "repo_hf",
        "shows": "Reader-facing comparison of the single-episode task suite, 128-episode aligned baselines, Qwen3-Omni packages, and Cosmos3 future-window branch.",
    },
    {
        "id": "omni_model_comparison_json",
        "title": "Omni model comparison JSON",
        "path": "docs/data/omni_model_comparison.json",
        "kind": "metrics_source",
        "surface": "repo_hf",
        "shows": "Machine-readable comparison of the current result versions, per-task aligned baselines, verified Qwen3 packages, and Cosmos3 package.",
    },
    {
        "id": "cosmos3_nano_verified_summary",
        "title": "Cosmos3-Nano verified package summary",
        "path": "results/omni_finetune/verified_public/xperience10m_cosmos3_nano_128ep_future_window_h5_compat_adapter_eval_test_full/verified_result_summary.json",
        "kind": "metrics_source",
        "surface": "repo_hf",
        "shows": "Machine-readable verified public summary for the Cosmos3-Nano future-window compatibility package.",
    },
    {
        "id": "cosmos3_nano_run_report",
        "title": "Cosmos3-Nano future-window run report",
        "path": "results/omni_finetune/verified_public/xperience10m_cosmos3_nano_128ep_future_window_h5_compat_adapter_eval_test_full/eval/RUN_REPORT.md",
        "kind": "scaleup_status",
        "surface": "repo_hf",
        "shows": "Reader-facing held-out metrics and interpretation for the Cosmos3-Nano future-window compatibility branch.",
    },
    {
        "id": "citation",
        "title": "Citation metadata",
        "path": "CITATION.cff",
        "kind": "citation",
        "surface": "repo_hf",
        "shows": "Makes the project externally citable.",
    },
    {
        "id": "license",
        "title": "License and data terms",
        "path": "LICENSE",
        "kind": "license",
        "surface": "repo_hf",
        "shows": "Separates MIT-scoped code from original Xperience-10M data terms.",
    },
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_stats(path: Path) -> dict:
    files = [item for item in path.rglob("*") if item.is_file()]
    return {
        "file_count": len(files),
        "bytes": sum(item.stat().st_size for item in files),
    }


def verified_public_package_artifacts() -> list[dict]:
    verified_root = ROOT / "results/omni_finetune/verified_public"
    if not verified_root.exists():
        return []

    artifacts: list[dict] = []
    for summary_path in sorted(verified_root.glob("*/verified_result_summary.json")):
        package_dir = summary_path.parent
        slug = package_dir.name
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        title = payload.get("backbone_display_name") or payload.get("eval_run_id") or slug
        backbone = payload.get("backbone", "unknown_backbone")
        status = payload.get("status", "unknown")
        eval_run_id = payload.get("eval_run_id", slug)
        artifacts.append(
            {
                "id": f"verified_public_package_{slug}",
                "title": f"Verified public package: {title}",
                "path": package_dir.relative_to(ROOT).as_posix(),
                "kind": "verified_public_package",
                "surface": "repo_hf",
                "shows": (
                    f"Public-safe verified package for {eval_run_id} "
                    f"({backbone}, status={status})."
                ),
            }
        )
        artifacts.append(
            {
                "id": f"verified_public_summary_{slug}",
                "title": f"Verified summary: {title}",
                "path": summary_path.relative_to(ROOT).as_posix(),
                "kind": "metrics_source",
                "surface": "repo_hf",
                "shows": f"Machine-readable verified summary for {eval_run_id}.",
            }
        )
        for relative_path, kind, label in [
            ("PUBLIC_RESULT_SUMMARY.md", "scaleup_status", "public result summary"),
            ("eval/RUN_REPORT.md", "scaleup_status", "run report"),
            ("eval/metrics.json", "metrics_source", "metrics JSON"),
            ("package_audit.json", "publication_audit", "package audit"),
        ]:
            path = package_dir / relative_path
            if not path.exists():
                continue
            safe_label = label.replace(" ", "_")
            artifacts.append(
                {
                    "id": f"verified_public_{safe_label}_{slug}",
                    "title": f"Verified {label}: {title}",
                    "path": path.relative_to(ROOT).as_posix(),
                    "kind": kind,
                    "surface": "repo_hf",
                    "shows": f"{label.capitalize()} for {eval_run_id}.",
                }
            )
    return artifacts


def artifact_entry(item: dict) -> dict:
    path = ROOT / item["path"]
    entry = {
        **item,
        "exists": path.exists(),
    }
    if path.is_file():
        entry["bytes"] = path.stat().st_size
        if item.get("volatile"):
            entry["hash_policy"] = "existence_and_size_only"
        else:
            entry["sha256"] = sha256(path)
    elif path.is_dir():
        entry.update(directory_stats(path))
    else:
        entry.update({"bytes": 0})
    return entry


def main() -> int:
    artifacts = [dict(item) for item in ARTIFACTS]
    artifacts.extend(verified_public_package_artifacts())
    summary_path = ROOT / "results/episode_task_suite/summary_report.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        feature_dim = int(summary.get("feature_dim", 0))
        for item in artifacts:
            if item["id"] == "feature_manifest" and feature_dim:
                item["shows"] = f"Maps the {feature_dim:,}-dimensional window vector back to source feature blocks."
    entries = [artifact_entry(item) for item in artifacts]
    missing = [entry["path"] for entry in entries if not entry["exists"]]
    by_kind: dict[str, int] = {}
    for entry in entries:
        by_kind[entry["kind"]] = by_kind.get(entry["kind"], 0) + 1

    report = {
        "title": "Ropedia Xperience-10M Task Suite Artifact Index",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "pass" if not missing else "fail",
        "artifact_count": len(entries),
        "missing": missing,
        "by_kind": by_kind,
        "artifacts": entries,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{report['status'].upper()}: wrote {OUTPUT}")
    if missing:
        for path in missing:
            print(f"- missing: {path}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

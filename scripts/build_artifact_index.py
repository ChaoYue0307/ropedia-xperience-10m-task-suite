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
QWEN3_FUTURE_TASK_PROBE_RUN_ID = "xperience10m_qwen3_omni_v6_future_task_probes_a100_20260616T143608Z"

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
        "id": "three_foundation_pipelines",
        "title": "Three foundation pipeline tracks",
        "path": "THREE_FOUNDATION_PIPELINES.md",
        "kind": "project_path",
        "surface": "repo_hf",
        "shows": "Frames spatial intelligence, human-video world modeling, and vision-language-action as three pipeline tracks with explicit inputs, outputs, maturity, and next evidence gates.",
    },
    {
        "id": "three_foundation_pipelines_json",
        "title": "Three foundation pipeline tracks JSON",
        "path": "docs/data/three_foundation_pipelines.json",
        "kind": "project_path",
        "surface": "website_hf",
        "shows": "Machine-readable pipeline-track contract for the website and Hugging Face mirrors.",
    },
    {
        "id": "spatial_intelligence_task_training_diagram",
        "title": "Spatial intelligence task-training diagram",
        "path": "docs/assets/foundation-pipelines/spatial-intelligence-pipeline.png",
        "kind": "visual_asset",
        "surface": "website_hf",
        "shows": "Inputs-to-tasks-to-training-to-evaluation diagram for the spatial intelligence model training pipeline.",
    },
    {
        "id": "human_video_world_model_task_training_diagram",
        "title": "Human-video world model task-training diagram",
        "path": "docs/assets/foundation-pipelines/human-video-world-model-pipeline.png",
        "kind": "visual_asset",
        "surface": "website_hf",
        "shows": "Observed-window-to-future-targets-to-training-to-evaluation diagram for the human-video world-model training pipeline.",
    },
    {
        "id": "vision_language_action_task_training_diagram",
        "title": "Vision-language-action task-training diagram",
        "path": "docs/assets/foundation-pipelines/vision-language-action-pipeline.png",
        "kind": "visual_asset",
        "surface": "website_hf",
        "shows": "Observation-and-language-to-action-targets-to-VLA-training-to-evaluation diagram for the vision-language-action training pipeline.",
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
        "id": "qwen3_private_gpu_repro_smoke",
        "title": "Qwen3 private staged-GPU reproduction smoke",
        "path": "scripts/omni/run_private_gpu_qwen3_v6_repro_smoke.sh",
        "kind": "reproducibility",
        "surface": "repo_hf",
        "shows": "Runs the owner-side Qwen3-Omni v6 one-sample reproduction smoke from a private staged model, adapter, JSONL, and exported media cache.",
    },
    {
        "id": "qwen3_video_feature_compat_patch",
        "title": "Qwen3 video-feature compatibility patch checker",
        "path": "scripts/omni/patch_qwen3_omni_video_features.py",
        "kind": "reproducibility",
        "surface": "repo_hf",
        "shows": "Checks and narrowly repairs the installed Qwen3-Omni video-feature branch so private staged-GPU reproduction uses the verified source-compatible behavior.",
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
        "id": "task_suite_20",
        "title": "Unified 20-task suite",
        "path": "TASK_SUITE_20.md",
        "kind": "evaluation_protocol",
        "surface": "repo_hf",
        "shows": "Reader-facing table for the single unified public-sample task suite: tasks 1-12 plus tasks 13-20 under the same window, split, feature, and baseline contract.",
    },
    {
        "id": "task_suite_20_json",
        "title": "Unified 20-task suite JSON",
        "path": "docs/data/task_suite_20.json",
        "kind": "website_data",
        "surface": "website_hf",
        "shows": "Machine-readable unified 20-task index for the website, Hugging Face mirrors, and live verification.",
    },
    {
        "id": "task_suite_20_builder",
        "title": "Unified 20-task suite builder",
        "path": "scripts/build_unified_task_suite.py",
        "kind": "evaluation_protocol",
        "surface": "repo_hf",
        "shows": "Regenerates the unified 20-task JSON and Markdown from the original 12-task metrics plus the tasks 13-20 result bundle.",
    },
    {
        "id": "unified_task_model_radar_json",
        "title": "Unified 20-task model radar JSON",
        "path": "docs/data/unified_task_model_radar.json",
        "kind": "website_data",
        "surface": "website_hf",
        "shows": "Stores normalized 20-axis radar values, raw task metrics, Qwen3/Cosmos overlay mappings, branch-card caveats, and explicit scoreless status records.",
    },
    {
        "id": "single_episode_task_model_radar_json",
        "title": "Single-episode 20-task model radar JSON",
        "path": "docs/data/single_episode_task_model_radar.json",
        "kind": "website_data",
        "surface": "website_hf",
        "shows": "Machine-readable split radar for the one-episode Minimal and Neural MLP baselines, both scored on all 20 task contracts.",
    },
    {
        "id": "episode128_task_model_radar_json",
        "title": "128-episode 20-task model radar JSON",
        "path": "docs/data/episode128_task_model_radar.json",
        "kind": "website_data",
        "surface": "website_hf",
        "shows": "Machine-readable split radar for selected 128-episode metadata/raw baselines and verified Qwen3/Cosmos branches, preserving explicit scoreless cells.",
    },
    {
        "id": "task_method_20_result_matrix_json",
        "title": "Task-method 20-result matrix JSON",
        "path": "docs/data/task_method_20_result_matrix.json",
        "kind": "website_data",
        "surface": "website_hf",
        "shows": "Machine-readable 9-method by 20-task matrix where every method has 20 records and scoreless cells carry unsupported/not-evaluated reasons.",
    },
    {
        "id": "task_method_20_result_matrix",
        "title": "Task-method 20-result matrix",
        "path": "TASK_METHOD_20_RESULT_MATRIX.md",
        "kind": "evaluation_protocol",
        "surface": "repo_hf",
        "shows": "Reader-facing table that separates 20 records per method from numeric scored axes, documented raw128 proxy scores, unsupported metadata targets, and model targets not evaluated in verified packages.",
    },
    {
        "id": "task_method_20_gap_audit_json",
        "title": "Task-method 20-result gap audit JSON",
        "path": "docs/data/task_method_20_gap_audit.json",
        "kind": "website_data",
        "surface": "website_hf",
        "shows": "Machine-readable 180-record gap ledger with numeric scores, scoreless cells, explicit status reasons, and next evidence needed before new scores can be published.",
    },
    {
        "id": "task_method_20_gap_audit",
        "title": "Task-method 20-result gap audit",
        "path": "TASK_METHOD_20_GAP_AUDIT.md",
        "kind": "evaluation_protocol",
        "surface": "repo_hf",
        "shows": "Reader-facing ledger that lists every scoreless method-task cell and the concrete target or model-output evidence required before it can become numeric.",
    },
    {
        "id": "unified_task_model_radar_chart",
        "title": "Unified 20-task model radar",
        "path": "docs/assets/charts/unified_task_model_radar.svg",
        "kind": "generated_figure",
        "surface": "website_hf",
        "shows": "Compares minimal and neural MLP baselines across all 20 tasks, with Qwen3/Cosmos task-aligned model overlays.",
    },
    {
        "id": "single_episode_task_model_radar_chart",
        "title": "Single-episode 20-task model radar",
        "path": "docs/assets/charts/single_episode_task_model_radar.svg",
        "kind": "generated_figure",
        "surface": "website_hf",
        "shows": "Separates the one-episode Minimal and Neural MLP 20/20 scored baselines into a clean two-polygon radar.",
    },
    {
        "id": "episode128_task_model_radar_chart",
        "title": "128-episode 20-task model radar",
        "path": "docs/assets/charts/episode128_task_model_radar.svg",
        "kind": "generated_figure",
        "surface": "website_hf",
        "shows": "Separates the selected 128-episode methods: raw-feature simple/NN as complete 20/20 scored polygons and metadata/Qwen/Cosmos as task-aligned overlays.",
    },
    {
        "id": "unified_task_model_radar_builder",
        "title": "Unified 20-task model radar builder",
        "path": "scripts/build_unified_task_model_radar.py",
        "kind": "visualization_builder",
        "surface": "repo_hf",
        "shows": "Regenerates the direction-aware radar chart and machine-readable metric overlay JSON.",
    },
    {
        "id": "task_method_20_gap_audit_builder",
        "title": "Task-method gap-audit builder",
        "path": "scripts/build_task_method_20_gap_audit.py",
        "kind": "publication_workflow",
        "surface": "repo_hf",
        "shows": "Regenerates the public gap audit from the 9-method by 20-task matrix without inventing scores for unsupported or unevaluated cells.",
    },
    {
        "id": "all_task_model_scoring_waiter",
        "title": "All-task model scoring guarded waiter",
        "path": "scripts/omni/launch_all_task_model_scoring_when_free.sh",
        "kind": "scaleup_status",
        "surface": "repo_hf",
        "shows": "Launches a user-provided all-task model scoring command only after enough private GPU capacity is idle, writing status logs under results/omni_finetune/deferred_launchers.",
    },
    {
        "id": "model_output_probe_readiness",
        "title": "Model-output probe readiness",
        "path": "results/omni_finetune/model_output_probe_readiness/model_output_probe_readiness.json",
        "kind": "scaleup_status",
        "surface": "repo_hf",
        "shows": "Checks whether Qwen3/Cosmos branches have train, validation, and test prediction files before extending model overlays to all 20 task contracts.",
    },
    {
        "id": "model_output_probe_script",
        "title": "Model-output probe readiness script",
        "path": "scripts/omni/score_model_output_probes.py",
        "kind": "scaleup_status",
        "surface": "repo_hf",
        "shows": "Audits model-output split availability and writes a readiness report without assigning new numeric task scores.",
    },
    {
        "id": "existing_model_output_task_probe",
        "title": "Existing model-output task probe package",
        "path": "results/omni_finetune/model_output_task_probes_20260616/summary.json",
        "kind": "model_result",
        "surface": "repo_hf",
        "shows": "Scores task 16 action-object relation only where verified held-out prediction JSON already contains action and object-set fields.",
    },
    {
        "id": "existing_model_output_task_probe_script",
        "title": "Existing model-output task probe scorer",
        "path": "scripts/omni/score_existing_model_output_task_probes.py",
        "kind": "scaleup_status",
        "surface": "repo_hf",
        "shows": "Derives task-specific scores from committed verified model outputs without running new inference or backfilling absent targets.",
    },
    {
        "id": "a100_128_metadata_task_baselines",
        "title": "128-episode metadata task baselines",
        "path": "results/omni_finetune/a100_128_metadata_task_baselines_20260616_v2/summary_report.json",
        "kind": "model_result",
        "surface": "repo_hf",
        "shows": "Rerun of JSONL metadata/text simple and neural baselines over the selected 128-episode multiscale dataset; supports radar overlays on JSONL-supported task axes.",
    },
    {
        "id": "a100_128_raw20_task_baselines",
        "title": "128-episode raw-feature 20-task baselines",
        "path": "results/omni_finetune/a100_128_raw20_task_baselines_complete20_proxy_20260616T091500Z/run_summary_all.json",
        "kind": "model_result",
        "surface": "repo_hf",
        "shows": "Rerun of simple and neural baselines over 34,269 windows and staged 4430-dimensional sensor NPZ features; covers 20 of 20 task axes, with interaction text and camera-view sync marked as compact-proxy completions because the 128 export lacks raw interaction strings and paired video-view embeddings.",
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
        "shows": "Measures audio contribution variants across the original task contracts.",
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
        "shows": "Bar chart of measured current-audio primary-metric deltas across the original tasks.",
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
        "id": "raw_sample_files_manifest",
        "title": "Raw public sample file manifest",
        "path": "docs/data/raw_sample_files.json",
        "kind": "dataset_context",
        "surface": "website_hf",
        "shows": "Lists the official public sample HDF5, MP4, and RRD files, derived browser-preview clips, playback/download URLs, file sizes, browser behavior, and HDF5 group organization.",
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
        "id": "public_reader_map",
        "title": "Public reader map",
        "path": "PUBLIC_READER_MAP.md",
        "kind": "project_path",
        "surface": "repo_hf",
        "shows": "Provides the first-pass navigation layer for GitHub, GitHub Pages, Hugging Face mirrors, model-branch repos, evidence layers, and claim boundaries.",
    },
    {
        "id": "public_reader_map_json",
        "title": "Public reader map JSON",
        "path": "docs/data/public_reader_map.json",
        "kind": "project_path",
        "surface": "website_hf",
        "shows": "Machine-readable public reader map used by the website and Hugging Face mirrors to keep entry points and surface responsibilities explicit.",
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
        "shows": "Confirms the public original-task cards use human-readable research names, representative modality thumbnails, and the interactive walkthrough/player JSON contract.",
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
        "title": "Original task summary report",
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
        "shows": "Stores matching PyTorch MLP results for the original task contracts.",
    },
    {
        "id": "research_direction_taxonomy",
        "title": "Research direction taxonomy",
        "path": "results/episode_task_suite/research_directions/research_direction_taxonomy.json",
        "kind": "taxonomy",
        "surface": "repo_hf",
        "shows": "Maps the original tasks to the four Ropedia research directions as direct/proxy/diagnostic.",
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
        "id": "tier2_task_suite",
        "title": "Tasks 13-20 result bundle",
        "path": "results/episode_task_suite/tier2_task_suite/tier2_task_suite_results.json",
        "kind": "metrics_source",
        "surface": "repo_hf",
        "shows": "Stores the historical result bundle for unified tasks 13-20 with minimal and neural baselines aligned to the same 20-task window/split setup.",
    },
    {
        "id": "tier2_task_suite_json",
        "title": "Tasks 13-20 result JSON",
        "path": "docs/data/tier2_task_suite.json",
        "kind": "website_data",
        "surface": "website_hf",
        "shows": "Machine-readable tasks 13-20 definitions, setup alignment, metrics, and public source paths; the file name is historical.",
    },
    {
        "id": "tier2_task_suite_chart",
        "title": "Tasks 13-20 chart",
        "path": "docs/assets/charts/tier2_task_suite.svg",
        "kind": "generated_figure",
        "surface": "website_hf",
        "shows": "Visual summary of the eight additional task baseline metrics in the unified 20-task suite.",
    },
    {
        "id": "tier2_task_suite_builder",
        "title": "Tasks 13-20 builder",
        "path": "scripts/tier2_task_suite.py",
        "kind": "evaluation_protocol",
        "surface": "repo_hf",
        "shows": "Regenerates tasks 13-20 from shared windows plus the local public-sample annotation HDF5; the script name is historical.",
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
        "title": "Original task-suite infographic",
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
        "shows": "Summarizes same-split simple and neural metadata baselines for the 12 original task ids, with unsupported markers for tasks that need missing raw 128 feature blocks.",
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


def qwen3_future_task_probe_artifacts() -> list[dict]:
    run_dir = ROOT / "results/omni_finetune" / QWEN3_FUTURE_TASK_PROBE_RUN_ID
    if not run_dir.exists():
        return []

    artifacts: list[dict] = [
        {
            "id": "qwen3_future_task_probe_package",
            "title": "Qwen3 v6 future-task probe package",
            "path": run_dir.relative_to(ROOT).as_posix(),
            "kind": "model_result",
            "surface": "repo_hf",
            "shows": (
                "Two-shard Qwen3-Omni v6 inference probe for tasks 13, 14, "
                "and 17, with public-safe metrics, predictions, progress logs, "
                "and merge report."
            ),
        }
    ]
    for relative_path, kind, label in [
        ("summary.json", "metrics_source", "merged summary"),
        ("collection_validation.json", "publication_audit", "collection validation"),
        ("RUN_REPORT.md", "scaleup_status", "run report"),
        ("long_horizon_next_action/metrics.json", "metrics_source", "task 13 metrics"),
        ("next_subtask_forecast/metrics.json", "metrics_source", "task 14 metrics"),
        ("object_set_forecast/metrics.json", "metrics_source", "task 17 metrics"),
    ]:
        path = run_dir / relative_path
        if path.exists():
            artifacts.append(
                {
                    "id": f"qwen3_future_task_probe_{relative_path.replace('/', '_').replace('.', '_')}",
                    "title": f"Qwen3 future-task probe {label}",
                    "path": path.relative_to(ROOT).as_posix(),
                    "kind": kind,
                    "surface": "repo_hf",
                    "shows": f"Public-safe {label} for {QWEN3_FUTURE_TASK_PROBE_RUN_ID}.",
                }
            )

    launcher_log = (
        ROOT
        / "results/omni_finetune/deferred_launchers"
        / f"{QWEN3_FUTURE_TASK_PROBE_RUN_ID}.launcher.log"
    )
    if launcher_log.exists():
        artifacts.append(
            {
                "id": "qwen3_future_task_probe_launcher_log",
                "title": "Qwen3 future-task probe launcher log",
                "path": launcher_log.relative_to(ROOT).as_posix(),
                "kind": "scaleup_status",
                "surface": "repo_hf",
                "shows": "Launch and merge log for the two-shard future-task probe.",
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
    artifacts.extend(qwen3_future_task_probe_artifacts())
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

# Public Reader Map

This project is intentionally evidence-heavy: the GitHub repo, website, and
Hugging Face mirrors each expose a different view of the same research package.
Use this map to choose the right entry point without losing the full artifact
trail.

## Fast Paths

| Reader goal | Start here | Then inspect |
| --- | --- | --- |
| Understand the project in one pass | `PROJECT_BRIEF.md` | `PROJECT_STATUS.md`, `RESEARCH_TAKEAWAYS.md` |
| See the visual public dashboard | GitHub Pages or the HF Space | `docs/index.html`, `docs/data/project_packet.json` |
| Understand the data unit | `results/episode_task_suite/windows.csv` | `results/episode_task_suite/feature_manifest.json`, `docs/data/raw_sample_files.json` |
| Inspect the 20-task benchmark | `TASK_SUITE_20.md` | `docs/data/task_suite_20.json`, `EVALUATION_PROTOCOL.md` |
| Compare current results | `RESEARCH_TAKEAWAYS.md` | `docs/data/task_method_20_result_matrix.json`, `docs/data/unified_task_model_radar.json` |
| Compare 1-episode and 128-episode methods | Homepage radar section | `docs/data/single_episode_task_model_radar.json`, `docs/data/episode128_task_model_radar.json` |
| Find all derived artifacts | `ARTIFACT_GUIDE.md` | HF artifact dataset, `docs/data/artifact_index.json` |
| Reproduce or extend the work | `REPRODUCIBILITY.md` | `QUALITY_GATES.md`, `scripts/`, `results/` |
| Understand foundation-model directions | `THREE_FOUNDATION_PIPELINES.md` | `FOUNDATION_MODEL_PLAN.md`, `docs/data/three_foundation_pipelines.json` |
| Check public-release health | `PUBLIC_SURFACE_QA.md` | `docs/data/live_publication_status.json`, `docs/data/mirror_parity.json` |

## Public Surfaces

| Surface | Responsibility | Best use |
| --- | --- | --- |
| GitHub repo | Source of truth for docs, scripts, generated data, validators, and commit history | Auditing implementation and citing exact files |
| GitHub Pages dashboard | Reader-facing visual overview of the dataset sample, tasks, methods, results, directions, and resources | Understanding the project quickly |
| Hugging Face Space | Hub-hosted copy of the dashboard and static app assets | Sharing the visual dashboard from HF |
| HF artifact dataset | Public-safe derived artifacts, reports, metrics, website JSON, and sanitized model result packages | Downloading evidence bundles |
| HF baseline model repo | Baseline weights, metrics, figures, and mirrored task artifacts | Reusing compact baseline outputs |
| Qwen3/Cosmos model repos | Adapter-specific public weights or package cards when a branch is verified and publishable | Inspecting model-branch artifacts |

## Evidence Layers

1. Dataset/source boundary: upstream Xperience-10M links, public sample scope,
   raw-data exclusion, and derived-file policy.
2. Data contract: 20-frame windows, feature blocks, modality availability,
   split policy, and leakage controls.
3. Task suite: 20 named tasks with inputs, outputs, metrics, baseline
   artifacts, and walkthroughs.
4. Results: minimal heads, neural heads, 128-episode aligned baselines,
   Qwen3-Omni diagnostics, Cosmos diagnostics, radar views, and explicit gaps.
5. Foundation directions: spatial intelligence, human-video world modeling, and
   vision-language-action training pipelines.
6. Public-release checks: website integrity, source alignment, mirror parity,
   publication package scan, and live URL/hash verification.

## Claim Boundaries

| Claim type | Public evidence | Boundary |
| --- | --- | --- |
| Single public-sample task behavior | `results/episode_task_suite/`, `docs/data/task_suite_20.json` | Describes one public sample episode, not the full dataset distribution |
| 128-episode baseline comparison | `results/omni_finetune/*128*`, `docs/data/omni_model_comparison.json` | Uses selected held-out episodes and derived public-safe summaries |
| Foundation-model branch quality | Verified Qwen3/Cosmos result packages and model cards | Numeric task scores appear only when a task-specific eval/probe exists |
| Reproducibility | `REPRODUCIBILITY.md`, `QUALITY_GATES.md`, release validators | Raw gated Xperience-10M files and full foundation weights are not redistributed |

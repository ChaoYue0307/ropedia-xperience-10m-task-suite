# Research Roadmap

This roadmap connects the current public-sample task lab to the next
multi-episode Xperience-10M experiments and the later foundation-model branches.
Each stage lists the entry condition, the deliverables, and the evidence that
should exist before the stage is treated as complete.

## Roadmap Summary

| Stage | Status | Entry condition | Research deliverables | Completion evidence |
| --- | --- | --- | --- | --- |
| Public-Sample Task Lab | Implemented | One public Xperience-10M sample episode is available. | 1,161 aligned windows, 12 task contracts, minimal heads, neural MLP heads, modality atlas, task walkthroughs, and derived figures. | `PROJECT_STATUS.md`, `EVALUATION_PROTOCOL.md`, `RESEARCH_TAKEAWAYS.md`, `docs/data/summary_metrics.json`, `results/episode_task_suite/summary_report.json` |
| Multi-Episode Data Staging | Active | Gated dataset access and enough storage for selected episodes. | 32 valid episodes, episode manifest, missing-view manifest, held-out episode split, and source-discovery report. | `results/omni_finetune/DATA_ACCESS_STATUS.md`, `results/omni_finetune/MULTI_EPISODE_ACCESS_STATUS.md`, `results/omni_finetune/source_discovery.json` |
| 32-Episode Qwen3-Omni LoRA Pilot | Next | At least 32 valid episodes staged locally with no train/test episode leakage. | Dataset JSONL/media manifests, LoRA adapter checkpoint, progress logs, held-out predictions, metrics, confusion matrices, and run report. | `dataset_manifest.json`, `training_metadata.json`, `progress.jsonl`, `metrics.json`, `predictions.jsonl`, `RUN_REPORT.md` |
| Foundation-Model Selection Matrix | Next | 32-episode data gate is satisfied, or a 3-8 episode dry run is staged for preprocessing checks. | Backbone registry, Cosmos 3 world-model branch plan, Qwen3-Omni baseline plan, OpenVLA/openpi/GR00T policy candidates, and model-specific evaluation additions. | `FOUNDATION_MODEL_PLAN.md`, `docs/data/foundation_model_plan.json`, `research_roadmap_interactive.json` |
| 64-128 Episode Robustness Run | Planned | The 32-episode pilot trains and evaluates cleanly. | Split-by-session metrics, modality ablations, calibration/object/language error analysis, and sensitivity to missing views. | Held-out metrics by session, task, and modality; ablation tables; qualitative error analysis. |
| Cosmos 3 and Policy-Model Extensions | Planned | Enough multi-episode data, compute budget, and model-specific action/world-state targets. | Cosmos 3 future-window or action-conditioned world-model probes, OpenVLA/openpi/GR00T action-policy baselines, modality-conditioning audits, affordance tasks, and synthetic-data usefulness tests. | Task-specific held-out evaluations, qualitative inspection, and updated model cards. |

## Current Decision Point

The useful next decision is data scale plus backbone fit: keep the public-sample
task suite as the development harness, stage enough official Xperience-10M
episodes to run the 32-episode held-out pilot, then choose larger model branches
by task fit. Qwen3-Omni remains the first trainable multimodal LoRA target.
Cosmos 3 becomes the first world-model/action-generation branch. OpenVLA,
openpi, GR00T, Octo, and SmolVLA-style models become policy/action branches only
after the action target is explicit. The public sample is already enough for
task design, feature contracts, walkthroughs, and baseline comparisons. It is
not enough to measure general embodied-AI model quality.

## Stage Details

### 1. Public-Sample Task Lab

This stage turns one synchronized egocentric episode into a clean research
surface. It defines what one model input is, what each task predicts, how the
split is constructed, and how minimal and neural heads are compared.

Evidence to inspect:

- `results/episode_task_suite/windows.csv`
- `results/episode_task_suite/feature_manifest.json`
- `results/episode_task_suite/summary_report.json`
- `results/episode_task_suite/neural_mlp/`
- `docs/data/task_walkthroughs.json`

### 2. Multi-Episode Data Staging

This stage expands the same data contract to official gated episodes. The key
research requirement is episode-level separation: training and test examples
must come from different episodes, not different windows inside the same
episode.

Evidence to inspect:

- `results/omni_finetune/DATA_ACCESS_STATUS.md`
- `results/omni_finetune/MULTI_EPISODE_ACCESS_STATUS.md`
- `scripts/omni/discover_xperience10m_sources.py`
- `results/omni_finetune/source_discovery.json`

### 3. 32-Episode Qwen3-Omni LoRA Pilot

This stage uses Qwen3-Omni as the multimodal backbone and trains lightweight
LoRA adapters. The first target is a complete held-out-episode training and
evaluation loop with inspectable manifests, predictions, and metrics.

Expected outputs:

- `dataset_manifest.json`
- `episode_manifest.json`
- `training_metadata.json`
- `progress.jsonl`
- `metrics.json`
- `predictions.jsonl`
- `predictions.csv`
- `confusion_matrix.csv`
- `RUN_REPORT.md`

### 4. 64-128 Episode Robustness Run

This stage asks whether the 32-episode conclusions survive more sessions,
different objects, missing views, and stronger modality ablations. It should
report performance by task, session, modality, and failure type.

### 5. Foundation-Model Selection Matrix

This stage records which foundation model is suitable for which Xperience-10M
objective. The current decision is:

- Qwen3-Omni first for multimodal instruction, structured JSON prediction, and
  LoRA over video/audio/language plus sensor-bridge features.
- Cosmos 3 next for world modeling, action-conditioned future prediction, and
  synthetic-data experiments.
- OpenVLA, openpi, GR00T, Octo, and SmolVLA-style policies after action-space
  conversion and retargeting are auditable.
- Gemini Robotics only as an external reasoning/reference surface unless local
  trainable access becomes available.

Evidence to inspect:

- `FOUNDATION_MODEL_PLAN.md`
- `docs/data/foundation_model_plan.json`
- `docs/data/research_roadmap_interactive.json`

### 6. Cosmos 3 and Policy-Model Extensions

This stage moves beyond lightweight heads and LoRA pilots into richer multimodal
objectives: audio-visible alignment, future-window prediction,
action-conditioned world modeling, synthetic-data usefulness tests, policy-style
next action, contact, object relevance, and affordance reasoning.

## Public Artifacts That Should Move Together

When a roadmap stage advances, update these public surfaces together:

- `README.md`
- `PROJECT_STATUS.md`
- `RESEARCH_TAKEAWAYS.md`
- `EVALUATION_PROTOCOL.md`
- `ARTIFACT_GUIDE.md`
- `docs/index.html`
- `docs/data/research_roadmap.json`
- Hugging Face Space, artifact dataset, and model cards

# Research Roadmap

This roadmap connects the current public-sample task lab to the next
multi-episode Xperience-10M experiments and the later foundation-model branches.
Each stage lists the entry condition, the deliverables, and the evidence that
should exist before the stage is treated as complete.

## Roadmap Summary

| Stage | Status | Entry condition | Research deliverables | Completion evidence |
| --- | --- | --- | --- | --- |
| Public-Sample Task Lab | Implemented | One public Xperience-10M sample episode is available. | 1,161 aligned windows, 12 task contracts, minimal heads, neural MLP heads, modality atlas, task walkthroughs, and derived figures. | `PROJECT_STATUS.md`, `EVALUATION_PROTOCOL.md`, `RESEARCH_TAKEAWAYS.md`, `docs/data/summary_metrics.json`, `results/episode_task_suite/summary_report.json` |
| Multi-Episode Data Preparation | Implemented for first selected pilot | Gated dataset availability and enough storage for selected episodes. | 128 selected episodes, episode manifest, missing-view manifest, held-out episode split, and source-discovery report. | `results/omni_finetune/DATA_ACCESS_STATUS.md`, `results/omni_finetune/MULTI_EPISODE_ACCESS_STATUS.md`, `results/omni_finetune/xperience10m_128_episode_selection.json` |
| Qwen3-Omni LoRA Final Diagnostic Result | Verified baseline | Selected episodes prepared locally with no train/test episode leakage. | Dataset JSONL/media manifests, LoRA adapter checkpoint, progress logs, validation monitoring, held-out predictions, metrics, confusion matrices, run report, and public LoRA adapter repo. | `docs/data/omni_finetune_verified_result.json`, `results/omni_finetune/verified_public/`, `metrics.json`, `predictions.jsonl`, `RUN_REPORT.md`, `https://huggingface.co/cy0307/ropedia-qwen3-omni-lora-128ep` |
| 128-Episode Same-Split Simple/NN Baselines | Verified companion result | Derived Qwen JSONL export for the selected 96/16/16 split. | Same 12 task ids, simple metadata/text baselines, neural MLP baselines where JSON labels support them, and explicit unsupported markers for tasks that still require raw 128 feature blocks. | `results/omni_finetune/multi_episode_128_task_baselines/BASELINE_ALIGNMENT_REPORT.md`, `summary_report.json`, `scripts/omni/run_128_task_baselines.py` |
| Action/Subtask Error-Analysis Pass | Active next step | The final diagnostic package meets strict JSON validity but has weak action/subtask held-out quality. | Same 96/16/16 split, action/subtask confusion analysis, unseen-label analysis, object/action family breakdowns, and comparison to the final verified Qwen baseline. | Updated error-analysis tables, held-out metrics by failure type, and verified public package. |
| Foundation-Model Selection Matrix | Current | The selected pilot episodes are prepared, or a 3-8 episode dry run is available for preprocessing checks. | Backbone registry, Cosmos 3 world-model branch plan, Qwen3-Omni baseline plan, OpenVLA/openpi/GR00T policy candidates, and model-specific evaluation additions. | `FOUNDATION_MODEL_PLAN.md`, `docs/data/foundation_model_plan.json`, `research_roadmap_interactive.json` |
| 64-128 Episode Robustness Run | Planned | The final selected-episode Qwen diagnostic run trains and evaluates cleanly. | Split-by-session metrics, modality ablations, calibration/object/language error analysis, and sensitivity to missing views. | Held-out metrics by session, task, and modality; ablation tables; qualitative error analysis. |
| Cosmos 3 and Policy-Model Extensions | Planned | Enough multi-episode data, compute budget, and model-specific action/world-state targets. | Cosmos 3 future-window or action-conditioned world-model probes, OpenVLA/openpi/GR00T action-policy baselines, modality-conditioning checks, affordance tasks, and synthetic-data usefulness tests. | Task-specific held-out evaluations, qualitative inspection, and updated model cards. |
| Xperience Embodied Foundation Model Pretraining | Future | Full-corpus access, PB-scale storage path, multi-node compute, and positive scaling evidence from smaller runs. | Xperience-native temporal multimodal model, full-corpus manifests, pretraining shards, scaling curves, held-out evaluations, and model card. | Pretraining metadata, checkpoint inventory, held-out metrics, scaling report, and data-boundary report. |

## Current Decision Point

The useful next decision is model-quality improvement plus backbone fit: keep
the public-sample task suite as the development harness, use the verified
Qwen3-Omni final diagnostic result as the first cross-episode
baseline, then improve action/subtask quality before claiming
model quality. The earlier simple and neural baseline framing is now aligned to
the same 96/16/16 split through metadata/text baselines for JSON-supported task
ids; raw-feature-only tasks remain marked as needing the 128-run sensor feature
blocks.
Qwen3-Omni remains the first trainable multimodal LoRA target. Cosmos 3 becomes
the first world-model/action-generation branch. OpenVLA, openpi, GR00T, Octo,
and SmolVLA-style models become policy/action branches only after the action
target is explicit. A from-scratch Xperience Embodied Foundation Model is the
long-term native-pretraining goal, not the immediate experiment. The public
sample is already enough for task design, feature contracts, walkthroughs, and
baseline comparisons. The first multi-episode pilot is enough to verify the
end-to-end training loop, but its weak metrics are not final model quality.

## Additional Concrete Development Directions

The project can also grow through smaller, high-leverage directions that do not
depend on immediately training a larger foundation model:

| Direction | First artifact | Research value |
| --- | --- | --- |
| Episode taxonomy and data engine | Episode atlas, category tags, balance report, and split builder. | Makes episode selection representative and measurable. |
| Standardized benchmark protocol | Fixed splits, task cards, metric scripts, and leakage checks. | Makes future model comparisons fair. |
| Multimodal representation learning | Contrastive and masked-window objectives over synchronized modalities. | Learns reusable encoders before expensive large-model training. |
| Skill and procedure graph mining | Steps, transitions, preconditions, effects, and temporal skill graphs. | Connects perception to planning and long-horizon reasoning. |
| Human-object interaction and affordance modeling | Contact, reachable-object, tool-use, and next-affordance tasks. | Models what the scene makes possible, not only the current label. |
| 3D/4D scene and object memory | Persistent scene/object maps from depth, pose, multiview video, and objects. | Supports object permanence and spatial reasoning. |
| Data quality and synchronization diagnostics | Per-episode QA for drift, missing streams, calibration, and corrupted files. | Prevents silent failures in large multimodal training. |
| Policy, retargeting, and simulation transfer | Action-token conversion and robot-compatible imitation examples. | Bridges human egocentric experience to robot policy work. |

The concise public source is
`ADDITIONAL_DEVELOPMENT_DIRECTIONS.md`; the website/Hugging Face data copy is
`docs/data/additional_development_directions.json`.

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

### 2. Multi-Episode Data Preparation

This stage expands the same data contract to official gated episodes. The key
research requirement is episode-level separation: training and test examples
must come from different episodes, not different windows inside the same
episode. The first selected 96/16/16 split has been used for a verified
Qwen3-Omni diagnostic pilot.

Evidence to inspect:

- `results/omni_finetune/DATA_ACCESS_STATUS.md`
- `results/omni_finetune/MULTI_EPISODE_ACCESS_STATUS.md`
- `scripts/omni/discover_xperience10m_sources.py`
- `results/omni_finetune/source_discovery.json`
- `results/omni_finetune/multi_episode_128_task_baselines/BASELINE_ALIGNMENT_REPORT.md`

### 3. Qwen3-Omni LoRA Pilot

This stage uses Qwen3-Omni as the multimodal backbone and trains lightweight
LoRA adapters. The final held-out diagnostic package now exists. It proves the
export, training, evaluation, validation, public-safe packaging, and adapter
publication loop. The current strict-label v3 evaluation reaches 100.00% JSON
validity, 97.32% transition accuracy, and 72.10% contact accuracy, but action
macro-F1 is 0.0022 and subtask accuracy is 0.0022. Treat it as a baseline and
error-analysis starting point, not as a strong action/subtask model.

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

This stage asks whether the pilot conclusions survive more sessions,
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
  conversion and retargeting are traceable.
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

Current Cosmos3-Super status: a camera-pose proxy action target export augments
all 3,808 selected 128-episode windows, passes the contract audit, and now has
a verified 8-GPU FSDP forward-dynamics LoRA run. The full run trains 26.2M LoRA
parameters on 2,848 train rows and evaluates 512 validation plus 448 held-out
test rows. It supervises noisy future vision velocity under camera-pose action
conditioning, not semantic JSON labels or `preds_action`; supervised
action-token prediction still needs a separate policy or inverse-dynamics
target export.

### 7. Xperience Embodied Foundation Model Pretraining

This stage is the long-term full-corpus goal. Instead of adapting an existing
backbone, it would pretrain a domain model directly on the synchronized
Xperience-10M modality structure: video, audio, depth, pose/SLAM, hand/body
mocap, IMU, calibration, and language annotations.

The first realistic target is a 3B-7B Xperience-native domain model after
smaller 0.3B-1B and 1B-3B pilots prove that the objectives and data loaders
scale. The training objective should combine masked multimodal modeling,
cross-modal alignment, future-state prediction, ego-motion and hand-motion
forecasting, action/procedure prediction, language grounding, contact and
affordance prediction, and optional policy-style targets after action
conversion.

This stage needs full-corpus access, PB-scale storage planning, high-throughput
media decoding, distributed training, reliable checkpoints, and held-out
evaluation across episodes, sessions, activities, objects, and missing
modalities. The plan is reader-facing in
`XPERIENCE_EMBODIED_FOUNDATION_MODEL_PRETRAINING.md`.

## Public Artifacts That Should Move Together

When a roadmap stage advances, update these public surfaces together:

- `README.md`
- `PROJECT_STATUS.md`
- `RESEARCH_TAKEAWAYS.md`
- `EVALUATION_PROTOCOL.md`
- `ARTIFACT_GUIDE.md`
- `ADDITIONAL_DEVELOPMENT_DIRECTIONS.md`
- `XPERIENCE_EMBODIED_FOUNDATION_MODEL_PRETRAINING.md`
- `docs/index.html`
- `docs/data/additional_development_directions.json`
- `docs/data/research_roadmap.json`
- Hugging Face Space, artifact dataset, and model cards

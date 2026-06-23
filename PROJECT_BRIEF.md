# Project Brief

This project presents Ropedia Xperience-10M through two public evidence lines.
Line 1 uses one public sample episode as a raw-inspectable 20-task embodied-AI
task lab. Line 2 uses selected-128 public-safe derived artifacts as a same-split
comparison surface for aligned baselines, Qwen3-Omni v6, Cosmos3-Super, and
Cosmos3-Nano.

## Research Intent

The public sample remains the raw-inspectable development harness; selected-128
is the model-comparison surface. The project does not blend those two evidence
types. A reader should be able to trace one model input, understand each task,
reproduce the public-sample results, compare the selected-128 method rows, and
see what remains before presenting stronger model-quality numbers.

## Capability Map

| Capability | Evidence in this project |
| --- | --- |
| Data understanding | `feature_manifest.json`, `available_modalities.json`, modality atlas, episode-window HF viewer, and selected-128 public-safe feature/index artifacts |
| Task design | 20 unified task contracts, task cards, case-study walkthroughs, and four research-direction extension probes |
| Evaluation rigor | chronological split, per-task metrics, predictions, confusion matrices, leakage notes, and generated takeaways |
| Scale-up planning | Final verified 96/16/16 Qwen3-Omni v6 diagnostic row, same-split selected-128 baseline alignment, Cosmos3-Nano compatibility diagnostics, Cosmos3-Super diagnostics, and policy-model candidates after action-space conversion |

## What Exists Now

| Evidence view | Current artifact |
| --- | --- |
| Line 1 data unit | 1 public sample episode, 5,821 frames, 1,161 synchronized 20-frame windows |
| Line 2 data unit | Selected 96/16/16 split over 128 source episodes, 34,269 Qwen3-Omni v6 multiscale windows, and public-safe processed features linked to official gated episode paths |
| Modalities | Video-derived features, audio, depth, pose/SLAM, mocap, IMU, calibration, and language-derived features |
| Task suite | 20 embodied-AI task contracts with inputs, targets, metrics, predictions, and setup alignment |
| Line 1 models | Minimal linear/ridge/logistic baselines plus compact PyTorch MLP heads for the unified 20-task public-sample suite |
| Line 2 methods | Metadata simple/NN, raw-feature simple/NN, Qwen3-Omni v6 LoRA, Cosmos3-Super Reasoner, and Cosmos3-Nano Future Window; 140/140 selected-128 scores, including 6 marked compact-proxy cells |
| Research map | Four Ropedia research directions with direct, proxy, diagnostic, and extension-task coverage |
| Qwen3 lineage | Qwen3-Omni v1-v6 are run versions inside Line 2: v1-v4 are pipeline-hardening/ablation evidence, v5 is the pinned prior multiscale release, and v6 is the current 20-task Qwen3-Omni row |

## How To Read It

1. Start with `PUBLIC_READER_MAP.md` if you need to choose between GitHub,
   the website, Hugging Face artifacts, baseline weights, model-result repos, or
   release-health files.
2. Start with the website or this brief to understand the project shape.
3. Open `RESEARCH_ROADMAP.md` to see how the work scales from the public
   sample to multi-episode modeling.
4. Open `EVALUATION_PROTOCOL.md` before comparing task scores.
5. Use `RESEARCH_TAKEAWAYS.md` for the current metric interpretation.
6. Inspect `results/episode_task_suite/feature_manifest.json` to understand one model input.
7. Use `TASK_SUITE_20.md` and `docs/data/task_suite_20.json` to read the unified 20-task suite; the historical `docs/data/tier2_task_suite.json` path stores provenance rows inside that same suite.
8. Use `QWEN3_OMNI_RUN_LINEAGE.md` and `docs/data/qwen3_omni_run_lineage.json` to read v1-v6 correctly.
9. Use `docs/data/omni_finetune_verified_result.json` for the current multi-episode Qwen3-Omni v6 result.

## What This Enables

Line 1 is enough to build and verify task definitions, feature contracts,
metrics, visualization, and baseline code. It is not enough to measure final
general embodied-AI model quality. Line 2 is the selected-128 held-out
comparison surface: it verifies same-split baselines, model diagnostics, and the
Qwen3-Omni v6 row. The next research stage is action/subtask error analysis,
stronger structured-output training, and policy-target conversion before broader
backbone comparisons.

## Best Entry Points

| Entry point | Link |
| --- | --- |
| Public reader map | https://github.com/ChaoYue0307/ropedia-xperience-10m-task-suite/blob/main/PUBLIC_READER_MAP.md |
| Visual dashboard | https://chaoyue0307.github.io/ropedia-xperience-10m-task-suite/ |
| Interactive HF Space | https://huggingface.co/spaces/cy0307/ropedia-xperience-10m-task-suite |
| Derived artifacts | https://huggingface.co/datasets/cy0307/ropedia-xperience-10m-task-suite-artifacts |
| Baseline model bundle | https://huggingface.co/cy0307/ropedia-xperience-10m-task-baselines |
| Official Xperience-10M dataset | https://huggingface.co/datasets/ropedia-ai/xperience-10m |

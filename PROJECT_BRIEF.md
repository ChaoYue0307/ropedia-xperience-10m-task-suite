# Project Brief

This project turns the public Ropedia Xperience-10M sample into a concrete
research task lab for embodied AI. It is designed to answer a practical
question: what can be built, measured, and extended from a richly synchronized
egocentric episode before scaling to held-out multi-episode training?

## Research Intent

The public sample is treated as a small but real research system. The project
does not try to inflate one episode into a final benchmark. Instead, it shows
the full path from data inspection to task design, baseline modeling,
evaluation, artifact packaging, and a guarded scale-up plan. A reader should be
able to trace one model input, understand each task, reproduce the public-sample
results, and see what remains before multi-episode model-quality claims.

## Capability Map

| Capability | Evidence in this project |
| --- | --- |
| Data understanding | `feature_manifest.json`, `available_modalities.json`, modality atlas, episode-window HF viewer |
| Task design | 12 task contracts, task cards, case-study walkthroughs, and four research-direction extension probes |
| Evaluation rigor | chronological split, per-task metrics, predictions, confusion matrices, leakage notes, and generated takeaways |
| Scale-up planning | Verified 96/16/16 Qwen3-Omni diagnostic pilot, validation-aware rerun path, Cosmos 3 branch, and policy-model candidates after action-space conversion |

## What Exists Now

| Layer | Current artifact |
| --- | --- |
| Data unit | 1 public sample episode, 5,821 frames, 1,161 synchronized 20-frame windows |
| Modalities | Video-derived features, audio, depth, pose/SLAM, mocap, IMU, calibration, and language-derived features |
| Task suite | 12 embodied-AI task contracts with inputs, targets, metrics, predictions, and case-study walkthroughs |
| Models | Minimal linear/ridge/logistic baselines plus compact PyTorch MLP heads for the same 12 tasks |
| Research map | Four Ropedia research directions with direct, proxy, diagnostic, and extension-task coverage |
| Scale-up path | A selected 96/16/16 Qwen3-Omni LoRA diagnostic pilot is verified; current model-quality metrics are weak and guide the next validation-aware rerun |

## How To Read It

1. Start with the website or this brief to understand the project shape.
2. Open `RESEARCH_ROADMAP.md` to see how the work scales from the public
   sample to multi-episode modeling.
3. Open `EVALUATION_PROTOCOL.md` before comparing task scores.
4. Use `RESEARCH_TAKEAWAYS.md` for the current metric interpretation.
5. Inspect `results/episode_task_suite/feature_manifest.json` to understand one model input.
6. Use `docs/data/omni_finetune_verified_result.json` for the current multi-episode Qwen3-Omni pilot result.

## What This Enables

The public sample is enough to build and verify task definitions, feature
contracts, metrics, visualization, and baseline code. It is not enough to
measure final model quality for a general embodied-AI model. The first
multi-episode Qwen3-Omni diagnostic pilot now verifies the held-out training
loop; the next research stage is to improve validation monitoring, JSON-format
reliability, and error analysis before larger robustness or alternative
backbone claims.

## Best Entry Points

| Entry point | Link |
| --- | --- |
| Visual dashboard | https://chaoyue0307.github.io/ropedia-xperience-10m-task-suite/ |
| Interactive HF Space | https://huggingface.co/spaces/cy0307/ropedia-xperience-10m-task-suite |
| Derived artifacts | https://huggingface.co/datasets/cy0307/ropedia-xperience-10m-task-suite-artifacts |
| Baseline model bundle | https://huggingface.co/cy0307/ropedia-xperience-10m-task-baselines |
| Official Xperience-10M dataset | https://huggingface.co/datasets/ropedia-ai/xperience-10m |

# Project Brief

This project turns the public Ropedia Xperience-10M sample into a concrete
research task lab for embodied AI. It is designed to answer a practical
question: what can be built, measured, and extended from a richly synchronized
egocentric episode before scaling to held-out multi-episode training?

## What Exists Now

| Layer | Current artifact |
| --- | --- |
| Data unit | 1 public sample episode, 5,821 frames, 1,161 synchronized 20-frame windows |
| Modalities | Video-derived features, depth, pose/SLAM, mocap, IMU, calibration, and language-derived features; audio is documented but not yet featurized |
| Task suite | 12 embodied-AI task contracts with inputs, targets, metrics, predictions, and case-study walkthroughs |
| Models | Minimal linear/ridge/logistic baselines plus compact PyTorch MLP heads for the same 12 tasks |
| Research map | Four Ropedia research directions with direct, proxy, diagnostic, and extension-task coverage |
| Scale-up path | Qwen3-Omni LoRA pilot code path prepared for 32 held-out episodes after gated data access |

## How To Read It

1. Start with the website or this brief to understand the project shape.
2. Open `EVALUATION_PROTOCOL.md` before comparing task scores.
3. Use `RESEARCH_TAKEAWAYS.md` for the current metric interpretation.
4. Inspect `results/episode_task_suite/feature_manifest.json` to understand one model input.
5. Use `results/omni_finetune/DATA_BLOCKER_REPORT.md` for the multi-episode gate.

## What This Enables

The public sample is enough to build and verify task definitions, feature
contracts, metrics, visualization, and baseline code. It is not enough to
measure final model quality for a general embodied-AI model. The next research
stage is to run the same contracts on held-out episodes, then fine-tune and
evaluate an omni-model with train/test separation at the episode level.

## Best Entry Points

| Entry point | Link |
| --- | --- |
| Visual dashboard | https://chaoyue0307.github.io/ropedia-xperience-10m-task-suite/ |
| Interactive HF Space | https://huggingface.co/spaces/cy0307/ropedia-xperience-10m-task-suite |
| Derived artifacts | https://huggingface.co/datasets/cy0307/ropedia-xperience-10m-task-suite-artifacts |
| Baseline model bundle | https://huggingface.co/cy0307/ropedia-xperience-10m-task-baselines |
| Official Xperience-10M dataset | https://huggingface.co/datasets/ropedia-ai/xperience-10m |

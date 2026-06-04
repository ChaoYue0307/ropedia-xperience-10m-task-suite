# Xperience Embodied Foundation Model Pretraining Goal

This document describes a future research direction for the project: a
domain-specific embodied foundation model pretrained on the full Xperience-10M
corpus, if full-episode access, storage, and compute become available.

Current status: this is a planning artifact. The public project currently
contains a public-sample task suite, lightweight baselines, Qwen3-Omni LoRA
preparation, and a smoke LoRA artifact. It does not currently contain a
from-scratch Xperience foundation model or full-corpus pretraining run.

## Why This Is A Natural Long-Term Goal

Xperience-10M is designed for physical-AI pretraining rather than only
single-task supervised learning. The official dataset card describes 10 million
experiences, 10,000 hours of synchronized first-person recordings, six video
streams, audio, stereo depth, camera pose, hand and full-body mocap, IMU, and
hierarchical language annotations. It also reports 2.88B RGB frames, 720M depth
frames, 576M pose/mocap frames, 7.2B IMU frames, and about 1 PB of total data.

That scale and alignment make a specific Xperience-native model plausible:
not a general web-scale omni model, but an embodied model specialized for
egocentric perception, human-object interaction, temporal dynamics, physical
state, and task intent.

## Target Model

The proposed model name is **Xperience Embodied Foundation Model**.

The model should learn a shared temporal representation of embodied experience:
what the wearer sees and hears, how the camera moves, how the body and hands
move, what objects are involved, what geometry is present, and what task is
being performed.

Expected modules:

| Module | Input | Role |
| --- | --- | --- |
| Multi-view video encoder | fisheye/stereo/RGB streams | visual state, egocentric context, object interaction |
| Audio encoder | synchronized MP4 audio | event cues, contact-like sound, temporal grounding |
| Depth and geometry encoder | depth, confidence, calibration | spatial structure and 3D/4D scene cues |
| Pose/SLAM encoder | camera trajectory and orientation | ego-motion, viewpoint, scene traversal |
| Mocap encoder | hand/body joints | human motion, hand-object interaction, affordance cues |
| IMU encoder | accelerometer/gyroscope streams | inertial dynamics and wearable motion |
| Language encoder/decoder | task/subtask/action/object annotations | semantic grounding and structured generation |
| Temporal fusion transformer | aligned per-window modality tokens | shared embodied representation across time |
| Task heads / decoders | fused representation | action, caption, future motion, retrieval, reconstruction, and world-state outputs |

## Pretraining Objectives

The model should not rely on one loss. It should combine complementary
objectives so that every modality contributes to the shared representation.

| Objective | What the model learns | Example output |
| --- | --- | --- |
| Masked multimodal modeling | recover hidden video/depth/sensor tokens from context | reconstructed latent patches or sensor features |
| Cross-modal contrastive alignment | align video, motion, audio, geometry, and language from the same time window | matching score or retrieval embedding |
| Future-state prediction | predict what changes after the current window | future visual/depth/motion latent |
| Ego-motion and hand-motion forecasting | model wearer/body dynamics | future camera delta or hand trajectory |
| Action and procedure prediction | connect physical state to task semantics | action, subtask, transition, next action |
| Language grounding and captioning | connect temporal windows to natural language | caption, object/action grounding, structured JSON |
| Contact and affordance prediction | learn interaction state from human-object motion | contact state, relevant object set |
| Optional policy-style targets | learn action-like outputs after target conversion | action token, motion chunk, retargeted policy target |

## Staged Pretraining Plan

### Stage 0: Data Contract And Quality Gate

Use the existing public-sample task suite to define the data contract. Before
pretraining, every episode must pass a strict manifest check:

- `annotation.hdf5` exists and is readable,
- video streams are present or missing views are explicitly recorded,
- audio can be extracted or marked unavailable,
- depth, pose, mocap, IMU, calibration, and language fields are indexed,
- windows are aligned by timestamp or frame index,
- train/val/test splits are episode-level, not window-level leakage splits,
- raw data remains outside public repos and Hugging Face artifact mirrors.

### Stage 1: 128-1,000 Episode Representation Pilot

Start with a smaller model and a selected subset. The goal is to test whether
the multimodal objectives train stably and improve held-out task performance.

Recommended scale:

- 128 to 1,000 episodes,
- frozen or lightly trainable video/audio encoders at first,
- 0.3B-1B temporal fusion model,
- all available sensor modalities represented as tokens,
- evaluation on the existing 12-task suite plus future-state/retrieval probes.

### Stage 2: 10K Episode Domain Model

Scale after the pilot proves value. This stage should train a stronger
Xperience-specific representation model rather than only fine-tuning a general
omni model.

Recommended scale:

- thousands to 10K episodes,
- 1B-3B parameter multimodal temporal model,
- mixed supervised, contrastive, and predictive objectives,
- held-out sessions and held-out activities,
- robustness to missing camera views and sensor dropout.

### Stage 3: Full-Corpus Xperience Embodied Foundation Model

Use this stage only if storage, data throughput, and multi-node compute are
available. The goal is a domain foundation model over embodied human experience,
not a general internet-scale language model.

Recommended scale:

- all available Xperience-10M episodes,
- 3B-7B domain model as a realistic first full-corpus target,
- larger models only after scaling curves justify the cost,
- mixture of reconstruction, retrieval, forecasting, language, and world-model
  objectives,
- downstream evaluation on held-out episodes, held-out sessions, unseen
  objects, unseen activities, and downstream robotics/world-model tasks.

## Hardware Requirements

These are planning ranges, not completed run measurements from this repo.

| Training goal | Typical compute | Storage and data path | Practical use |
| --- | --- | --- | --- |
| 0.3B-1B pilot | 8-32 modern 80GB-class data-center GPUs | tens of TB plus fast local cache | prove objectives and data loaders |
| 1B-3B domain model | 32-128 GPUs | 100TB-scale cache, high-throughput decoding | serious research-scale pretraining |
| 3B-7B full-corpus domain model | 128-512 GPUs | PB-scale storage plus 100-400Gbps networking | first full Xperience-native foundation model |
| 30B-class omni model from scratch | 512-2,000+ GPUs | PB-scale storage, multi-node orchestration, large checkpoint budget | lab-scale project, not the first target |
| frontier general omni model | thousands of GPUs | data beyond Xperience-10M plus large infrastructure | out of scope for this project |

For full-corpus work, storage is as important as GPU count:

- raw corpus storage around the official dataset scale,
- 1.5-3x extra capacity for derived shards, caches, checkpoints, and metadata,
- fast NVMe cache for active shards,
- parallel media decoding and feature extraction workers,
- distributed training with reliable checkpoint/restart,
- per-episode provenance and split manifests.

## Evaluation Protocol

The model should not be judged only by training loss. Evaluation should include:

- JSON validity and structured task metrics from the current task suite,
- action/subtask/contact/object metrics on held-out episodes,
- text-to-window and window-to-text retrieval,
- future ego-motion and hand-motion forecasting,
- cross-modal reconstruction and missing-modality robustness,
- held-out object/activity/session generalization,
- qualitative inspection of retrieved or generated future states,
- downstream transfer to Qwen3-Omni, Cosmos-style world modeling, and
  policy/action branches.

## Relationship To Existing Public Work

The current public project is the harness for this future model:

- the 12-task suite defines concrete input/output contracts,
- minimal and neural baselines provide initial supervised targets,
- audio/modality diagnostics show which signals contribute,
- Qwen3-Omni LoRA provides the first trainable multi-episode adapter path,
- Cosmos and policy branches define downstream model families,
- the pretraining goal unifies these into a long-term representation-learning
  direction.

The next practical step is still selected multi-episode preparation and
held-out Qwen3-Omni LoRA evaluation. Full-corpus pretraining should come after
the smaller scaling stages show measurable value.

## Source Links

- Official Xperience-10M dataset: https://huggingface.co/datasets/ropedia-ai/xperience-10m
- Ropedia Xperience-10M release page: https://ropedia.com/blog/20260316_xperience_10m
- Ropedia physical-AI data infrastructure page: https://ropedia-dev.com/

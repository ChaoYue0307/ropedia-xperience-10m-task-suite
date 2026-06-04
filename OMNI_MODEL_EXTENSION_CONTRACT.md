# Omni Model Extension Contract

This project uses one shared Xperience-10M data spine and separate backbone
adapters. Qwen3-Omni is the first implemented fine-tuning path; future
Cosmos-style world models and VLA/policy models should plug into the same
manifest, split, artifact, and evaluation discipline.

## Shared Pipeline

Every trainable branch should keep these stages:

1. **Episode selection:** choose complete Xperience-10M episodes before export.
2. **Episode split:** split by episode/session, not by adjacent windows.
3. **Manifest guard:** record every episode id, path, split, size, and missing
   modality before training.
4. **Backbone export:** convert raw windows into the model-specific sample
   format.
5. **Training:** save model config, adapter config, progress JSONL, and
   checkpoint path.
6. **Held-out evaluation:** evaluate on test episodes only after training.
7. **Run report:** write metrics, predictions, confusion matrices or
   task-specific scoring files, and skipped-episode reasons.

The current 128-episode pilot uses a fixed `96/16/16` train/val/test split by
episode.

## Backbone Registry

Backbone contracts live in:

```text
configs/omni_backbones/
```

Inspect them with:

```bash
python scripts/omni/backbone_registry.py --validate --json
```

Current contracts:

| Backbone | Status | Purpose |
| --- | --- | --- |
| `qwen3_omni_lora` | implemented | Structured episode-understanding JSON QA over video/audio/text plus sensor bridge features |
| `cosmos_world_model` | planned adapter | Future-window and action-conditioned world modeling |
| `policy_vla_branch` | planned adapter | Observation-to-action or motion-policy training after action-space conversion |

## Qwen3-Omni Contract

Qwen3-Omni consumes:

- rendered multi-camera mosaic video,
- extracted MP4 audio,
- language prompt and label options,
- optional sensor-bridge summaries/features.

It predicts strict JSON:

```json
{
  "action": "string",
  "subtask": "string",
  "objects": ["string"],
  "contact": "string",
  "transition": "string",
  "next_action": "string",
  "evidence_window": {"start_frame": 0, "end_frame": 0}
}
```

Implemented entrypoints:

- `scripts/omni/parallel_export_qwen3_omni_action_dataset.py`
- `scripts/omni/train_qwen3_omni_lora.py`
- `scripts/omni/eval_qwen3_omni_lora.py`
- `scripts/omni/run_128_fullsplit_parallel_export_8gpu.sh`

## Cosmos-Style World Model Contract

Cosmos-style work should not reuse the JSON QA exporter as-is. It needs a
future-window exporter with samples shaped like:

```json
{
  "episode_id": "session__ep",
  "split": "train",
  "context_window": {"start_frame": 0, "end_frame": 119},
  "target_window": {"start_frame": 120, "end_frame": 179},
  "conditioning": {
    "video": "path-or-latent",
    "audio": "path-or-features",
    "pose": "feature path",
    "depth": "feature path",
    "mocap": "feature path",
    "imu": "feature path",
    "language": "task context"
  },
  "target": {
    "future_video": "path-or-latent",
    "future_sensor_features": "path",
    "transition": "label"
  }
}
```

Minimum evaluators:

- future retrieval MRR / recall@5,
- temporal consistency,
- feature reconstruction error,
- transition/contact prediction,
- qualitative generated or retrieved examples.

## VLA / Policy Contract

Policy branches need an explicit action target before training. A valid sample
must state whether the target is an action class, next action, hand trajectory,
contact event, retargeted humanoid action, or robot-compatible action token.

The first policy exporter should save:

- observation media/features,
- language instruction or task context,
- action target,
- action normalization metadata fit on train episodes only,
- target provenance from the original annotation/mocap/contact fields.

Minimum evaluators:

- action or next-action accuracy,
- contact accuracy,
- trajectory MPJPE when trajectories are used,
- object-affordance F1,
- held-out episode count and leakage check.

## Non-Negotiable Invariants

- Do not train on held-out test episodes.
- Do not report model quality without predictions and metrics from held-out
  episodes.
- Do not redistribute raw gated MP4, HDF5, RRD, full checkpoint, or full model
  weight files.
- Do not treat a smoke run or one-episode overfit run as a real held-out model
  result.
- Record skipped episodes with reasons instead of silently dropping them.

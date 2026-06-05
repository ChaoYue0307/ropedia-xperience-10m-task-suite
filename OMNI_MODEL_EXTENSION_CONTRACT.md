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
8. **Long-run observability:** stream `progress.jsonl` and
   `predictions.partial.jsonl` during evaluation so multi-hour held-out runs can
   be monitored and resumed without changing the final metric definitions.

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

Create a new planned backbone config from an existing contract template with:

```bash
python scripts/omni/scaffold_omni_backbone.py \
  --template-backbone policy_vla_branch \
  --id new_policy_branch \
  --display-name "New Policy Branch" \
  --model-family "Model family name" \
  --dataset-contract xperience10m_observation_action_v1 \
  --training-objective observation_to_action_policy \
  --checkpoint-gate policy_checkpoint_action_space_and_normalizer \
  --dry-run
```

Current contracts:

| Backbone | Status | Purpose |
| --- | --- | --- |
| `qwen3_omni_lora` | implemented | Structured episode-understanding JSON QA over video/audio/text plus sensor bridge features |
| `cosmos_world_model` | planned adapter | Future-window and action-conditioned world modeling |
| `policy_vla_branch` | planned adapter | Observation-to-action or motion-policy training after action-space conversion |

## Model-Neutral Window Index

The Qwen exporter produces model-ready JSONL records. To avoid tying future
branches to Qwen chat-message formatting, convert those records into a
backbone-neutral window index:

```bash
python scripts/omni/export_model_neutral_window_index.py \
  --dataset-jsonl results/omni_finetune/<run_id>_dataset/dataset.jsonl
```

This writes:

- `window_index.jsonl`
- `window_index_manifest.json`

Each neutral record keeps the same episode split and window boundaries, then
separates:

- media paths,
- sensor feature pointers,
- language context,
- JSON supervision,
- Qwen, Cosmos-style, and policy/VLA adapter views.

Future exporters should consume this neutral index when possible, then add only
the model-specific target conversion that they need.

## Artifact Contract

Every backbone config must declare an `artifact_contract` with:

- `checkpoint_gate`: the model-specific checkpoint validation rule,
- `required_training_files`: files that prove training state and configuration,
- `required_eval_files`: files that prove held-out evaluation outputs,
- `public_package_allowed`: small derived artifacts that may be published,
- `public_package_forbidden`: raw data, weights, checkpoints, or large files
  that must stay out of public packages.

`scripts/omni/backbone_registry.py --validate --json` checks that the contract
exists for Qwen, Cosmos-style, and policy/VLA branches. The validator and
public-safe packager read `required_eval_files`, `primary_metrics`, and
publication rules from the selected backbone config. Export, training, and
evaluation code still remain model-specific, but the final validation and
publication gate follows the same contract for every future branch.

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
- `scripts/omni/watch_omni_train_then_eval.py`
- `scripts/omni/run_128_fullsplit_parallel_export_8gpu.sh`

The watcher is the current post-training gate runner. For the Qwen3-Omni LoRA
branch it waits for `progress.jsonl` to end in `complete`, checks the PEFT LoRA
safetensors shapes, runs the training validator, runs a held-out eval smoke,
then runs the full held-out test evaluation.

The Qwen evaluator writes partial predictions during inference and finalizes the
same `predictions.jsonl`, `predictions.csv`, `metrics.json`,
`confusion_matrix.csv`, and `RUN_REPORT.md` files after all selected held-out
windows finish. A restarted eval can resume from the partial prediction file.
For faster held-out evaluation, the Qwen evaluator can also run deterministic
sample shards via `--sample-offset` and `--sample-stride`. Sharded outputs must
be merged with `scripts/omni/merge_qwen3_omni_eval_shards.py`, which recomputes
the final metrics from combined predictions and checks missing or duplicate
sample ids.

Future model families can reuse the same wait/eval sequence only if their
checkpoint artifact has a compatible gate. Otherwise they should provide a
model-specific checkpoint check and evaluator, while keeping the same episode
split and held-out reporting discipline.

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

Cosmos-style checkpoints are not LoRA adapters by default. Their post-training
gate should verify generated latent/video checkpoints, model config, scheduler
state, and future-window evaluator outputs instead of using the Qwen LoRA
safetensors check.

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

Policy checkpoints should additionally save the action-space definition,
normalization statistics, and retargeting/conversion metadata. These must be
fit from train episodes only and validated before any held-out policy metrics
are reported.

## Non-Negotiable Invariants

- Do not train on held-out test episodes.
- Do not report model quality without predictions and metrics from held-out
  episodes.
- Do not redistribute raw gated MP4, HDF5, RRD, full checkpoint, or full model
  weight files.
- Do not treat a smoke run or one-episode overfit run as a real held-out model
  result.
- Record skipped episodes with reasons instead of silently dropping them.

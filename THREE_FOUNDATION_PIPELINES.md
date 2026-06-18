# Three Foundation Pipeline Tracks

Xperience-10M can support the three directions shown in the presentation, but
they should be positioned as **pipeline tracks**, not as three already solved
model claims. The same dataset can feed all three tracks because it combines
egocentric and multiview video, audio, depth, camera pose, hand/body motion,
inertial signals, object/contact annotations, and language captions.

## Track Summary

| Track | Question | Core inputs | First public pipeline | Current maturity |
| --- | --- | --- | --- | --- |
| Spatial intelligence models | Can the model recover and reason over space from video? | Multiview RGB, egocentric video, depth, camera pose, calibration, object cues, language questions. | Window or episode exporter that builds scene/object memory targets, then evaluates spatial QA, object permanence, counting, retrieval, and pose-aware consistency. | Ready as a pipeline and evaluation contract; strong claims require raw depth/pose access and held-out multi-episode evaluation. |
| Human-video world models | Can the model predict what happens next? | Observed video/audio/sensor windows, hand/body motion, object/contact state, action/subtask labels, future windows. | Future-state and future-action probes over the existing split, then Cosmos-style or latent world-model training with separate dynamics metrics. | Partially evidenced through current future-task probes and Cosmos-style branch artifacts; still needs stronger visual/latent future metrics. |
| Vision-language-action models | Can the model turn what it sees and reads into action? | Egocentric video, language captions, hand/body motion, contacts, objects, procedure/subtask labels. | Observation-language-to-action target conversion, action-chunk scoring, policy-token baselines, then VLA/policy model fine-tuning. | Feasible but gated by action-target conversion; do not claim policy quality until action tokens, normalization, and held-out policy metrics exist. |

## Published Direction Photos

The repo and public mirrors now include three high-resolution direction images
from the original direction slides. Spatial intelligence and human-video world
modeling now use the clean high-resolution slide PNGs supplied for publication;
the VLA card stays on the earlier restored source until a matching clean VLA
slide PNG is supplied. The 2026-06-18 refresh verified the two clean PNGs and
left VLA on the photo-restored source because the third uploaded image was a
duplicate of the Spatial intelligence slide. They are communication assets, not
evidence of completed model-quality training. The exact technical scope remains
the text and JSON contract in this document and
`docs/data/three_foundation_pipelines.json`.

| Track | Enhanced public asset | Committed source |
| --- | --- | --- |
| Spatial intelligence models | `docs/assets/foundation-pipelines/spatial-intelligence-pipeline.png` | `docs/assets/foundation-pipelines/source-slides/spatial-intelligence-slide.png` |
| Human-video world models | `docs/assets/foundation-pipelines/human-video-world-model-pipeline.png` | `docs/assets/foundation-pipelines/source-slides/human-video-world-model-slide.png` |
| Vision-language-action models | `docs/assets/foundation-pipelines/vision-language-action-pipeline.png` | `docs/assets/foundation-pipelines/source-photos/vision-language-action-source.jpg` |

The deterministic restoration script is
`scripts/render_foundation_pipeline_diagrams.py`; it remains available for the
photo-derived VLA card, while the two clean slide PNGs are used directly after
public-resolution packaging.

## 1. Spatial Intelligence Pipeline

Purpose: train and evaluate models that turn flat video into spatial state and
spatial reasoning.

Data contract:

- Inputs: multiview RGB, egocentric RGB, depth, camera pose, calibration, object
  labels, contact labels, optional language queries.
- Intermediate artifacts: synchronized camera window manifest, pose/depth
  availability report, scene/object memory records, object permanence targets,
  spatial relation targets, and spatial QA prompts.
- Outputs: object count, object persistence, relative location, 3D geometry
  consistency, multiview retrieval, camera-motion-aware scene memory, and
  language answers grounded in the scene.

First practical implementation:

1. Build a spatial-memory exporter over the same episode split discipline.
2. Start with metric depth and pose consistency tasks before adding heavier
   reconstruction.
3. Evaluate with retrieval rank, count accuracy, relation accuracy, and
   object-memory consistency.
4. Add qualitative scene-memory examples only when they are backed by saved
   target files and metrics.

What to avoid claiming now: full neural rendering, full 3D reconstruction, or
general spatial intelligence unless those artifacts and held-out metrics exist.

## 2. Human-Video World Model Pipeline

Purpose: train and evaluate models that predict future state from human
interaction video.

Data contract:

- Inputs: observed video/audio/sensor windows, hand/body motion, camera pose,
  object/contact state, action/subtask labels, and optional language context.
- Intermediate artifacts: observed/future window pairs, future label targets,
  action-conditioned target records, visual or latent reconstruction targets,
  and temporal consistency metadata.
- Outputs: next action, next subtask, future object set, future state embedding,
  camera-motion delta, contact transition, and future-window quality metrics.

First practical implementation:

1. Keep Qwen-style structured future probes for task-level interpretability.
2. Keep Cosmos-style branches separate because they answer dynamics and visual
   future questions, not JSON task classification.
3. Add latent or feature-reconstruction metrics before claiming world-model
   quality.
4. Compare future-task metrics by held-out episode, task family, and visible
   object/action family.

What to avoid claiming now: a strong world model from low structured scores
alone. The public result should say that the pipeline and first probes exist,
while stronger future-state training remains the next step.

## 3. Vision-Language-Action Pipeline

Purpose: train and evaluate models that map visual-language context to action
chunks or policy-compatible targets.

Data contract:

- Inputs: egocentric video, language captions, hand/body motion, object/contact
  state, action/subtask labels, and optional retargeting metadata.
- Intermediate artifacts: action-token vocabulary, action-chunk windows,
  normalization stats, retargeting report, leakage audit, and action-space
  model card.
- Outputs: next action, action chunk, object-conditioned action, contact state,
  subtask transition, and policy/VLA held-out metrics.

First practical implementation:

1. Define the action space before training any policy model.
2. Start with next-action, next-subtask, contact, and object-conditioned action
   tasks using the existing 20-task surface.
3. Add hand-trajectory or robot-compatible action chunks only after the
   conversion is traceable.
4. Treat OpenVLA, openpi, GR00T, Octo, and SmolVLA-style models as policy
   branches that inherit the same split, manifest, and package rules.

What to avoid claiming now: robot policy quality. The current project can
build the conversion and scoring pipeline; policy quality needs action-space
evidence and held-out model results.

## Shared Pipeline Discipline

All three tracks should reuse the same public discipline:

- episode-level train/validation/test split,
- manifest-first exporters,
- no target leakage from future labels or captions into inputs unless the task
  explicitly asks for them,
- task-specific metrics and saved predictions,
- public-safe packages that exclude raw private data and heavyweight base
  model weights,
- website and model cards updated only after validators pass.

This framing lets the project pursue all three directions at once while keeping
the claims precise: spatial intelligence is the geometry/reasoning pipeline,
world modeling is the future-state pipeline, and VLA is the action-conversion
and policy pipeline.

# Additional Development Directions

This note records concrete directions that can grow from Xperience-10M beyond
the current minimal baselines, Qwen3-Omni LoRA plan, Cosmos/world-model branch,
and long-term Xperience-native pretraining goal. These are project directions,
not completed benchmark results.

| Direction | What to build first | Why it matters |
| --- | --- | --- |
| Episode taxonomy and data engine | Episode atlas, category tags, balance report, and split builder across activities, objects, scenes, people, sessions, and missing modalities. | Fine-tuning quality depends on selecting representative episodes instead of sampling randomly from a large corpus. |
| Standardized benchmark protocol | Fixed train/val/test manifests, task cards, leakage checks, metric scripts, and small reference baselines. | Makes future model results comparable across Qwen, Cosmos-style world models, policy models, and smaller task heads. |
| Multimodal representation learning | Contrastive and masked-prediction objectives over video, audio, depth, pose, mocap, IMU, and language windows. | Turns Xperience-10M into a reusable encoder-learning dataset before committing to expensive large-model training. |
| Skill and procedure graph mining | Segment actions into steps, transitions, preconditions, effects, and temporal skill graphs. | Connects egocentric perception to task structure, planning, and long-horizon embodied reasoning. |
| Human-object interaction and affordance modeling | Contact, hand-object state, reachable object, likely tool use, and next-affordance prediction tasks. | Uses the dataset's hands, mocap, objects, contacts, and language to model what actions the scene affords. |
| 3D/4D scene and object memory | Fuse depth, pose/SLAM, multiview video, and object cues into persistent scene/object maps. | Moves beyond frame-level recognition toward world-state tracking, object permanence, and spatial reasoning. |
| Data quality, synchronization, and missing-modality diagnostics | Per-episode QA for timestamp drift, camera/audio/depth availability, calibration consistency, and corrupted files. | Large multimodal training fails quietly without strong data-quality gates; this should become a first-class artifact. |
| Policy, retargeting, and simulation transfer | Convert mocap/hand/contact traces into action tokens, robot-compatible targets, imitation-learning data, and simulation probes. | Creates a bridge from human egocentric experience to robot policies while keeping action-space assumptions explicit. |

## Practical Order

1. Build the episode taxonomy and data-quality diagnostics first.
2. Lock the benchmark protocol and split manifests before reporting model scores.
3. Add representation-learning and skill-graph objectives once enough episodes
   are staged.
4. Add affordance, 3D/4D memory, and policy-retargeting branches after the
   labels and action targets are measurable.

The current public sample is useful for prototyping the contracts and visual
explanations. Strong claims for these directions require multi-episode training,
held-out evaluation, and artifact-level evidence.

# Foundation Model Plan

This plan extends the current Xperience-10M scale-up path beyond the prepared
Qwen3-Omni LoRA pilot. It separates immediate trainable work from later
world-model and robot-policy branches, so the project can choose a backbone
without mixing different research goals.

Current status: this remains the backbone-selection plan, but the repo now has
verified held-out multi-episode foundation-model diagnostics: Qwen3-Omni LoRA
for structured JSON tasks, Cosmos3-Nano for future-window compatibility,
Cosmos3-Super Reasoner as a base-weight JSON-task evaluation, and Cosmos3-Super
Forward-Dynamics LoRA as the first fine-tuned Super adapter branch.

## Backbone Decision

| Priority | Model family | Best role for this project | Why it fits Xperience-10M | Current decision |
| --- | --- | --- | --- | --- |
| 1 | Qwen3-Omni | Multimodal instruction model and JSON task predictor | Accepts video/audio/language directly; depth, pose, mocap, and IMU can enter through the existing sensor bridge | Keep as the first selected-episode LoRA pilot |
| 2 | Cosmos 3 | Embodied world model, action generation, and synthetic future prediction | Designed for physical-world video generation, action-conditioned world modeling, and robot/world simulation style objectives | Add as the first world-model branch after the data gate |
| 3 | NVIDIA GR00T | Humanoid/action-policy foundation model | Xperience-10M mocap, hand motion, contacts, and egocentric interaction can support retargeting and action-understanding probes | Track as a humanoid policy branch, not the first LoRA pilot |
| 4 | OpenVLA / OpenVLA-OFT | Open vision-language-action policy baseline | Useful when windows are converted into visual observation plus action-token targets | Use after action-space design is explicit |
| 5 | openpi pi0/pi0.5 | Open robot policy and action expert baseline | Useful for action chunking, policy fine-tuning, and embodiment transfer experiments | Candidate for policy branch once action labels are retargeted |
| 6 | Gemini Robotics | Closed/API embodied reasoning reference | Strong candidate for qualitative reasoning and task interpretation, but not a local fine-tune target | Use only as an external comparison or annotation assistant |
| 7 | Octo / SmolVLA-style lightweight policies | Smaller reproducible robot-policy baselines | Good for cheaper action-policy experiments, but less directly omni-modal | Optional baseline branch after selected-episode data preparation |
| Future | Xperience Embodied Foundation Model | Xperience-native domain model pretrained from scratch on full-corpus embodied experience | Would learn a shared temporal representation across video, audio, depth, pose, mocap, IMU, and language | Long-term goal after smaller pilots prove value and full-corpus storage/compute are available |

## Why Qwen3-Omni Still Goes First

The immediate pilot is about proving the full data path:

- prepared multi-episode Xperience-10M data,
- episode-level train/test separation,
- window-level supervised examples,
- multimodal prompt construction,
- sensor bridge for depth, pose, mocap, and IMU,
- LoRA training,
- held-out predictions and metrics.

Qwen3-Omni is the most direct first target because the existing scripts already
prepare video/audio/language prompts and adapter inputs. It is also suitable for
the 12 current task contracts, which mostly produce labels, structured JSON, or
short task answers.

The executable Qwen branch and future branch contracts are now represented as
config files under `configs/omni_backbones/`. Validate them with:

```bash
python scripts/omni/backbone_registry.py --validate --json
```

The shared extension rules are in
[`OMNI_MODEL_EXTENSION_CONTRACT.md`](OMNI_MODEL_EXTENSION_CONTRACT.md). A new
foundation branch should add a config first, then implement the exporter,
trainer, evaluator, and launcher required by that config.

## Long-Term Native Pretraining Goal

Qwen3-Omni, Cosmos 3, GR00T, OpenVLA, and openpi are backbone choices for the
next experiments. The longer-term goal is different: train an
**Xperience Embodied Foundation Model** that is native to the Xperience-10M
modality structure.

That model would not start as a general internet-scale omni model. It would be
a domain model over synchronized embodied experience: multi-view egocentric
video, audio, depth, pose/SLAM, hand and body mocap, IMU, calibration, and
language annotations. Its pretraining should combine masked multimodal
modeling, cross-modal contrastive alignment, future-state prediction,
ego-motion and hand-motion forecasting, action/procedure prediction, language
grounding, contact/affordance prediction, and optional policy-style targets
after action conversion.

This is not a current result in the repo. It becomes appropriate only after:

- the selected multi-episode pipeline trains and evaluates cleanly,
- scaling from 128 episodes to thousands of episodes shows measurable value,
- raw-corpus storage and derived-shard capacity are available,
- distributed training and checkpoint/restart infrastructure are reliable,
- evaluation covers held-out episodes, sessions, activities, objects, and
  missing-modality robustness.

The full plan is documented in
[`XPERIENCE_EMBODIED_FOUNDATION_MODEL_PRETRAINING.md`](XPERIENCE_EMBODIED_FOUNDATION_MODEL_PRETRAINING.md).

## Why Cosmos 3 Should Be Added Next

Cosmos 3 should not replace the Qwen3-Omni pilot. It should become the first
world-model branch after the data gate. The reason is that the Xperience-10M
modalities are unusually aligned with physical-world modeling:

- video streams for visual state,
- embedded audio for event cues,
- depth and calibration for spatial structure,
- pose/SLAM for camera motion,
- hand/body mocap for embodied state,
- IMU for inertial dynamics,
- language annotations for task semantics.

The practical Cosmos 3 branch should start with three targets:

1. **Future-window prediction:** condition on earlier video/sensor windows and
   predict future visual or latent state.
2. **Action-conditioned world modeling:** use mocap/action labels as controls
   and predict what changes in the scene.
3. **Synthetic data expansion:** generate or score candidate futures, then test
   whether synthetic windows improve downstream task heads.

A Cosmos 3 branch is now represented by two public-safe verified packages:
Cosmos3-Nano future-window compatibility and Cosmos3-Super forward-dynamics
LoRA. The Super LoRA target is camera-pose-conditioned future vision velocity,
so it should be analyzed as a world-model loss result rather than a JSON-task
classifier.

## Policy-Model Branch

OpenVLA, openpi, GR00T, Octo, and SmolVLA-style models should be treated as
policy/action branches. They need a clear action target before training:

- egocentric action class,
- next subtask,
- hand trajectory chunk,
- contact state,
- object-affordance target,
- retargeted humanoid/body action,
- or robot-compatible action tokens.

The current public sample can prototype the data conversion, but policy quality
requires multi-episode diversity. The first useful policy experiment should be a
64-128 episode run, not a one-sample demonstration.

## Evaluation Additions

The foundation-model stage should add metrics beyond the current 12-task suite:

| Evaluation target | Metric family | Applies to |
| --- | --- | --- |
| Structured task prediction | JSON validity, macro-F1, accuracy, micro-F1 | Qwen3-Omni, Gemini Robotics comparison |
| Future state prediction | retrieval rank, temporal consistency, feature reconstruction, visual inspection | Cosmos 3 |
| Action-conditioned dynamics | transition accuracy, contact accuracy, next-action accuracy | Cosmos 3, OpenVLA, openpi, GR00T |
| Affordance and object interaction | object micro-F1, contact-object consistency, caption grounding | all branches |
| Cross-episode generalization | held-out episode metrics, held-out session metrics, leakage checks | all trainable branches |

## Execution Order

1. Keep the selected 96/16/16 split as the comparison spine.
2. Treat the verified Qwen3-Omni LoRA package as the structured JSON baseline.
3. Treat Cosmos3-Nano compatibility and Cosmos3-Super Forward-Dynamics LoRA as separate world-model branches with different metrics.
4. Run a model-selection dry run on 3-8 episodes for any next backbone before scaling beyond the selected split.
5. Promote Cosmos 3 to larger world-model experiments if video/sensor
   preprocessing, storage, and loss metrics justify the extra cost.
6. Promote OpenVLA/openpi/GR00T only after action targets are explicit and
   retargeting artifacts are traceable.
7. Update public cards only when a branch has real manifests, predictions,
   metrics, and qualitative examples.
8. Start Xperience-native pretraining only after smaller scaling stages,
   full-corpus storage, multi-node compute, and held-out evaluation protocols
   are in place.

## Source Links

- Qwen3-Omni: https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct
- NVIDIA Cosmos: https://www.nvidia.com/en-us/ai/cosmos/
- NVIDIA Isaac GR00T: https://developer.nvidia.com/isaac/gr00t
- OpenVLA: https://openvla.github.io/
- openpi: https://github.com/Physical-Intelligence/openpi
- Gemini Robotics: https://deepmind.google/discover/blog/gemini-robotics-brings-ai-into-the-physical-world/
- Octo: https://octo-models.github.io/
- LeRobot / SmolVLA: https://github.com/huggingface/lerobot
- Xperience Embodied Foundation Model pretraining plan:
  `XPERIENCE_EMBODIED_FOUNDATION_MODEL_PRETRAINING.md`

# Foundation Model Plan

This plan extends the current Xperience-10M scale-up path beyond the prepared
Qwen3-Omni LoRA pilot. It separates immediate trainable work from later
world-model and robot-policy branches, so the project can choose a backbone
without mixing different research goals.

Current status: this is a planning artifact. The public repo has verified
single-episode task heads and setup-stage Qwen3-Omni scripts. It has not yet
run a held-out multi-episode foundation-model evaluation.

## Backbone Decision

| Priority | Model family | Best role for this project | Why it fits Xperience-10M | Current decision |
| --- | --- | --- | --- | --- |
| 1 | Qwen3-Omni | Multimodal instruction model and JSON task predictor | Accepts video/audio/language directly; depth, pose, mocap, and IMU can enter through the existing sensor bridge | Keep as the first selected-episode LoRA pilot |
| 2 | Cosmos 3 | Embodied world model, action generation, and synthetic future prediction | Designed for physical-world video generation, action-conditioned world modeling, and robot/world simulation style objectives | Add as the first world-model branch after the data gate |
| 3 | NVIDIA GR00T | Humanoid/action-policy foundation model | Xperience-10M mocap, hand motion, contacts, and egocentric interaction can support retargeting and action-understanding probes | Track as a humanoid policy branch, not the first LoRA pilot |
| 4 | OpenVLA / OpenVLA-OFT | Open vision-language-action policy baseline | Useful when windows are converted into visual observation plus action-token targets | Use after action-space design is explicit |
| 5 | openpi pi0/pi0.5 | Open robot policy and action expert baseline | Useful for action chunking, policy fine-tuning, and embodiment transfer experiments | Candidate for policy branch once action labels are retargeted |
| 6 | Gemini Robotics | Closed/API embodied reasoning reference | Strong candidate for qualitative reasoning and task interpretation, but not a local fine-tune target | Use only as an external comparison or annotation assistant |
| 7 | Octo / SmolVLA-style lightweight policies | Smaller reproducible robot-policy baselines | Good for cheaper action-policy experiments, but less directly omni-modal | Optional baseline branch after selected-episode data staging |

## Why Qwen3-Omni Still Goes First

The immediate pilot is about proving the full data path:

- staged multi-episode Xperience-10M data,
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

Do not claim a Cosmos 3 result until there are committed manifests, generated
outputs, held-out metrics, and qualitative examples.

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
| Cross-episode generalization | held-out episode metrics, held-out session metrics, leakage audit | all trainable branches |

## Execution Order

1. Finish multi-episode data staging for the selected relay.
2. Run the Qwen3-Omni LoRA pilot exactly once as the first held-out baseline.
3. Run a model-selection dry run on 3-8 episodes: Qwen3-Omni prompt-only,
   Qwen3-Omni LoRA, Cosmos 3 world-model preprocessing, and one policy baseline.
4. Promote Cosmos 3 to the first world-model experiment if video/sensor
   preprocessing and storage fit.
5. Promote OpenVLA/openpi/GR00T only after action targets are explicit and
   retargeting artifacts are traceable.
6. Update public cards only when a branch has real manifests, predictions,
   metrics, and qualitative examples.

## Source Links

- Qwen3-Omni: https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct
- NVIDIA Cosmos: https://www.nvidia.com/en-us/ai/cosmos/
- NVIDIA Isaac GR00T: https://developer.nvidia.com/isaac/gr00t
- OpenVLA: https://openvla.github.io/
- openpi: https://github.com/Physical-Intelligence/openpi
- Gemini Robotics: https://deepmind.google/discover/blog/gemini-robotics-brings-ai-into-the-physical-world/
- Octo: https://octo-models.github.io/
- LeRobot / SmolVLA: https://github.com/huggingface/lerobot

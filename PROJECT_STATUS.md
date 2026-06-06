# Project Status

This is the fastest way to understand the current research project state.
It summarizes what has already been implemented from the public
Xperience-10M sample, what the first multi-episode Qwen3-Omni diagnostic pilot
shows, and which artifacts support the next development step.

## Research Positioning

The project is a research-engineering study of Xperience-10M rather than a
single demo result. It makes the public sample episode inspectable, defines
embodied-AI tasks over synchronized modalities, records baseline behavior, and
keeps the next multi-episode modeling stage explicit. The current evidence is
useful for judging data understanding, task design, evaluation discipline, and
scale-up readiness; it is not presented as final full-dataset model quality.

| Area | Current state | Evidence | Research readout |
| --- | --- | --- | --- |
| Public-sample pipeline | Verified | `results/episode_task_suite/summary_report.json`, `results/episode_task_suite/windows.csv`, `results/episode_task_suite/feature_manifest.json` | One public Xperience-10M sample episode is converted into 5,821 frames, 1,161 aligned 20-frame windows, and an 8,546-dimensional current feature contract. |
| Task suite | Verified | `scripts/episode_task_suite.py`, `results/episode_task_suite/`, `docs/data/summary_metrics.json` | All 12 task contracts have committed metrics, predictions, and minimal baseline outputs. |
| Neural heads | Verified | `scripts/neural_task_models.py`, `results/episode_task_suite/neural_mlp/` | Each task also has a compact PyTorch MLP run over the same feature tensor and chronological split. |
| Audio contribution study | Verified | `scripts/audio_ablation_and_raw_upgrade.py`, `results/audio_ablation/`, `docs/data/audio_ablation_summary.json` | Audio variants are compared across all 12 task contracts; audio improves the primary metric on 6 of 12 tasks, and a 588-d audio-window representation improves over the baseline audio variant on 6 of 12 tasks. |
| Research takeaways | Verified | `RESEARCH_TAKEAWAYS.md`, `docs/data/research_takeaways.json`, `scripts/build_research_takeaways.py` | The main result interpretation is generated from committed metrics: chronological class shift, neural gains on dynamics/order/alignment, open retrieval/reconstruction problems, and the need for held-out episodes. |
| Research roadmap | Current | `RESEARCH_ROADMAP.md`, `docs/data/research_roadmap.json` | The roadmap connects public-sample task development to the verified validation-aware Qwen3-Omni diagnostic baseline, structured-output improvement pass, robustness runs, world/policy branches, and the future Xperience-native pretraining goal. |
| Foundation-model plan | Current | `FOUNDATION_MODEL_PLAN.md`, `docs/data/foundation_model_plan.json` | Qwen3-Omni remains the first trainable held-out LoRA baseline; Cosmos 3 is added as the first world-model/action-generation branch; OpenVLA/openpi/GR00T are policy candidates after action targets are explicit. |
| Omni model extension contract | Current | `OMNI_MODEL_EXTENSION_CONTRACT.md`, `configs/omni_backbones/`, `scripts/omni/backbone_registry.py`, `scripts/omni/smoke_test_backbone_packaging.py` | Future model branches must keep the same episode split discipline, held-out metrics, validation gate, public-safe package contract, and explicit forbidden-artifact policy before reporting results. |
| Xperience Embodied Foundation Model | Future goal | `XPERIENCE_EMBODIED_FOUNDATION_MODEL_PRETRAINING.md` | A future full-corpus pretraining plan describes target modules, objectives, staged scale-up, hardware ranges, and evaluation for a domain-specific embodied foundation model. |
| Evaluation protocol | Verified | `EVALUATION_PROTOCOL.md`, `docs/data/evaluation_protocol.json`, `scripts/build_evaluation_protocol.py` | Windowing, chronological split, per-task metrics, leakage controls, and current limitations are generated from committed metric artifacts. |
| Dataset context | Verified | `XPERIENCE10M_DATASET_CARD_ALIGNMENT.md`, official Xperience-10M and sample cards | The README and dashboard distinguish the public sample used here from the gated full dataset used for the selected multi-episode pilot. |
| Public dashboard and Hub pages | Verified | GitHub Pages, HF Space, artifact dataset, baseline model repo, Qwen3-Omni LoRA repo | Readers can move between the website, code, derived artifacts, baseline weights, and Qwen3-Omni pilot status without needing local infrastructure details. |
| Public package policy | Verified | `DATA_NOTICE.md`, `REPRODUCIBILITY.md` | Raw Xperience-10M data, private gated files, large archives, credentials, and full Qwen weights are not redistributed. |
| Reproducibility | Verified for the public sample | `REPRODUCIBILITY.md`, `docs/data/reproducibility_matrix.json`, `notes/reproducibility_audit.md` | The public sample workflow has explicit commands, expected outputs, and exact-match reproduction evidence. |
| Qwen3-Omni fine-tuning | Verified validation-aware diagnostic held-out pilot; quality target not met | `docs/data/omni_finetune_verified_result.json`, `results/omni_finetune/verified_public/`, `scripts/omni/package_verified_omni_result.py`, `scripts/omni/audit_verified_omni_package.py` | The selected 96/16/16 episode split produced a validation-aware public-safe held-out package with 3,808 exported windows, 512 validation windows, and 448 test predictions. JSON validity is 87.50%, below the 98% target, so the result is a diagnostic baseline and the next pass should focus on structured-output improvements and error analysis. |
| Raw Xperience-10M redistribution | Not included | `DATA_NOTICE.md`, `docs/data/publication_audit.json` | Raw MP4, HDF5, RRD files, private gated data, and full Qwen weights are intentionally excluded. |

## Fast Research Route

1. Read this status file to establish the current project scope.
2. Open the visual dashboard for the fastest overview of data, tasks,
   directions, and scale-up status.
3. Inspect `RESEARCH_TAKEAWAYS.md` and
   `docs/data/research_takeaways.json` for the generated result interpretation.
4. Inspect `RESEARCH_ROADMAP.md` and `docs/data/research_roadmap.json` for
   the path from public-sample task work to multi-episode modeling.
5. Inspect `FOUNDATION_MODEL_PLAN.md` and
   `docs/data/foundation_model_plan.json` before choosing a backbone branch.
6. Inspect `OMNI_MODEL_EXTENSION_CONTRACT.md` and run
   `python scripts/omni/backbone_registry.py --validate --json` before adding
   a new Qwen, Cosmos-style, or VLA/policy branch.
7. Inspect `XPERIENCE_EMBODIED_FOUNDATION_MODEL_PRETRAINING.md` for the
   long-term full-corpus pretraining goal.
8. Inspect `docs/data/summary_metrics.json` and
   `results/episode_task_suite/neural_mlp/` to check the 12-task outputs.
9. Inspect `results/audio_ablation/AUDIO_ABLATION_SUMMARY.md` before judging
   whether audio helps the current task suite.
10. Inspect `EVALUATION_PROTOCOL.md` before judging task metrics or leakage
   controls.
11. Inspect `XPERIENCE10M_DATASET_CARD_ALIGNMENT.md` only if you need the
   detailed upstream dataset-card context.
12. Inspect `docs/data/omni_finetune_verified_result.json` before judging the
   Qwen3-Omni diagnostic pilot.

## Current Reading Notes

- Cross-episode generalization is a later multi-episode evaluation target; the
  current results use one public sample episode.
- Public-facing fine-tuning results should come from the verified result
  package, not from live process logs or setup-only artifacts.
- The first Qwen3-Omni held-out package verifies the pipeline, not strong model
  quality: JSON validity is 87.50%, action macro-F1 is 0.0027, and subtask
  accuracy is 0.0067.
- The current reconstruction task reconstructs feature vectors, not pixel
  depth, meshes, NeRF outputs, or Gaussian splats.
- Audio is part of the current 8,546-dimensional baseline feature vector.
- Audio contribution is evaluated across all 12 task contracts in
  `results/audio_ablation/`.
- Foundation-model selection is now explicit: Qwen3-Omni is the immediate
  trainable pilot, Cosmos 3 is the first world-model branch, and policy models
  such as OpenVLA/openpi/GR00T wait for action-target conversion.
- Future model branches should be added through the backbone registry and
  verified package contract, not by creating one-off result folders with
  incompatible metrics or publication rules.
- The Xperience Embodied Foundation Model is a future native-pretraining goal,
  not a completed model or current benchmark.

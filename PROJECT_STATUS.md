# Project Status

This is the fastest way to understand the current research project state.
It summarizes what has already been implemented from the public
Xperience-10M sample, what is being staged for multi-episode training, and
which artifacts support the next development step.

| Area | Current state | Evidence | Research readout |
| --- | --- | --- | --- |
| Public-sample pipeline | Verified | `results/episode_task_suite/summary_report.json`, `results/episode_task_suite/windows.csv`, `results/episode_task_suite/feature_manifest.json` | One public Xperience-10M sample episode is converted into 5,821 frames, 1,161 aligned 20-frame windows, and an 8,546-dimensional current feature contract. |
| Task suite | Verified | `scripts/episode_task_suite.py`, `results/episode_task_suite/`, `docs/data/summary_metrics.json` | All 12 task contracts have committed metrics, predictions, and minimal baseline outputs. |
| Neural heads | Verified | `scripts/neural_task_models.py`, `results/episode_task_suite/neural_mlp/` | Each task also has a compact PyTorch MLP run over the same feature tensor and chronological split. |
| Audio contribution study | Verified | `scripts/audio_ablation_and_raw_upgrade.py`, `results/audio_ablation/`, `docs/data/audio_ablation_summary.json` | Audio variants are compared across all 12 task contracts; audio improves the primary metric on 6 of 12 tasks, and a 588-d audio-window representation improves over the baseline audio variant on 6 of 12 tasks. |
| Research takeaways | Verified | `RESEARCH_TAKEAWAYS.md`, `docs/data/research_takeaways.json`, `scripts/build_research_takeaways.py` | The main result interpretation is generated from committed metrics: chronological class shift, neural gains on dynamics/order/alignment, open retrieval/reconstruction problems, and the need for held-out episodes. |
| Research roadmap | Current | `RESEARCH_ROADMAP.md`, `docs/data/research_roadmap.json` | The staged path connects public-sample task development to 128-episode data staging, Qwen3-Omni LoRA, foundation-model selection, robustness runs, and larger omni/world-model extensions. |
| Foundation-model plan | Current | `FOUNDATION_MODEL_PLAN.md`, `docs/data/foundation_model_plan.json` | Qwen3-Omni remains the first trainable held-out LoRA baseline; Cosmos 3 is added as the first world-model/action-generation branch; OpenVLA/openpi/GR00T are policy candidates after action targets are explicit. |
| Evaluation protocol | Verified | `EVALUATION_PROTOCOL.md`, `docs/data/evaluation_protocol.json`, `scripts/build_evaluation_protocol.py` | Windowing, chronological split, per-task metrics, leakage controls, and current limitations are generated from committed metric artifacts. |
| Official dataset wording | Verified | `XPERIENCE10M_DATASET_CARD_ALIGNMENT.md`, `docs/data/xperience10m_dataset_card_alignment.json` | Public wording is aligned to the official gated Xperience-10M dataset card, public sample card, and HF API metadata, including modalities, scale, access path, sample license/tooling, and current project coverage. |
| Source alignment | Verified | `SOURCE_ALIGNMENT_AUDIT.md`, `docs/data/source_alignment_audit.json`, `scripts/validate_source_alignment.py` | Source facts, sample details, API-listing notes, and project coverage are checked across repo docs, website, and HF cards. |
| Website and HF mirrors | Verified | `docs/data/website_integrity.json`, `docs/data/rendered_site_check.json`, `docs/data/mirror_parity.json`, `docs/data/live_publication_status.json` | Local website links/assets pass, the rendered walkthrough flow has a browser-level check, prepared mirrors match, and public GitHub/HF URLs have been verified after upload. |
| Public bundle contents | Verified | `docs/data/publication_audit.json`, `QUALITY_GATES.md`, `docs/data/quality_gates.json` | Public bundles exclude raw data, caches, heavy archives, token strings, and stale public-card copy. |
| Reproducibility | Verified for the public sample | `REPRODUCIBILITY.md`, `docs/data/reproducibility_matrix.json`, `notes/reproducibility_audit.md` | The public sample workflow has explicit commands, expected outputs, and exact-match reproduction evidence. |
| Qwen3-Omni fine-tuning | Data staging; full metrics pending | `results/omni_finetune/DATA_ACCESS_STATUS.md`, `results/omni_finetune/MULTI_EPISODE_ACCESS_STATUS.md` | Full-dataset access is granted and a 128-episode selected relay is in progress with chunked parallel transfer and overlapping batch prefetch; final held-out metrics require completed staging, manifest construction, training, and evaluation. |
| Raw Xperience-10M redistribution | Not included | `DATA_NOTICE.md`, `docs/data/publication_audit.json` | Raw MP4, HDF5, RRD files, private gated data, and full Qwen weights are intentionally excluded. |

## Fast Research Route

1. Read this status file and `EVIDENCE_CONTRACT.md` to establish the current
   project scope.
2. Open `docs/data/project_packet.json` for the machine-readable project path.
3. Inspect `RESEARCH_TAKEAWAYS.md` and
   `docs/data/research_takeaways.json` for the generated result interpretation.
4. Inspect `RESEARCH_ROADMAP.md` and `docs/data/research_roadmap.json` for
   the staged path from public-sample task work to multi-episode modeling.
5. Inspect `FOUNDATION_MODEL_PLAN.md` and
   `docs/data/foundation_model_plan.json` before choosing a backbone branch.
6. Inspect `docs/data/summary_metrics.json` and
   `results/episode_task_suite/neural_mlp/` to check the 12-task outputs.
7. Inspect `results/audio_ablation/AUDIO_ABLATION_SUMMARY.md` before judging
   whether audio helps the current task suite.
8. Inspect `EVALUATION_PROTOCOL.md` before judging task metrics or leakage
   controls.
9. Inspect `SOURCE_ALIGNMENT_AUDIT.md` and
   `XPERIENCE10M_DATASET_CARD_ALIGNMENT.md` before judging dataset
   wording.
10. Inspect `results/omni_finetune/DATA_ACCESS_STATUS.md` before judging
   Qwen3-Omni scale-up status.

## Current Reading Notes

- Cross-episode generalization is a later multi-episode evaluation target; the
  current results use one public sample episode.
- Older pilot path names refer to setup files, not completed held-out
  training results.
- The current reconstruction task reconstructs feature vectors, not pixel
  depth, meshes, NeRF outputs, or Gaussian splats.
- Audio is part of the current 8,546-dimensional baseline feature vector.
- Audio contribution is evaluated across all 12 task contracts in
  `results/audio_ablation/`.
- Foundation-model selection is now explicit: Qwen3-Omni is the immediate
  trainable pilot, Cosmos 3 is the first world-model branch, and policy models
  such as OpenVLA/openpi/GR00T wait for action-target conversion.

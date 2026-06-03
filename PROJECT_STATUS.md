# Project Status

This is the fastest way to understand the current research project state.
It summarizes what has already been implemented from the public
Xperience-10M sample, what remains data-gated, and which artifacts support
the next development step.

| Area | Current state | Evidence | Research readout |
| --- | --- | --- | --- |
| Public-sample pipeline | Verified | `results/episode_task_suite/summary_report.json`, `results/episode_task_suite/windows.csv`, `results/episode_task_suite/feature_manifest.json` | One public Xperience-10M sample episode is converted into 5,821 frames, 1,161 aligned 20-frame windows, and an 8,546-dimensional current feature contract. |
| Task suite | Verified | `scripts/episode_task_suite.py`, `results/episode_task_suite/`, `docs/data/summary_metrics.json` | All 12 task contracts have committed metrics, predictions, and minimal baseline outputs. |
| Neural heads | Verified | `scripts/neural_task_models.py`, `results/episode_task_suite/neural_mlp/` | Each task also has a compact PyTorch MLP run over the same feature tensor and chronological split. |
| Audio ablation and raw-audio upgrade | Verified | `scripts/audio_ablation_and_raw_upgrade.py`, `results/audio_ablation/`, `docs/data/audio_ablation_summary.json` | Current AAC audio improves the primary metric on 6 of 12 task contracts; replacing the current handcrafted block with a 588-d raw log-mel feature improves over current audio on 6 of 12 tasks. |
| Research takeaways | Verified | `RESEARCH_TAKEAWAYS.md`, `docs/data/research_takeaways.json`, `scripts/build_research_takeaways.py` | The main result interpretation is generated from committed metrics: chronological class shift, neural gains on dynamics/order/alignment, open retrieval/reconstruction problems, and the need for held-out episodes. |
| Research roadmap | Current | `RESEARCH_ROADMAP.md`, `docs/data/research_roadmap.json` | The staged path connects public-sample task development to multi-episode data staging, the 32-episode Qwen3-Omni LoRA pilot, robustness runs, and larger omni-model extensions. |
| Evaluation protocol | Verified | `EVALUATION_PROTOCOL.md`, `docs/data/evaluation_protocol.json`, `scripts/build_evaluation_protocol.py` | Windowing, chronological split, per-task metrics, leakage controls, and current limitations are generated from committed metric artifacts. |
| Official dataset wording | Verified | `XPERIENCE10M_DATASET_CARD_ALIGNMENT.md`, `docs/data/xperience10m_dataset_card_alignment.json` | Public wording is aligned to the official gated Xperience-10M dataset card, public sample card, and HF API metadata, including modalities, scale, access path, sample license/tooling, and current project coverage. |
| Source alignment | Verified | `SOURCE_ALIGNMENT_AUDIT.md`, `docs/data/source_alignment_audit.json`, `scripts/validate_source_alignment.py` | Source facts, sample details, API-listing notes, and project coverage are checked across repo docs, website, and HF cards. |
| Website and HF mirrors | Verified | `docs/data/website_integrity.json`, `docs/data/rendered_site_check.json`, `docs/data/mirror_parity.json`, `docs/data/live_publication_status.json` | Local website links/assets pass, the rendered walkthrough flow has a browser-level check, prepared mirrors match, and public GitHub/HF URLs have been verified after upload. |
| Public bundle contents | Verified | `docs/data/publication_audit.json`, `QUALITY_GATES.md`, `docs/data/quality_gates.json` | Public bundles exclude raw data, caches, heavy archives, token strings, and stale public-card copy. |
| Reproducibility | Verified for the public sample | `REPRODUCIBILITY.md`, `docs/data/reproducibility_matrix.json`, `notes/reproducibility_audit.md` | The public sample workflow has explicit commands, expected outputs, and exact-match reproduction evidence. |
| Qwen3-Omni fine-tuning | Data-gated; full metrics pending | `results/omni_finetune/DATA_ACCESS_STATUS.md`, `results/omni_finetune/MULTI_EPISODE_ACCESS_STATUS.md` | The 32-episode LoRA pilot is prepared; final held-out metrics require gated data access, manifest construction, training, and evaluation. |
| Raw Xperience-10M redistribution | Not included | `DATA_NOTICE.md`, `docs/data/publication_audit.json` | Raw MP4, HDF5, RRD files, private gated data, and full Qwen weights are intentionally excluded. |

## Fast Research Route

1. Read this status file and `EVIDENCE_CONTRACT.md` to establish the current
   project scope.
2. Open `docs/data/project_packet.json` for the machine-readable project path.
3. Inspect `RESEARCH_TAKEAWAYS.md` and
   `docs/data/research_takeaways.json` for the generated result interpretation.
4. Inspect `RESEARCH_ROADMAP.md` and `docs/data/research_roadmap.json` for
   the staged path from public-sample task work to multi-episode modeling.
5. Inspect `docs/data/summary_metrics.json` and
   `results/episode_task_suite/neural_mlp/` to check the 12-task outputs.
6. Inspect `results/audio_ablation/AUDIO_ABLATION_SUMMARY.md` before judging
   whether audio helps the current task suite.
7. Inspect `EVALUATION_PROTOCOL.md` before judging task metrics or leakage
   controls.
8. Inspect `SOURCE_ALIGNMENT_AUDIT.md` and
   `XPERIENCE10M_DATASET_CARD_ALIGNMENT.md` before judging dataset
   wording.
9. Inspect `results/omni_finetune/DATA_ACCESS_STATUS.md` before judging
   Qwen3-Omni scale-up status.

## Current Reading Notes

- Cross-episode generalization is a later multi-episode evaluation target; the
  current results use one public sample episode.
- Historical `32ep` path names refer to setup files, not completed 32-episode
  training results.
- The current reconstruction task reconstructs feature vectors, not pixel
  depth, meshes, NeRF outputs, or Gaussian splats.
- AAC audio is decoded from `fisheye_cam0.mp4` and included in the current
  8,546-dimensional baseline feature vector.
- Audio is now evaluated directly: the current AAC block and a raw log-mel
  replacement are compared across all 12 task contracts in
  `results/audio_ablation/`.

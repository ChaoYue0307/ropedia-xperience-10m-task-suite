# Reviewer Scorecard

This scorecard is the fastest way to decide what the current release proves.
It is intentionally stricter than the visual presentation: a row is marked
verified only when committed artifacts and validation reports support it.

| Area | Current decision | Evidence | Reviewer readout |
| --- | --- | --- | --- |
| Public-sample pipeline | Verified | `results/episode_task_suite/summary_report.json`, `results/episode_task_suite/windows.csv`, `results/episode_task_suite/feature_manifest.json` | One public Xperience-10M sample episode is converted into 5,821 frames, 1,161 aligned 20-frame windows, and an 8,378-dimensional current feature contract. |
| Task suite | Verified | `scripts/episode_task_suite.py`, `results/episode_task_suite/`, `docs/data/summary_metrics.json` | All 12 task contracts have committed metrics, predictions, and minimal baseline outputs. |
| Neural heads | Verified | `scripts/neural_task_models.py`, `results/episode_task_suite/neural_mlp/` | Each task also has a compact PyTorch MLP run over the same feature tensor and chronological split. |
| Evaluation protocol | Verified | `EVALUATION_PROTOCOL.md`, `docs/data/evaluation_protocol.json`, `scripts/build_evaluation_protocol.py` | Windowing, chronological split, per-task metrics, leakage controls, and unsupported interpretations are generated from committed metric artifacts. |
| Official dataset wording | Verified | `XPERIENCE10M_DATASET_CARD_ALIGNMENT.md`, `docs/data/xperience10m_dataset_card_alignment.json` | Public wording is aligned to the official gated Xperience-10M dataset card, public sample card, and HF API metadata, including modalities, scale, access boundary, sample license/tooling, and unsupported claims. |
| Website and HF mirrors | Verified | `docs/data/website_integrity.json`, `docs/data/mirror_parity.json`, `docs/data/live_publication_status.json` | Local website links/assets pass, prepared mirrors match, and public GitHub/HF URLs have been checked after upload. |
| Publication hygiene | Verified | `docs/data/publication_audit.json`, `QUALITY_GATES.md`, `docs/data/quality_gates.json` | Public bundles are checked for raw-data exclusion, cache exclusion, heavy-archive exclusion, token-string hygiene, and stale presentation copy. |
| Reproducibility | Verified for the public sample | `REPRODUCIBILITY.md`, `docs/data/reproducibility_matrix.json`, `notes/reproducibility_audit.md` | The public sample workflow has explicit commands, expected outputs, and exact-match audit evidence. |
| Qwen3-Omni fine-tuning | Data-gated, not a model-quality claim | `results/omni_finetune/DATA_BLOCKER_REPORT.md`, `results/omni_finetune/A100_HF_RELAY_STATUS.md` | The 32-episode LoRA pilot is prepared, but no real held-out 32-episode result is claimed until gated data access, manifest construction, training, and held-out evaluation pass. |
| Raw Xperience-10M redistribution | Not included | `DATA_NOTICE.md`, `docs/data/publication_audit.json` | Raw MP4, HDF5, RRD files, private gated data, and full Qwen weights are intentionally excluded. |

## Fast Reviewer Route

1. Read this scorecard and `EVIDENCE_CONTRACT.md` to establish what is
   claimed.
2. Open `docs/data/reviewer_packet.json` for the machine-readable review path.
3. Inspect `docs/data/summary_metrics.json` and
   `results/episode_task_suite/neural_mlp/` to check the 12-task outputs.
4. Inspect `EVALUATION_PROTOCOL.md` before judging task metrics or leakage
   controls.
5. Inspect `XPERIENCE10M_DATASET_CARD_ALIGNMENT.md` before judging dataset
   wording.
6. Inspect `results/omni_finetune/DATA_BLOCKER_REPORT.md` before judging
   Qwen3-Omni scale-up status.

## Do Not Infer

- Do not infer cross-episode generalization from the single public sample.
- Do not treat historical `32ep` path names as real 32-episode training
  results.
- Do not treat the current reconstruction task as pixel-depth, mesh, NeRF, or
  Gaussian reconstruction.
- Do not assume audio has entered the current 8,378-dimensional baseline
  feature vector; it is documented and visualized but not yet featurized.

# Evidence Contract

This project is intentionally audit-first. Every visible claim should point to a
local artifact that a reader can inspect before trusting the dashboard.

| Claim | Current evidence | Status | Boundary |
| --- | --- | --- | --- |
| The public dataset description is aligned with the official gated Xperience-10M dataset card. | `XPERIENCE10M_DATASET_CARD_ALIGNMENT.md`, `docs/data/xperience10m_dataset_card_alignment.json` | Verified description alignment | Summarizes upstream public metadata and card facts; does not grant access or mirror raw data |
| The public Xperience-10M sample has been converted into aligned model windows. | `results/episode_task_suite/windows.csv`, `results/episode_task_suite/shared_windows.npz`, `results/episode_task_suite/summary_report.json` | Verified for 5,821 frames and 1,161 windows | One public sample episode only |
| The current feature contract is explicit and reviewable. | `results/episode_task_suite/feature_manifest.json`, `results/episode_task_suite/available_modalities.json` | Verified for an 8,378-d feature vector | Audio is present in MP4 streams but not yet a feature block |
| The public sample modalities are inspectable without raw data redistribution. | `docs/data/modality_atlas.json`, `docs/assets/modalities/`, website modality atlas | Verified derived thumbnail atlas | Thumbnails are presentation/review assets, not a replacement for official raw data access |
| The 12 task heads are real scripts and artifacts, not presentation placeholders. | `scripts/episode_task_suite.py`, `results/episode_task_suite/*/metrics.json`, `results/episode_task_suite/*/predictions.*` | Verified for all 12 task definitions | Chronological single-episode split, not cross-episode generalization |
| Minimal and neural heads use the same task contracts. | `scripts/neural_task_models.py`, `results/episode_task_suite/neural_mlp/`, `docs/assets/task_architectures.png` | Verified for 12 minimal heads and 12 neural MLP heads | Small heads only; not a foundation model |
| Four Ropedia research directions are mapped honestly as direct, proxy, or diagnostic evidence. | `results/episode_task_suite/research_directions/research_direction_taxonomy.json`, `docs/data/research_directions.json` | Verified taxonomy | Some directions remain proxy-only |
| Four extra direction probes are coded and evaluated. | `results/episode_task_suite/research_direction_extensions/research_direction_extension_results.json`, `docs/data/research_direction_extensions.json` | Verified single-episode probes | Not full human modeling, neural rendering, intent modeling, or world modeling solutions |
| Qwen3-Omni infrastructure has passed technical smoke checks. | `results/omni_finetune/RUN_REPORT.md`, `results/omni_finetune/dataset_manifest.json`, `results/omni_finetune/metrics_eval.json` | Smoke-only evidence | One episode, 128 train windows; not a 32-episode pilot |
| The real 32-episode LoRA pilot is blocked on gated data access, not on repo presentation. | `results/omni_finetune/DATA_BLOCKER_REPORT.md`, `results/omni_finetune/A100_HF_RELAY_STATUS.md`, `results/omni_finetune/source_discovery.json` | Blocker documented | No 32-episode metric should be claimed until the gate passes |
| Historical `32ep` path strings are not treated as 32-episode results. | `scripts/validate_scope_claims.py`, `docs/data/scope_claims_audit.json` | Verified pass | Classifies old run/path identifiers and fails if public presentation claims real 32-episode metrics |
| Prepared GitHub/Hugging Face mirrors carry matching critical files. | `scripts/validate_mirror_parity.py`, `docs/data/mirror_parity.json` | Verified pass | Compares prepared data files, visual assets, website HTML, and validator scripts before upload; live URLs are checked after publishing |
| The public GitHub and Hugging Face bundles are publication-clean. | `scripts/validate_publication_package.py`, `docs/data/publication_audit.json` | Verified pass | Checks public files, HF bundles, and public-card freshness; ignored local scratch outputs are excluded |
| The public website has checked local references. | `scripts/validate_website_integrity.py`, `docs/data/website_integrity.json` | Verified pass | Checks local links, anchors, JSON data, and referenced images; external URLs are not fetched |
| The release gate is explicit and reviewable. | `QUALITY_GATES.md`, `scripts/build_quality_gates.py`, `docs/data/quality_gates.json` | Verified pass | Summarizes packaging and live-mirror checks; it does not prove cross-episode model quality |
| The live public mirrors are checked after upload. | `scripts/verify_live_publication.py`, `docs/data/live_publication_status.json` | Verified pass | Fetches public GitHub/HF URLs; it does not validate private training state |
| The core proof artifacts are indexed and grouped for fast review. | `ARTIFACT_GUIDE.md`, `scripts/build_artifact_index.py`, `docs/data/artifact_index.json` | Verified guide and index | Selective source-of-truth catalog, not a complete inventory of every output file |
| The public reproduction path is documented. | `REPRODUCIBILITY.md`, `docs/data/reproducibility_matrix.json`, `notes/reproducibility_audit.md` | Verified documentation and prior exact-match audit | Publicly reproduces the single-episode pipeline, not the gated 32-episode Qwen3-Omni pilot |
| The project is externally citable and machine-readable. | `CITATION.cff`, `codemeta.json`, `docs/data/project_manifest.json`, `LICENSE` | Verified metadata files | Code license does not override original Xperience-10M dataset terms |
| A first-time reviewer has an explicit audit path. | `docs/data/reviewer_packet.json`, website reviewer section, README reviewer path | Verified reviewer packet | It guides inspection; it does not add new experimental claims |

## Review Order

1. Read `docs/data/reviewer_packet.json` for the shortest audit path and proof
   boundary.
2. Read `XPERIENCE10M_DATASET_CARD_ALIGNMENT.md` and
   `docs/data/xperience10m_dataset_card_alignment.json` to check the official
   dataset-card wording and how the current repo is scoped against it.
3. Read `ARTIFACT_GUIDE.md` and `docs/data/artifact_index.json` to see grouped
   reviewer artifacts, indexed proof artifacts,
   sizes, and stable-file hashes.
4. Read `docs/assets/task_suite_infographic.png` and
   `docs/data/modality_atlas.json` for the high-level map and modality atlas.
5. Read `REPRODUCIBILITY.md` and `docs/data/reproducibility_matrix.json` before
   rerunning the public pipeline.
6. Inspect `results/episode_task_suite/summary_report.json` for the task and
   metric source of truth.
7. Inspect `results/episode_task_suite/feature_manifest.json` to see which
   modalities enter the current feature vector.
8. Inspect `results/episode_task_suite/neural_mlp/` to compare minimal and
   neural heads under the same splits.
9. Inspect `docs/data/scope_claims_audit.json` before interpreting historical
   `32ep` strings in Qwen3-Omni smoke artifacts.
10. Inspect `docs/data/mirror_parity.json` before assuming the GitHub and
   Hugging Face mirrors contain the same critical data, visual, HTML, and
   validator files.
11. Inspect `results/omni_finetune/DATA_BLOCKER_REPORT.md` before interpreting
   any Qwen3-Omni artifact.
12. Inspect `QUALITY_GATES.md`, `docs/data/quality_gates.json`,
   `docs/data/publication_audit.json`, and `docs/data/website_integrity.json`
   before publishing or sharing the project externally.
13. Inspect `CITATION.cff`, `codemeta.json`, and `LICENSE` before reusing or
   citing the project.

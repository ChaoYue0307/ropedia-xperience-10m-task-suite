# Evidence Contract

This project is organized as a research-development workspace. Every visible
project statement should point to a local artifact that a reader can inspect before using
the dashboard as a basis for further work.

| Project statement | Current evidence | Status | Current scope |
| --- | --- | --- | --- |
| A first-pass reader has a compact current-state summary. | `PROJECT_STATUS.md`, `docs/data/project_status.json` | Verified guide | Summarizes existing evidence and current limitations |
| The public dataset description is aligned with the official gated Xperience-10M dataset card and public sample card. | `XPERIENCE10M_DATASET_CARD_ALIGNMENT.md`, `docs/data/xperience10m_dataset_card_alignment.json` | Verified description alignment | Summarizes upstream public metadata, API listing facts, sample license/tooling, and card facts; does not grant access or mirror raw data |
| Source facts and current-project markers are validated across repo, website, and HF cards. | `SOURCE_ALIGNMENT_AUDIT.md`, `docs/data/source_alignment_audit.json`, `scripts/validate_source_alignment.py` | Verified source alignment | Offline committed-fact check; does not fetch private gated data |
| Public figures are indexed as project evidence. | `FIGURE_INDEX.md`, `docs/data/figure_index.json`, `scripts/build_figure_index.py` | Verified visual evidence | Derived figures and thumbnails only; does not include raw MP4/HDF5/RRD data |
| The project logo is consistently packaged across public surfaces. | `docs/data/brand_assets.json`, `docs/assets/brand/`, `scripts/build_brand_assets.py` | Verified brand packaging | Generated presentation assets only; does not contain raw Xperience-10M data or model weights |
| The public Xperience-10M sample has been converted into aligned model windows. | `results/episode_task_suite/windows.csv`, `results/episode_task_suite/shared_windows.npz`, `results/episode_task_suite/summary_report.json` | Verified for 5,821 frames and 1,161 windows | One public sample episode only |
| The current feature contract is explicit and inspectable. | `results/episode_task_suite/feature_manifest.json`, `results/episode_task_suite/available_modalities.json` | Verified for an 8,378-d feature vector | Audio is present in MP4 streams but not yet a feature block |
| The task evaluation protocol is explicit and generated from committed metrics. | `EVALUATION_PROTOCOL.md`, `docs/data/evaluation_protocol.json`, `scripts/build_evaluation_protocol.py` | Verified protocol | Defines windows, split, per-task metrics, leakage controls, and current limitations |
| The public sample modalities are inspectable without raw data redistribution. | `docs/data/modality_atlas.json`, `docs/assets/modalities/`, website modality atlas | Verified derived thumbnail atlas | Thumbnails are presentation assets, not a replacement for official raw data access |
| Public task cards stay readable for non-expert readers. | `docs/data/task_surface_integrity.json`, `scripts/validate_task_surface.py`, website task cards/player | Verified task-surface gate | Presentation integrity only; it does not add model quality or new data |
| The 12 task heads are real scripts and artifacts, not presentation placeholders. | `scripts/episode_task_suite.py`, `results/episode_task_suite/*/metrics.json`, `results/episode_task_suite/*/predictions.*` | Verified for all 12 task definitions | Chronological single-episode split, not cross-episode generalization |
| Minimal and neural heads use the same task contracts. | `scripts/neural_task_models.py`, `results/episode_task_suite/neural_mlp/`, `docs/assets/task_architectures.png` | Verified for 12 minimal heads and 12 neural MLP heads | Small heads only; not a foundation model |
| Four Ropedia research directions are mapped honestly as direct, proxy, or diagnostic evidence. | `results/episode_task_suite/research_directions/research_direction_taxonomy.json`, `docs/data/research_directions.json` | Verified taxonomy | Some directions remain proxy-only |
| Four extra direction probes are coded and evaluated. | `results/episode_task_suite/research_direction_extensions/research_direction_extension_results.json`, `docs/data/research_direction_extensions.json` | Verified single-episode probes | Not full human modeling, neural rendering, intent modeling, or world modeling solutions |
| Qwen3-Omni infrastructure has passed setup checks. | `results/omni_finetune/RUN_REPORT.md`, `results/omni_finetune/dataset_manifest.json`, `results/omni_finetune/metrics_eval.json` | Setup-stage evidence | One episode, 128 train windows; full metrics require the 32-episode pilot |
| The 32-episode LoRA pilot is waiting on gated data access. | `results/omni_finetune/DATA_BLOCKER_REPORT.md`, `results/omni_finetune/MULTI_EPISODE_ACCESS_STATUS.md`, `results/omni_finetune/source_discovery.json` | Data access pending | Held-out metrics come after the data gate, manifest construction, training, and test evaluation |
| Historical `32ep` path strings are tracked as setup-file provenance. | `scripts/validate_scope_claims.py`, `docs/data/scope_claims_audit.json` | Verified pass | Old run/path identifiers stay separate from completed 32-episode results |
| Prepared GitHub/Hugging Face mirrors carry matching critical files. | `scripts/validate_mirror_parity.py`, `docs/data/mirror_parity.json` | Verified pass | Compares prepared data files, visual assets, website HTML, and validator scripts before upload; live URLs are checked after publishing |
| The public GitHub and Hugging Face bundles are publication-clean. | `scripts/validate_publication_package.py`, `docs/data/publication_audit.json` | Verified pass | Checks public files, HF bundles, and public-card freshness; ignored local scratch outputs are excluded |
| The public repo, website, and Hugging Face cards present one coherent research surface. | `PUBLIC_SURFACE_QA.md`, `scripts/build_public_surface_qa.py`, `docs/data/public_surface_qa.json` | Verified public presentation | Checks SEO/social metadata, accessible tab semantics, public links, project-check links, and reader-facing copy consistency |
| The public website has checked local references. | `scripts/validate_website_integrity.py`, `docs/data/website_integrity.json` | Verified pass | Checks local links, anchors, JSON data, and referenced images; external URLs are not fetched |
| The release gate is explicit. | `QUALITY_GATES.md`, `scripts/build_quality_gates.py`, `docs/data/quality_gates.json` | Verified pass | Summarizes packaging and live-mirror checks; cross-episode model quality is measured by later held-out reports |
| The live public mirrors are checked after upload. | `scripts/verify_live_publication.py`, `docs/data/live_publication_status.json` | Verified pass | Fetches public GitHub/HF URLs; it does not validate private training state |
| The core project artifacts are indexed and grouped for fast reading. | `ARTIFACT_GUIDE.md`, `scripts/build_artifact_index.py`, `docs/data/artifact_index.json` | Verified guide and index | Selective source-of-truth catalog, not a complete inventory of every output file |
| The public reproduction path is documented. | `REPRODUCIBILITY.md`, `docs/data/reproducibility_matrix.json`, `notes/reproducibility_audit.md` | Verified documentation and prior exact-match check | Publicly reproduces the single-episode pipeline, not the gated 32-episode Qwen3-Omni pilot |
| The project is externally citable and machine-readable. | `CITATION.cff`, `codemeta.json`, `docs/data/project_manifest.json`, `LICENSE` | Verified metadata files | Code license does not override original Xperience-10M dataset terms |
| A first-time reader has an explicit project path. | `docs/data/project_packet.json`, website project path section, README project path | Verified project packet | Guides inspection across data, tasks, results, and scale-up status |

## Reading Order

1. Read `PROJECT_STATUS.md` and `docs/data/project_status.json` for
   the fastest current-state decision table.
2. Read `docs/data/project_packet.json` for the shortest project path and
   current scope.
3. Read `XPERIENCE10M_DATASET_CARD_ALIGNMENT.md` and
   `docs/data/xperience10m_dataset_card_alignment.json` to check the official
   dataset-card wording and how the current repo is scoped against it.
4. Read `SOURCE_ALIGNMENT_AUDIT.md` and
   `docs/data/source_alignment_audit.json` to verify the same source facts are
   present across repo, website, and HF cards.
5. Read `FIGURE_INDEX.md`, `docs/data/figure_index.json`, and
   `docs/data/brand_assets.json` to verify public figures, charts, modality
   thumbnails, logo assets, dimensions, hashes, and source scripts.
6. Read `EVALUATION_PROTOCOL.md` and `docs/data/evaluation_protocol.json` to
   check windowing, split policy, per-task metrics, leakage controls, and
   current limitations.
7. Read `ARTIFACT_GUIDE.md` and `docs/data/artifact_index.json` to see grouped
   project artifacts, indexed supporting artifacts,
   sizes, and stable-file hashes.
8. Read `docs/assets/task_suite_infographic.png` and
   `docs/data/modality_atlas.json` for the high-level map and modality atlas.
9. Read `REPRODUCIBILITY.md` and `docs/data/reproducibility_matrix.json` before
   rerunning the public pipeline.
10. Inspect `results/episode_task_suite/summary_report.json` for the task and
   metric source of truth.
11. Inspect `results/episode_task_suite/feature_manifest.json` to see which
   modalities enter the current feature vector.
12. Inspect `results/episode_task_suite/neural_mlp/` to compare minimal and
   neural heads under the same splits.
13. Inspect `docs/data/scope_claims_audit.json` before interpreting historical
   `32ep` strings in Qwen3-Omni setup artifacts.
14. Inspect `docs/data/mirror_parity.json` before assuming the GitHub and
   Hugging Face mirrors contain the same critical data, visual, HTML, and
   validator files.
15. Inspect `results/omni_finetune/DATA_BLOCKER_REPORT.md` and
   `results/omni_finetune/MULTI_EPISODE_ACCESS_STATUS.md` before interpreting
   any Qwen3-Omni artifact.
16. Inspect `QUALITY_GATES.md`, `docs/data/quality_gates.json`,
   `PUBLIC_SURFACE_QA.md`, `docs/data/public_surface_qa.json`,
   `docs/data/publication_audit.json`, and `docs/data/website_integrity.json`
   before publishing or sharing the project externally.
17. Inspect `CITATION.cff`, `codemeta.json`, and `LICENSE` before reusing or
   citing the project.

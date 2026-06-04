# Evidence Contract

This project is organized as a research-development workspace. Every visible
project statement should point to a local artifact that a reader can inspect before using
the dashboard as a basis for further work.

| Project statement | Current evidence | Status | Current scope |
| --- | --- | --- | --- |
| A first-pass reader has a compact current-state summary. | `PROJECT_STATUS.md`, `docs/data/project_status.json` | Verified guide | Summarizes existing evidence and current limitations |
| The research roadmap is explicit. | `RESEARCH_ROADMAP.md`, `docs/data/research_roadmap.json` | Current roadmap | Connects public-sample task development to multi-episode data preparation, Qwen3-Omni LoRA, robustness runs, and larger omni-model extensions |
| The public dataset description is aligned with the official gated Xperience-10M dataset card and public sample card. | `XPERIENCE10M_DATASET_CARD_ALIGNMENT.md`, `docs/data/xperience10m_dataset_card_alignment.json` | Verified description alignment | Summarizes upstream public metadata, API listing facts, sample license/tooling, and card facts; does not grant access or mirror raw data |
| Source facts, sample details, API-listing notes, and project coverage are aligned across repo, website, and HF cards. | `SOURCE_ALIGNMENT_AUDIT.md`, `docs/data/source_alignment_audit.json`, `scripts/validate_source_alignment.py` | Source alignment recorded | Offline committed-fact report; does not fetch private gated data |
| Public figures are indexed as project evidence. | `FIGURE_INDEX.md`, `docs/data/figure_index.json`, `scripts/build_figure_index.py` | Verified visual evidence | Derived figures and thumbnails only; does not include raw MP4/HDF5/RRD data |
| The project logo is consistently packaged across public surfaces. | `docs/data/brand_assets.json`, `docs/assets/brand/`, `scripts/build_brand_assets.py` | Verified brand packaging | Generated presentation assets only; does not contain raw Xperience-10M data or model weights |
| The public Xperience-10M sample has been converted into aligned model windows. | `results/episode_task_suite/windows.csv`, `results/episode_task_suite/shared_windows.npz`, `results/episode_task_suite/summary_report.json` | Verified for 5,821 frames and 1,161 windows | One public sample episode only |
| The current feature contract is explicit and inspectable. | `results/episode_task_suite/feature_manifest.json`, `results/episode_task_suite/available_modalities.json` | Verified for an 8,546-d feature vector | Synchronized video, audio, depth, pose/SLAM, motion, inertial, calibration, and language signals are represented |
| The task evaluation protocol is explicit and generated from committed metrics. | `EVALUATION_PROTOCOL.md`, `docs/data/evaluation_protocol.json`, `scripts/build_evaluation_protocol.py` | Verified protocol | Defines windows, split, per-task metrics, leakage controls, and current limitations |
| The public sample modalities are inspectable without raw data redistribution. | `docs/data/modality_atlas.json`, `docs/assets/modalities/`, website modality atlas | Verified derived thumbnail atlas | Thumbnails are presentation assets, not a replacement for official raw data access |
| Public task cards stay readable for non-expert readers. | `docs/data/task_surface_integrity.json`, `scripts/validate_task_surface.py`, website task cards/player | Task-surface report | Presentation layer only; it does not add model quality or new data |
| The 12 task heads are implemented as scripts with saved metrics and predictions. | `scripts/episode_task_suite.py`, `results/episode_task_suite/*/metrics.json`, `results/episode_task_suite/*/predictions.*` | Verified for all 12 task definitions | Chronological single-episode split, not cross-episode generalization |
| Minimal and neural heads use the same task contracts. | `scripts/neural_task_models.py`, `results/episode_task_suite/neural_mlp/`, `docs/assets/task_architectures.png` | Verified for 12 minimal heads and 12 neural MLP heads | Small heads only; not a foundation model |
| Four Ropedia research directions are mapped honestly as direct, proxy, or diagnostic evidence. | `results/episode_task_suite/research_directions/research_direction_taxonomy.json`, `docs/data/research_directions.json` | Verified taxonomy | Some directions remain proxy-only |
| Four extra direction probes are coded and evaluated. | `results/episode_task_suite/research_direction_extensions/research_direction_extension_results.json`, `docs/data/research_direction_extensions.json` | Verified single-episode probes | Not full human modeling, neural rendering, intent modeling, or world modeling solutions |
| Qwen3-Omni infrastructure has passed setup checks. | `results/omni_finetune/RUN_REPORT.md`, `results/omni_finetune/dataset_manifest.json`, `results/omni_finetune/metrics_eval.json` | Setup-stage evidence | One episode, 128 train windows; full metrics require completed multi-episode data preparation and held-out evaluation |
| The Qwen3-Omni LoRA pilot is in selected multi-episode preparation. | `results/omni_finetune/DATA_ACCESS_STATUS.md`, `results/omni_finetune/MULTI_EPISODE_ACCESS_STATUS.md`, `results/omni_finetune/source_discovery.json` | Data preparation | The gated Xperience-10M dataset is available; held-out metrics come after manifest construction, training, and test evaluation |
| Older pilot path strings are tracked as setup-file provenance. | `scripts/validate_scope_claims.py`, `docs/data/scope_claims_audit.json` | Multi-episode pilot status | Run/path identifiers stay separate from completed held-out-episode results |
| Prepared GitHub/Hugging Face mirrors carry matching critical files. | `scripts/validate_mirror_parity.py`, `docs/data/mirror_parity.json` | Mirror parity report | Compares prepared data files, visual assets, website HTML, and validator scripts before upload; live URLs are checked after publishing |
| The public GitHub and Hugging Face bundles are ready to share. | `scripts/validate_publication_package.py`, `docs/data/publication_audit.json` | Public bundle contents | Covers public files, HF bundles, and current public-card assets; temporary local outputs are excluded |
| The public repo, website, and Hugging Face cards present one cohesive research project. | `PUBLIC_SURFACE_QA.md`, `scripts/build_public_surface_qa.py`, `docs/data/public_surface_qa.json` | Public project surface | Covers SEO/social metadata, accessible tab semantics, public links, project links, and clear project presentation |
| The public website has validated local references. | `scripts/validate_website_integrity.py`, `docs/data/website_integrity.json` | Website reference report | Covers local links, anchors, JSON data, and referenced images; external URLs are not fetched |
| The rendered website walkthrough has a browser-level interaction check. | `RENDERED_SITE_CHECK.md`, `scripts/build_rendered_site_check.py`, `docs/data/rendered_site_check.json` | Rendered website check | Covers local page load, tab switch, walkthrough deep link, player controls, and console health |
| The release checks are explicit. | `QUALITY_GATES.md`, `scripts/build_quality_gates.py`, `docs/data/quality_gates.json` | Release checks | Summarizes packaging and live-mirror checks; cross-episode model quality is measured by later held-out reports |
| The live public mirrors are verified after upload. | `scripts/verify_live_publication.py`, `docs/data/live_publication_status.json` | Live publication report | Fetches public GitHub/HF URLs; it does not validate private training state |
| The core project artifacts are indexed and grouped for fast reading. | `ARTIFACT_GUIDE.md`, `scripts/build_artifact_index.py`, `docs/data/artifact_index.json` | Verified guide and index | Selective source-of-truth catalog, not a complete inventory of every output file |
| The public reproduction path is documented. | `REPRODUCIBILITY.md`, `docs/data/reproducibility_matrix.json`, `notes/reproducibility_audit.md` | Verified documentation and prior exact-match check | Publicly reproduces the single-episode pipeline; multi-episode Qwen3-Omni metrics are added only after staging and held-out evaluation |
| The project is externally citable and machine-readable. | `CITATION.cff`, `codemeta.json`, `docs/data/project_manifest.json`, `LICENSE` | Verified metadata files | Code license does not override original Xperience-10M dataset terms |
| A first-time reader has an explicit project path. | `docs/data/project_packet.json`, website project path section, README project path | Verified project packet | Guides inspection across data, tasks, results, and scale-up status |

## Reading Order

1. Read `PROJECT_STATUS.md` and `docs/data/project_status.json` for
   the fastest current-state decision table.
2. Read `RESEARCH_ROADMAP.md` and `docs/data/research_roadmap.json` for the
   staged path from public-sample development to multi-episode modeling.
3. Read `docs/data/project_packet.json` for the shortest project path and
   current scope.
4. Read `XPERIENCE10M_DATASET_CARD_ALIGNMENT.md` and
   `docs/data/xperience10m_dataset_card_alignment.json` to check the official
   dataset-card wording and how the current repo is scoped against it.
5. Read `SOURCE_ALIGNMENT_AUDIT.md` and
   `docs/data/source_alignment_audit.json` to inspect the same source facts
   present across repo, website, and HF cards.
6. Read `FIGURE_INDEX.md`, `docs/data/figure_index.json`, and
   `docs/data/brand_assets.json` to inspect public figures, charts, modality
   thumbnails, logo assets, dimensions, hashes, and source scripts.
7. Read `EVALUATION_PROTOCOL.md` and `docs/data/evaluation_protocol.json` to
   check windowing, split policy, per-task metrics, leakage controls, and
   current limitations.
8. Read `ARTIFACT_GUIDE.md` and `docs/data/artifact_index.json` to see grouped
   project artifacts, indexed supporting artifacts,
   sizes, and stable-file hashes.
9. Read `docs/assets/task_suite_infographic.png` and
   `docs/data/modality_atlas.json` for the high-level map and modality atlas.
10. Read `REPRODUCIBILITY.md` and `docs/data/reproducibility_matrix.json` before
   rerunning the public pipeline.
11. Inspect `results/episode_task_suite/summary_report.json` for the task and
   metric source of truth.
12. Inspect `results/episode_task_suite/feature_manifest.json` to see which
   modalities enter the current feature vector.
13. Inspect `results/episode_task_suite/neural_mlp/` to compare minimal and
   neural heads under the same splits.
14. Inspect `docs/data/scope_claims_audit.json` before interpreting older
   Qwen3-Omni setup artifacts.
15. Inspect `docs/data/mirror_parity.json` before assuming the GitHub and
   Hugging Face mirrors contain the same critical data, visual, HTML, and
   validator files.
16. Inspect `results/omni_finetune/DATA_ACCESS_STATUS.md` and
   `results/omni_finetune/MULTI_EPISODE_ACCESS_STATUS.md` before interpreting
   any Qwen3-Omni artifact.
17. Inspect `QUALITY_GATES.md`, `docs/data/quality_gates.json`,
   `PUBLIC_SURFACE_QA.md`, `docs/data/public_surface_qa.json`,
   `docs/data/publication_audit.json`, and `docs/data/website_integrity.json`
   before sharing a new public release.
18. Inspect `CITATION.cff`, `codemeta.json`, and `LICENSE` before reusing or
   citing the project.

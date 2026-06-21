# Glossary

This glossary defines project terms that can be easy to confuse across the
GitHub repo, website, Hugging Face Space, artifact dataset, model repos, and
result matrices. Use it with `PUBLIC_READER_MAP.md` when choosing what to read
first, and with `docs/data/glossary.json` when a tool needs the same terms in
machine-readable form.

## How To Read The Terms

| Category | What it clarifies |
| --- | --- |
| Dataset and scope | Which data is public, which data is gated upstream, and what each evidence line can support. |
| Files and features | How raw sample files, derived windows, feature manifests, and public-safe artifacts relate to each other. |
| Tasks and metrics | What a scored task row means, when a score is direct, and when a compact proxy is being used. |
| Models and runs | How simple/NN baselines, Qwen3-Omni, Cosmos3, LoRA adapters, and full-parameter gates differ. |
| Public surfaces | Which repo or Hub surface owns which part of the public package. |

## Core Terms

| Term | Plain meaning | In this project | Do not confuse with |
| --- | --- | --- | --- |
| Xperience-10M | The upstream embodied human-interaction dataset. | The source dataset behind the public sample, selected-128 features, task suite, and model diagnostics. | This repo itself; the repo only redistributes public-safe derived artifacts. |
| Public sample episode | One officially available sample episode. | The fully inspectable Line 1 unit used for raw-file browsing, 20-frame windows, task construction, and single-episode baselines. | Multi-episode generalization. |
| Selected 128 episodes | A public-safe selected subset of official gated episode paths. | Line 2 uses derived windows/features and keeps links back to official episode ids and gated source paths. | Redistributed raw MP4/HDF5/RRD data. |
| Evidence line | A claim boundary for a group of results. | Line 1 is one public sample episode; Line 2 is selected-128 held-out comparison. | Qwen run versions v1-v6, which are model-run lineage, not evidence lines. |
| Official gated data | Upstream files that require official dataset access. | Raw Xperience-10M MP4/HDF5/RRD files and full source directories remain outside the public repo. | Public-safe metrics, derived features, figures, and manifests. |
| Public-safe artifact | A file that can be mirrored publicly without raw gated content. | Metrics, JSON summaries, model cards, figures, derived manifests, and approved lightweight weights/adapters. | Raw dataset redistribution. |
| Episode | One recorded interaction sequence. | The basic source unit behind windows, labels, and train/val/test splits. | A 20-frame window, which is a smaller model input slice. |
| 20-frame window | A fixed short clip slice. | The sample episode is converted into aligned 20-frame units for features, labels, and many task heads. | A full episode or an arbitrary video segment. |
| Window stride | The frame step between neighboring windows. | Used to create overlapping examples while preserving chronological order and leakage controls. | Video frame rate. |
| Feature manifest | A map from model-input columns to source modalities. | `results/episode_task_suite/feature_manifest.json` explains the feature groups and dimensions. | The raw annotation file. |
| Raw sample file map | A human-readable inventory of the sample episode files. | `docs/data/raw_sample_files.json` explains videos, annotations, calibration, motion, and derived previews. | A training manifest. |
| annotation.hdf5 | Upstream annotation container for the sample. | Contains original labels/metadata; some public derived files expose hashed or processed features rather than every raw text field. | `summary_report.json` or task result JSON. |
| Interaction text | Natural-language interaction/caption content. | Used by task 15 and some derived text features; public matrices record when text targets are direct or compact-proxy. | Numeric action ids or subtask ids. |
| Modality | A type of signal. | Video, audio, depth, pose/SLAM, motion capture, inertial, calibration, and language-derived signals. | A task target. |
| Task contract | The definition of one benchmark task. | Includes input, target/output, metric, split, source artifact, and limitation. | A model architecture. |
| Unified 20-task suite | The current task surface. | All 20 task contracts are presented together and scored across methods where real artifacts exist. | Historical `tier2_task_suite` filenames; those are provenance paths, not a second suite. |
| Task-method record | One method evaluated on one task. | 9 methods x 20 tasks gives 180 public result records. | A single prediction row. |
| Direct score | A metric computed against the task target directly. | The preferred score type in the 20-task matrix. | Compact-proxy score. |
| Compact-proxy score | A bounded proxy metric when a direct raw target is not publicly available. | Kept explicit in the matrix and gap audit so readers do not over-read it. | A direct target measurement. |
| Raw metric value | The original metric value emitted by the runner or verified result package. | This is the value to cite from the 180-result table. | The normalized radar value. |
| Normalized radar value | A 0-1 plotting value used only to draw comparable radar polygons. | Helps visualize metrics with different scales and directions. | The raw metric value to cite. |
| Gap audit | A coverage and source-status audit. | `docs/data/task_method_20_gap_audit.json` explains scored, proxy, and unsupported cells. | A performance leaderboard. |
| Leakage control | A split or feature rule that prevents using future/target information unfairly. | Chronological splits, held-out splits, and source audits protect task interpretation. | Lower training accuracy. |
| Minimal baseline | A simple non-neural task head; the "minimum" reference row in casual wording. | Provides a reproducible lower-complexity comparison for task feasibility. | The metadata-only baseline family in the selected-128 matrix. |
| Simple baseline | A non-neural baseline family for the selected-128 rows. | Used for metadata/text and raw-feature 128-episode comparisons before NN/foundation-model rows. | The single-episode Minimal baseline. |
| Neural MLP | A compact neural task head. | Used for single-episode and selected-128 baseline comparisons. | Foundation-model fine-tuning. |
| Metadata baseline | A selected-128 baseline using metadata/text-derived public-safe features. | Helps compare simple and neural heads on the held-out split. | Raw video/depth/audio feature baselines. |
| Raw-feature baseline | A selected-128 baseline using exported public-safe raw-feature groups. | Tracks what non-foundation heads can do with richer processed inputs. | Raw gated media redistribution. |
| Qwen3-Omni | The multimodal foundation-model family used for the Qwen branch. | The current public 20-task Qwen row is Qwen3-Omni v6 LoRA plus task-specific probes. | Cosmos3 or the single-episode task-head baselines. |
| Qwen v1-v6 | The Qwen3-Omni run lineage. | v1-v4 are earlier pipeline/ablation evidence, v5 is the prior pinned release, and v6 is the current public 20-task row. | Six different evidence lines. |
| Cosmos3-Super | The larger Cosmos3-style branch tracked in this project. | Published as Reasoner diagnostics and a separate forward-dynamics LoRA adapter/result branch when verified. | Cosmos3-Nano. |
| Cosmos3-Nano | A smaller Cosmos3 compatibility/future-window branch. | Used for the Nano Future Window row and related diagnostics. | Cosmos3-Super fine-tuned adapter. |
| LoRA adapter | A lightweight set of trainable adapter weights. | Published only when the package is verified and public-safe. | Full base-model weights. |
| Full-parameter fine-tuning | Updating the whole model rather than only adapters. | This project records feasibility gates and short pilots, but does not publish full checkpoints. | LoRA adapter publication. |
| Foundation pipeline | A high-level training direction. | Spatial intelligence, human-video world modeling, and vision-language-action are documented as trainable directions with task mappings. | A completed public result row. |
| Spatial intelligence | Learning geometry and spatial reasoning from egocentric data. | Uses video, depth, camera pose, and language tasks to target 3D/space reasoning. | World-model future prediction. |
| Human-video world model | Learning future frames, actions, and interaction dynamics from human video. | Uses temporal prediction, next-action, transition, and object-forecast tasks. | Robot policy execution. |
| Vision-language-action | Mapping perception and language to action chunks. | A future policy/VLA direction that needs action-target conversion and stronger policy packaging. | Qwen3-Omni diagnostic scoring. |
| HF Space | Hugging Face-hosted app/site surface. | Mirrors the dashboard and static website assets. | HF artifact dataset or model repo. |
| HF artifact dataset | Hugging Face dataset repo for derived evidence. | Stores public-safe reports, metrics, website JSON, and sanitized result packages. | Original Xperience-10M dataset. |
| HF baseline model repo | Hugging Face model repo for lightweight baseline artifacts. | Mirrors baseline weights, figures, metrics, and task artifacts. | Qwen/Cosmos adapter-specific repos. |
| HF weights/results repo | Consolidated public-safe model-result bundle. | Groups baseline weights, verified Qwen/Cosmos artifacts, analysis files, and manifests. | The upstream raw dataset. |
| Mirror parity | A check that public copies match the source files. | `docs/data/mirror_parity.json` records whether GitHub, website, and HF mirrors agree. | A model-quality metric. |
| Publication audit | A public-package validation report. | Confirms required files exist and forbidden raw/private assets are not included. | Scientific peer review. |
| Verified package | A result or artifact bundle that passed local/public validators. | Only verified packages are promoted to README, website, and HF surfaces as public evidence. | A running or exploratory experiment. |

## File Entry Points

| Need | Open |
| --- | --- |
| Reader navigation | `PUBLIC_READER_MAP.md`, `docs/data/public_reader_map.json` |
| Task definitions | `TASK_SUITE_20.md`, `docs/data/task_suite_20.json` |
| Result matrix | `TASK_METHOD_20_RESULT_MATRIX.md`, `docs/data/task_method_20_result_matrix.json` |
| Direct/proxy status | `TASK_METHOD_20_GAP_AUDIT.md`, `docs/data/task_method_20_gap_audit.json` |
| Qwen lineage | `QWEN3_OMNI_RUN_LINEAGE.md`, `docs/data/qwen3_omni_run_lineage.json` |
| 128-episode source/features | `XPERIENCE10M_128_EPISODE_FEATURE_INDEX.md`, `docs/data/xperience10m_128_episode_feature_index.json` |
| Public mirrors | `PUBLIC_SURFACE_QA.md`, `docs/data/mirror_parity.json`, `docs/data/live_publication_status.json` |

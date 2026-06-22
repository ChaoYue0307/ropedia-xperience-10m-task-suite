# Glossary

This glossary defines project-specific terms and adjacent technical field terms that can be easy to confuse across the GitHub repo, website, Hugging Face Space, artifact dataset, model repos, result matrices, and embodied-AI training discussions. Use it with `PUBLIC_READER_MAP.md` when choosing what to read first, and with `docs/data/glossary.json` when a tool needs the same terms in machine-readable form.

## How To Read The Terms

| Category | What it clarifies |
| --- | --- |
| Dataset and scope | Public data boundaries, evidence lines, and how each result family should be read. |
| Files and features | Raw sample files, windows, feature manifests, and public-safe derivatives. |
| Multimodal sensing | Video, audio, depth, IMU, motion capture, calibration, and synchronization terms. |
| Spatial geometry | Camera pose, SLAM, coordinate frames, point clouds, 3D reconstruction, and spatial grounding. |
| Temporal and world models | Future prediction, rollouts, forward dynamics, long-horizon forecasting, and temporal leakage. |
| Robotics and VLA | Vision-language-action, policies, action chunks, imitation learning, contact, and dexterity. |
| Tasks and metrics | Task contracts, scored records, direct scores, compact proxies, and audits. |
| Training and evaluation | Splits, held-out evaluation, metric types, prompt/schema checks, adapters, and distributed training. |
| Models and runs | Baseline families, Qwen3-Omni, Cosmos3, LoRA adapters, and full-parameter gates. |
| Public surfaces | GitHub, website, Hugging Face repos, parity checks, and package validation. |

## Core And Field Terms

### Dataset and scope

| Term | Plain meaning | In this project | Do not confuse with |
| --- | --- | --- | --- |
| Evidence line | A reading lane for a group of results. | Line 1 is one public sample episode; Line 2 is selected-128 held-out comparison. | Qwen run versions v1-v6, which are model-run lineage. |
| Official gated data | Upstream files that require official dataset access. | Raw Xperience-10M MP4/HDF5/RRD files and full source directories remain outside the public repo. | Public-safe metrics, derived features, figures, and manifests. |
| Public sample episode | One officially available sample episode. | The fully inspectable Line 1 unit used for raw-file browsing, 20-frame windows, task construction, and single-episode baselines. | The selected-128 comparison rows. |
| Selected 128 episodes | A public-safe selected subset of official gated episode paths. | Line 2 uses derived windows/features and keeps links back to official episode ids and gated source paths. | Redistributed raw MP4/HDF5/RRD data. |
| Xperience-10M | The upstream embodied human-interaction dataset. | Source dataset behind the public sample, selected-128 features, task suite, and model diagnostics. | This repo, which only redistributes public-safe derived artifacts. |

### Files and features

| Term | Plain meaning | In this project | Do not confuse with |
| --- | --- | --- | --- |
| 20-frame window | A fixed short clip slice. | The sample episode is converted into aligned 20-frame units for features, labels, and many task heads. | A full episode or arbitrary video segment. |
| annotation.hdf5 | Upstream annotation container for the sample. | Contains original labels/metadata; some public derived files expose processed features instead of every raw text field. | Task result summaries. |
| Episode | One recorded interaction sequence. | The basic source unit behind windows, labels, and train/val/test splits. | A 20-frame window. |
| Feature manifest | A map from model-input columns to source modalities. | Explains feature groups and dimensions for the sample task suite. | The raw annotation file. |
| Interaction text | Natural-language interaction/caption content. | Used by task 15 and some derived text features; public matrices record direct or compact-proxy status. | Numeric action ids or subtask ids. |
| Modality | A type of signal. | Video, audio, depth, pose/SLAM, motion capture, inertial, calibration, and language-derived signals. | A task target. |
| Raw sample file map | A human-readable inventory of the sample episode files. | Explains videos, annotations, calibration, motion, and derived previews. | A training manifest. |
| visualization.rrd | Rerun viewer recording for visual inspection. | Can be downloaded from the official sample dataset and opened in Rerun 0.29.0 to inspect the sample episode. It is not used for published training or metric rows. | MP4 video streams or model inputs. |
| Window stride | The frame step between neighboring windows. | Creates overlapping examples while preserving chronological order and leakage controls. | Video frame rate. |

### Multimodal sensing

| Term | Plain meaning | In this project | Do not confuse with |
| --- | --- | --- | --- |
| Audio waveform | A time-series pressure signal from sound. | The audio ablation measures whether embedded audio helps selected task contracts. | Language captions or text labels. |
| Calibration | Parameters that relate sensors to each other and to physical space. | Needed to interpret camera streams, depth, pose, and synchronized multimodal features together. | A model training hyperparameter. |
| Camera extrinsics | A camera position and orientation relative to another coordinate frame. | Connects different camera streams and world coordinates. | Camera intrinsics. |
| Camera intrinsics | Internal camera parameters such as focal length and distortion. | Explain how image pixels project to rays for geometry tasks. | Camera extrinsics. |
| Depth map | A per-pixel estimate of distance from the camera. | Depth-derived signals support spatial and geometry-oriented tasks. | RGB brightness or semantic segmentation. |
| Egocentric video | Video captured from a first-person or body-mounted viewpoint. | The sample streams are egocentric views of human interaction and are the visual basis for many tasks. | Third-person robot-camera footage. |
| Fisheye camera | A wide-angle camera with strong lens distortion. | Multiple fisheye MP4 streams give broad room coverage but need calibration-aware interpretation. | A rectilinear pinhole camera image. |
| IMU | An inertial measurement unit with accelerometer and gyroscope signals. | Supports motion, temporal, and sensor-bridging tasks. | Motion capture skeleton data. |
| Metric depth | Depth expressed in physical units rather than arbitrary relative scale. | Useful for distance-sensitive spatial reasoning and reconstruction targets. | Relative monocular depth. |
| Motion capture | A system that records body or hand motion over time. | Provides hand/body motion evidence when exposed through public-safe derived features. | Video-only pose estimation. |
| RGB frame | A color image frame from a video stream. | Used for visual statistics, previews, and many model inputs. | Depth values or point-cloud coordinates. |
| Sensor alignment | Putting different sensor streams into a shared temporal or spatial reference. | Used to make video, audio, pose, depth, IMU, and mocap usable in the same task input. | Model ensembling. |
| Stereo camera | A paired-camera setup that supports depth or geometry estimation. | The sample browser exposes stereo streams as part of the visual modality set. | Single-view RGB video. |
| Timestamp synchronization | Aligning sensor samples by time. | The task suite assumes aligned windows across modalities so labels and features refer to the same moment. | Randomly joining files with similar names. |

### Spatial geometry

| Term | Plain meaning | In this project | Do not confuse with |
| --- | --- | --- | --- |
| 3D reconstruction | Recovering 3D scene structure from sensor data. | One core spatial-intelligence direction for Xperience-style data. | Next-action classification. |
| Affordance | An action possibility offered by an object or scene. | Relevant when moving from observed human interaction to robot-action or VLA tasks. | A detected object category alone. |
| Camera pose | The camera position and orientation at a time step. | Supports spatial-intelligence tasks, view synchronization, and geometry diagnostics. | The human body pose. |
| Coordinate frame | A reference system for positions and orientations. | Needed when comparing camera, body, object, and world measurements. | A video frame. |
| Object-centric representation | A representation organized around objects and their relations. | Useful for object relevance, object-set forecast, and action-object relation tasks. | A flat feature vector without object identity. |
| Odometry | Motion estimated from sensor changes over time. | A relevant spatial term for ego-motion and camera-pose reasoning. | Ground-truth motion capture. |
| Point cloud | A set of 3D points representing scene structure. | A likely target or intermediate representation for spatial-intelligence extensions. | A 2D image grid. |
| SLAM | Simultaneous localization and mapping. | A field term for estimating camera motion and scene structure from sensor observations. | A task label or action class. |
| Spatial grounding | Linking language or labels to locations, objects, or geometry. | Connects language grounding tasks with 3D/spatial reasoning. | General text classification. |
| Trajectory | A sequence of positions over time. | Used for hand motion, camera motion, and future-path tasks. | A single coordinate or label. |

### Temporal and world models

| Term | Plain meaning | In this project | Do not confuse with |
| --- | --- | --- | --- |
| Action forecasting | Predicting a future action before it happens. | Covered by next-action and long-horizon task contracts. | Recognizing the current action only. |
| Autoregressive prediction | Generating each future token, state, or frame conditioned on prior outputs. | Relevant for model branches that produce structured JSON or temporal predictions. | A one-shot classifier. |
| Forward dynamics | Predicting the next state from the current state and action/context. | The Cosmos3-Super LoRA branch uses a forward-dynamics-style diagnostic contract. | Reverse inference from result back to cause. |
| Latent state | A hidden representation that summarizes observed context. | Useful for future foundation-model and world-model training plans. | A visible annotation column. |
| Long-horizon prediction | Predicting outcomes several seconds or steps ahead. | Tasks 13 and 14 test longer temporal context beyond immediate recognition. | Single-frame classification. |
| Next-frame prediction | Predicting future visual frames from past frames. | A field-level world-model objective related to the human-video world-model direction. | Next-action prediction. |
| Object persistence | Tracking that an object remains present over time even when view or interaction changes. | Relevant for object-set forecast and long-video reasoning. | A single-frame object detection. |
| Rollout | Repeatedly predicting future steps from a model state. | Important for judging world models beyond one-step prediction. | A held-out static test row. |
| Subtask forecasting | Predicting the next higher-level step in an activity. | Used in the future-task probe line for Qwen3-Omni. | Frame-level action classification. |
| Teacher forcing | Training a sequence model using ground-truth previous outputs. | A likely training option for future sequence/world-model baselines. | Free-running rollout evaluation. |
| Temporal leakage | Using future information that would not be available at prediction time. | Avoided by chronological splits and target-side feature controls. | A low model score. |
| Transition timing | Estimating when the next state or action transition happens. | Task 20 turns temporal change into a regression target. | Classifying the transition type only. |

### Robotics and VLA

| Term | Plain meaning | In this project | Do not confuse with |
| --- | --- | --- | --- |
| Action chunk | A short sequence of low-level actions predicted together. | The VLA figure and plan use action chunks as the policy-output concept. | A natural-language action label. |
| Behavior cloning | A supervised imitation-learning method for predicting demonstrated actions. | A plausible baseline once action targets are converted. | Generative video modeling. |
| Contact event | A moment when a hand, body, or tool touches an object or surface. | Used in contact-related tasks and action-quality interpretation. | Visual co-occurrence without touch. |
| Dexterity | Fine-grained physical manipulation ability. | Relevant to hand-object interaction, contact, and VLA/policy directions. | High text-generation accuracy. |
| End effector | The robot part that acts on the world, such as a gripper or hand. | A key target frame for future manipulation-policy conversion. | A camera or global scene coordinate. |
| Hand-object interaction | A physical interaction between hands and objects. | A central signal family behind action, contact, object relevance, and interaction-text tasks. | Object detection without action. |
| Imitation learning | Training a policy to imitate demonstrated behavior. | Relevant when converting human video/motion into action supervision. | Reinforcement learning from online robot trials. |
| Language grounding | Connecting text to observed objects, actions, or spatial context. | Task 8 and VLA directions use language as grounded supervision rather than standalone text. | Caption fluency alone. |
| Policy | A mapping from observations to actions. | A future target for robot-compatible Xperience-derived action data. | A benchmark metric. |
| Robot-compatible action target | An action representation a robot policy can execute or imitate. | Needed before OpenVLA/openpi/GR00T-style policy training is meaningful here. | Human-only caption text. |
| Vision-language-action model | A model that maps visual context and language into actions. | The VLA direction is a future path after action targets are converted into robot-compatible chunks. | A vision-language model that only answers text. |

### Tasks and metrics

| Term | Plain meaning | In this project | Do not confuse with |
| --- | --- | --- | --- |
| Compact-proxy score | A bounded proxy metric when a direct raw target is not publicly available. | Kept explicit in the matrix and gap audit so readers do not over-read it. | A direct target measurement. |
| Direct score | A metric computed against the task target directly. | The preferred score type in the 20-task matrix. | Compact-proxy score. |
| Gap audit | A coverage and source-status audit. | Explains scored, proxy, and unsupported cells. | A performance leaderboard. |
| Leakage control | A split or feature rule that prevents using target information unfairly. | Chronological splits, held-out splits, and source audits protect task interpretation. | Lower training accuracy. |
| Normalized radar value | A 0-1 plotting value used only to draw comparable radar polygons. | Helps visualize metrics with different scales and directions. | The raw metric value to cite. |
| Raw metric value | The original metric value emitted by the runner or verified result package. | This is the value to cite from the 180-result table. | The normalized radar value. |
| Task contract | The definition of one benchmark task. | Includes input, target/output, metric, split, source artifact, and limitation. | A model architecture. |
| Task-method record | One method evaluated on one task. | 9 methods x 20 tasks gives 180 public result records. | A single prediction row. |
| Unified 20-task suite | The current task surface. | All 20 task contracts are presented together and scored across methods where real artifacts exist. | Historical tier2_task_suite filenames, which are provenance paths rather than a second suite. |

### Training and evaluation

| Term | Plain meaning | In this project | Do not confuse with |
| --- | --- | --- | --- |
| Adapter checkpoint | Saved adapter weights from a fine-tuning run. | The public model branches publish adapters when validated and public-safe. | Full base-model checkpoint. |
| Balanced accuracy | Accuracy averaged across classes to reduce majority-class dominance. | Useful for imbalanced task labels. | Overall accuracy. |
| Chronological split | A split ordered by time. | Used for the single-episode baselines to reduce future-window leakage. | A random row split. |
| Confusion matrix | A table of predicted classes versus true classes. | Helps inspect which task labels a method confuses. | A scalar leaderboard score. |
| FSDP | Fully Sharded Data Parallel, a distributed training strategy. | Appears in full-parameter feasibility and multi-GPU training notes. | A model architecture. |
| Held-out evaluation | Testing on examples not used for training. | Required before promoting Qwen/Cosmos results to public evidence. | Training-set loss. |
| JSON validity | Whether model output parses as the required JSON schema. | A key diagnostic for Qwen3-Omni structured-output runs. | Task correctness after parsing. |
| Macro F1 | The average F1 score across classes, usually treating classes equally. | Used when class imbalance matters in classification tasks. | Accuracy dominated by frequent classes. |
| Mean absolute error | The average absolute difference between predicted and true numeric values. | Used for regression-style task rows such as timing or trajectory targets. | A classification F1 score. |
| Overfit check | A small training test that verifies a model can learn a tiny subset. | Useful for catching data/model wiring bugs before full training. | Evidence of generalization. |
| Parameter-efficient fine-tuning | Updating a small number of added or selected parameters. | LoRA is the current parameter-efficient path for Qwen/Cosmos branches. | Full-parameter fine-tuning. |
| Schema compliance | Whether an output follows the expected field names and value types. | Needed for structured task probes and public package validation. | High semantic accuracy. |
| Smoke run | A short run that checks whether a pipeline can start and execute key steps. | Used for feasibility gates before expensive full runs. | A complete benchmark result. |
| Top-k accuracy | A score that counts a prediction correct if the target is among the k highest-ranked outputs. | Useful for large-label or retrieval-style tasks. | Top-1 exact accuracy. |
| Train/validation/test split | A partition that separates model fitting, tuning, and final evaluation examples. | The selected-128 setup uses a held-out split discipline for model branches. | A random shuffle without temporal or episode boundaries. |

### Models and runs

| Term | Plain meaning | In this project | Do not confuse with |
| --- | --- | --- | --- |
| Cosmos3-Nano | A smaller Cosmos3 compatibility/future-window branch. | Used for the Nano Future Window row and related diagnostics. | Cosmos3-Super fine-tuned adapter. |
| Cosmos3-Super | The larger Cosmos3-style branch tracked in this project. | Published as Reasoner diagnostics and a separate forward-dynamics LoRA adapter/result branch when verified. | Cosmos3-Nano. |
| Foundation pipeline | A high-level training direction. | Spatial intelligence, human-video world modeling, and vision-language-action are documented as trainable directions with task mappings. | A completed public result row. |
| Full-parameter fine-tuning | Updating the whole model rather than only adapters. | This project records feasibility gates and short pilots, but does not publish full checkpoints. | LoRA adapter publication. |
| Human-video world model | Learning future frames, actions, and interaction dynamics from human video. | Uses temporal prediction, next-action, transition, and object-forecast tasks. | Robot policy execution. |
| LoRA adapter | A lightweight set of trainable adapter weights. | Published only when the package is verified and public-safe. | Full base-model weights. |
| Metadata baseline | A selected-128 baseline using metadata or text-derived public-safe features. | Compares simple and neural heads on the held-out split. | Raw video, depth, or audio feature baselines. |
| Minimal baseline | A simple non-neural task head; the "minimum" reference row in casual wording. | Provides a reproducible lower-complexity comparison for task feasibility. | Metadata-only selected-128 baseline family. |
| Neural MLP | A compact neural task head. | Used for single-episode and selected-128 baseline comparisons. | Foundation-model fine-tuning. |
| Qwen v1-v6 | The Qwen3-Omni run lineage. | v1-v4 are earlier pipeline/ablation evidence, v5 is the prior pinned release, and v6 is the current public 20-task row. | Six different evidence lines. |
| Qwen3-Omni | The multimodal foundation-model family used for the Qwen branch. | The current public 20-task Qwen row is Qwen3-Omni v6 LoRA plus task-specific probes. | Cosmos3 or single-episode task-head baselines. |
| Raw-feature baseline | A selected-128 baseline using exported public-safe raw-feature groups. | Tracks what non-foundation heads can do with richer processed inputs. | Raw gated media redistribution. |
| Simple baseline | A non-neural baseline family for the selected-128 rows. | Used for metadata/text and raw-feature 128-episode comparisons before NN/foundation-model rows. | The single-episode Minimal baseline. |
| Spatial intelligence | Learning geometry and spatial reasoning from egocentric data. | Uses video, depth, camera pose, and language tasks to target 3D/space reasoning. | World-model future prediction. |
| Vision-language-action | Mapping perception and language to action chunks. | A future policy/VLA direction that needs action-target conversion and stronger policy packaging. | Qwen3-Omni diagnostic scoring. |

### Public surfaces

| Term | Plain meaning | In this project | Do not confuse with |
| --- | --- | --- | --- |
| HF artifact dataset | Hugging Face dataset repo for derived evidence. | Stores public-safe reports, metrics, website JSON, and sanitized result packages. | Original Xperience-10M dataset. |
| HF baseline model repo | Hugging Face model repo for lightweight baseline artifacts. | Mirrors baseline weights, figures, metrics, and task artifacts. | Qwen/Cosmos adapter-specific repos. |
| HF Space | Hugging Face-hosted app/site surface. | Mirrors the dashboard and static website assets. | HF artifact dataset or model repo. |
| HF weights/results repo | A consolidated public-safe model-result bundle. | Groups baseline weights, verified model artifacts, analysis files, and manifests. | The upstream raw dataset. |
| Mirror parity | A check that public copies match the source files. | Records whether GitHub, website, and HF mirrors agree. | A model-quality metric. |
| Public-safe artifact | A file that can be mirrored publicly without raw gated content. | Metrics, JSON summaries, model cards, figures, derived manifests, and approved lightweight weights/adapters. | Raw dataset redistribution. |
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

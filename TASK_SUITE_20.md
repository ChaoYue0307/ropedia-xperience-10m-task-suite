# Unified 20-Task Suite

The public Xperience-10M sample task surface is one unified set of 20 tasks.
All task contracts are presented together under the same window, split,
feature, baseline, and leakage-control contract.

Historical artifact paths containing `tier2_task_suite` are kept for stable
links, but they should be read as provenance directories inside the unified task suite, not
as a separate benchmark tier.

## Shared Setup

- Episode scope: `1` public sample episode.
- Frames/windows: `5,821` frames and `1,161` aligned windows.
- Windowing: `20` frames per window, stride `5` frames.
- Feature vector: `8,546` dimensions from the shared feature manifest.
- Split: chronological 70/30 train/test by time within the sample episode.
- Baselines: minimal interpretable heads and compact neural MLP heads.
- Raw data: MP4/HDF5/RRD files are not redistributed.

## Task Table

| # | Task | Artifact id | Origin | Input -> output | Primary metric | Minimal | Neural |
| ---: | --- | --- | --- | --- | --- | ---: | ---: |
| 1 | Action Recognition | `timeline_action` | original task | 20-frame multimodal window -> current action class | macro-F1 (higher better) | 0.0500 | 0.0148 |
| 2 | Procedure Step Recognition | `timeline_subtask` | original task | 20-frame multimodal window -> current procedure step | macro-F1 (higher better) | 0.0506 | 0.0281 |
| 3 | Action Boundary Detection | `transition_detection` | original task | current window with boundary target -> boundary or steady | macro-F1 (higher better) | 0.6118 | 0.5862 |
| 4 | Next-Action Prediction | `next_action` | original task | current window at time t -> action at t+20 frames | macro-F1 (higher better) | 0.0593 | 0.0419 |
| 5 | Hand Trajectory Forecasting | `hand_trajectory_forecast` | original task | current multimodal window -> future hand-joint trajectory | MPJPE (lower better) | 0.8647 | 0.1079 |
| 6 | Contact State Prediction | `contact_prediction` | original task | non-contact, non-caption features -> contact or no contact | macro-F1 (higher better) | 1.0000 | 1.0000 |
| 7 | Object Relevance Prediction | `object_relevance` | original task | non-caption multimodal features -> relevant object set | micro-F1 (higher better) | 0.1803 | 0.1679 |
| 8 | Language Grounding | `caption_grounding` | original task | text-like query and candidate windows -> ranked matching moments | MRR (higher better) | 0.0160 | 0.0168 |
| 9 | Cross-Modal Retrieval | `cross_modal_retrieval` | original task | motion/IMU/pose query; depth/video candidates -> ranked visual windows | MRR (higher better) | 0.2693 | 0.1300 |
| 10 | Cross-Modal Reconstruction | `modality_reconstruction` | original task | motion, IMU, and camera/pose features -> reconstructed depth/video vector | R2 (higher better) | -0.0153 | -0.0102 |
| 11 | Temporal Order Verification | `temporal_order` | original task | two adjacent windows plus difference vector -> correct or reversed | F1 (higher better) | 0.5400 | 0.8520 |
| 12 | Multimodal Synchronization Detection | `misalignment_detection` | original task | motion-side and visual/depth-side feature groups -> aligned or shifted | F1 (higher better) | 0.5052 | 0.7153 |
| 13 | Long-Horizon Next-Action Forecasting | `long_horizon_next_action` | additional task | Current 20-frame non-caption multimodal window. -> Action label five seconds later. | macro-F1 (higher better) | 0.0750 | 0.0655 |
| 14 | Long-Horizon Next-Subtask Forecasting | `next_subtask_forecast` | additional task | Current 20-frame non-caption multimodal window. -> Procedure subtask label five seconds later. | macro-F1 (higher better) | 0.0455 | 0.0507 |
| 15 | Interaction Text Prediction | `interaction_text_prediction` | additional task | Current 20-frame sensor window with caption-text features removed. -> Raw annotation interaction phrase for the same window. | macro-F1 (higher better) | 0.0444 | 0.0381 |
| 16 | Action-Object Relation Prediction | `action_object_relation` | additional task | Current 20-frame sensor window with caption-text features removed. -> Joint action plus active object-set relation. | macro-F1 (higher better) | 0.0000 | 0.0000 |
| 17 | Future Object-Set Forecasting | `object_set_forecast` | additional task | Current 20-frame sensor window with caption-text features removed. -> Object set active five seconds later. | micro-F1 (higher better) | 0.1694 | 0.1972 |
| 18 | IMU-to-Hand Pose Reconstruction | `imu_to_hand_pose` | additional task | Current IMU acceleration/gyroscope feature block only. -> Current left/right hand joint feature blocks. | MAE (lower better) | 0.0420 | 0.0426 |
| 19 | Camera-View Synchronization Retrieval | `camera_view_sync_retrieval` | additional task | Fisheye camera-1 feature query projected into fisheye camera-3 feature space. -> The synchronized held-out camera-3 window. | MRR (higher better) | 0.4943 | 0.2409 |
| 20 | Time-to-Next-Transition Regression | `time_to_transition` | additional task | Current 20-frame non-caption multimodal window. -> Frames until the next action-label boundary, capped at 200 frames. | MAE frames (lower better) | 10.5374 | 10.5545 |

## Machine-Readable Copy

The JSON mirror is `docs/data/task_suite_20.json`.

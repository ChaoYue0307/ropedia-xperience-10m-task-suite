# 128-Episode Aligned Baselines

These results align the earlier simple and neural baseline framing to the same selected 128-episode split used by the Qwen3-Omni pilot.

The runner uses the derived Qwen JSONL export and public-safe metadata. It does not use raw Xperience-10M videos, HDF5 files, sensor NPZ blocks, Qwen weights, or LoRA weights.

## Split

- Train windows: `25629`
- Validation windows: `4608`
- Test windows: `4032`
- Exported episodes: `{'test': 16, 'train': 96, 'val': 16}`

## Coverage

| task | artifact id | simple status | simple primary | neural status | neural primary |
| --- | --- | --- | ---: | --- | ---: |
| Action Recognition | `timeline_action` | pass | 0.0083 | pass | 0.0042 |
| Procedure Step Recognition | `timeline_subtask` | pass | 0.0002 | pass | 0.0001 |
| Action Boundary Detection | `transition_detection` | pass | 0.2965 | pass | 0.4842 |
| Next-Action Prediction | `next_action` | pass | 0.0065 | pass | 0.0049 |
| Hand Trajectory Forecasting | `hand_trajectory_forecast` | unsupported_without_raw_128_feature_blocks |  | not_run |  |
| Contact State Prediction | `contact_prediction` | pass | 0.4381 | pass | 0.5683 |
| Object Relevance Prediction | `object_relevance` | pass | 0.1776 | pass | 0.1866 |
| Language Grounding | `caption_grounding` | pass | 0.0023 | pass | 0.0082 |
| Cross-Modal Retrieval | `cross_modal_retrieval` | unsupported_without_raw_128_feature_blocks |  | not_run |  |
| Cross-Modal Reconstruction | `modality_reconstruction` | unsupported_without_raw_128_feature_blocks |  | not_run |  |
| Temporal Order Verification | `temporal_order` | pass | 0.4199 | pass | 0.8252 |
| Multimodal Synchronization Detection | `misalignment_detection` | unsupported_without_raw_128_feature_blocks |  | not_run |  |
| Long Horizon Next Action | `long_horizon_next_action` | pass | 0.0046 | pass | 0.0030 |
| Next Subtask Forecast | `next_subtask_forecast` | pass | 0.0001 | pass | 0.0000 |
| Interaction Text Prediction | `interaction_text_prediction` | unsupported_without_raw_128_feature_blocks |  | not_run |  |
| Action Object Relation | `action_object_relation` | pass | 0.0000 | pass | 0.0000 |
| Object Set Forecast | `object_set_forecast` | pass | 0.1766 | pass | 0.1742 |
| Imu To Hand Pose | `imu_to_hand_pose` | unsupported_without_raw_128_feature_blocks |  | not_run |  |
| Camera View Sync Retrieval | `camera_view_sync_retrieval` | unsupported_without_raw_128_feature_blocks |  | not_run |  |
| Time To Transition | `time_to_transition` | pass | 624.8109 | pass | 41.4664 |

## Interpretation

The trainable scores are metadata/text baselines, not replacements for full raw-modality baselines. They are useful for checking split alignment, label difficulty, train/test label coverage, and whether the Qwen diagnostic run is being compared against the same 96/16/16 episode setup.

Tasks marked `unsupported_without_raw_128_feature_blocks` still need the 128-run sensor feature NPZ blocks to reproduce the single-episode feature-level target exactly.

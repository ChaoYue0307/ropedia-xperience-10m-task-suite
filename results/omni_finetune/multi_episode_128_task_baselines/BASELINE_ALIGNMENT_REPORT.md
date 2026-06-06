# 128-Episode Aligned Baselines

These results align the earlier simple and neural baseline framing to the same selected 128-episode split used by the Qwen3-Omni pilot.

The runner uses the derived Qwen JSONL export and public-safe metadata. It does not use raw Xperience-10M videos, HDF5 files, sensor NPZ blocks, Qwen weights, or LoRA weights.

## Split

- Train windows: `2848`
- Validation windows: `512`
- Test windows: `448`
- Exported episodes: `{'test': 16, 'train': 96, 'val': 16}`

## Coverage

| task | artifact id | simple status | simple primary | neural status | neural primary |
| --- | --- | --- | ---: | --- | ---: |
| Action Recognition | `timeline_action` | pass | 0.0002 | pass | 0.0000 |
| Procedure Step Recognition | `timeline_subtask` | pass | 0.0000 | pass | 0.0000 |
| Action Boundary Detection | `transition_detection` | pass | 0.5220 | pass | 0.4582 |
| Next-Action Prediction | `next_action` | pass | 0.0002 | pass | 0.0000 |
| Hand Trajectory Forecasting | `hand_trajectory_forecast` | unsupported_without_raw_128_feature_blocks |  | not_run |  |
| Contact State Prediction | `contact_prediction` | pass | 0.5168 | pass | 0.2195 |
| Object Relevance Prediction | `object_relevance` | pass | 0.1822 | pass | 0.1054 |
| Language Grounding | `caption_grounding` | pass | 0.0128 | not_run |  |
| Cross-Modal Retrieval | `cross_modal_retrieval` | unsupported_without_raw_128_feature_blocks |  | not_run |  |
| Cross-Modal Reconstruction | `modality_reconstruction` | unsupported_without_raw_128_feature_blocks |  | not_run |  |
| Temporal Order Verification | `temporal_order` | pass | 0.3271 | not_run |  |
| Multimodal Synchronization Detection | `misalignment_detection` | unsupported_without_raw_128_feature_blocks |  | not_run |  |

## Interpretation

The trainable scores are metadata/text baselines, not replacements for full raw-modality baselines. They are useful for checking split alignment, label difficulty, train/test label coverage, and whether the Qwen diagnostic run is being compared against the same 96/16/16 episode setup.

Tasks marked `unsupported_without_raw_128_feature_blocks` still need the 128-run sensor feature NPZ blocks to reproduce the single-episode feature-level target exactly.

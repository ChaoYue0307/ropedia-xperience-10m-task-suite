# Unified 20-Task Provenance Baselines

This historical result bundle is part of the unified 20-task public-sample suite. The rows here reuse the same 20-frame windows, 5-frame stride, shared feature tensor, chronological split, and minimal/neural baseline discipline as the rest of the suite.

The file and directory names still contain `tier2_task_suite` for backwards-compatible public links, but this is not a separate benchmark tier.

## Setup Alignment

- Unified task contracts: `20`
- Provenance rows in this historical bundle: `8`
- Long-horizon offset: `100` frames, about `5.0` seconds at 20 FPS
- Raw public-sample HDF5 is required to regenerate the interaction/object targets; raw media/HDF5 files are not redistributed.

## Results

| # | Task | Input | Output | Minimal | Neural MLP | Meaning |
| ---: | --- | --- | --- | ---: | ---: | --- |
| 13 | Long-Horizon Next-Action Forecasting | Current 20-frame non-caption multimodal window. | Action label five seconds later. | 0.0750 macro-F1 | 0.0655 macro-F1 | Tests whether the current state carries enough procedure context to forecast beyond the one-second core next-action task. |
| 14 | Long-Horizon Next-Subtask Forecasting | Current 20-frame non-caption multimodal window. | Procedure subtask label five seconds later. | 0.0455 macro-F1 | 0.0507 macro-F1 | Moves from immediate action anticipation to higher-level procedure-state prediction. |
| 15 | Interaction Text Prediction | Current 20-frame sensor window with caption-text features removed. | Raw annotation interaction phrase for the same window. | 0.0444 macro-F1 | 0.0381 macro-F1 | Uses the raw caption JSON interaction field as a language target instead of only the hashed text feature. |
| 16 | Action-Object Relation Prediction | Current 20-frame sensor window with caption-text features removed. | Joint action plus active object-set relation. | 0.0000 macro-F1 | 0.0000 macro-F1 | Evaluates whether a model can bind what action is happening to which objects are involved. |
| 17 | Future Object-Set Forecasting | Current 20-frame sensor window with caption-text features removed. | Object set active five seconds later. | 0.1694 micro-F1 | 0.1972 micro-F1 | Predicts which objects will become relevant soon, not only which objects are relevant now. |
| 18 | IMU-to-Hand Pose Reconstruction | Current IMU acceleration/gyroscope feature block only. | Current left/right hand joint feature blocks. | 0.0420 MAE | 0.0426 MAE | A sensor-bridge probe for how much hand configuration can be recovered from inertial motion alone. |
| 19 | Camera-View Synchronization Retrieval | Fisheye camera-1 feature query projected into fisheye camera-3 feature space. | The synchronized held-out camera-3 window. | 0.4943 MRR | 0.2409 MRR | Stress-tests multi-camera time alignment beyond the core cross-modal retrieval task. |
| 20 | Time-to-Next-Transition Regression | Current 20-frame non-caption multimodal window. | Frames until the next action-label boundary, capped at 200 frames. | 10.5374 MAE frames | 10.5545 MAE frames | Turns boundary detection into a continuous timing estimate for procedural control. |

## Interpretation Boundary

These sample-level baselines are part of the same unified public-sample suite. They prove that the sample can support richer task contracts, but they do not prove cross-episode model quality.

# Task Method 20-Result Matrix

Every method has one record for each of the 20 unified task contracts. Numeric scores appear only where a committed runner or verified package produced that task target.

Legend: `score` = numeric task score, `proxy` = documented raw128 compact proxy score, `unsupported` = artifact exists but required target is not present, `not supported` = metadata-only package cannot form that target, `not evaluated` = verified model package did not request that target.

| Method | Records | Scored | Proxy scored | Scoreless | Status counts |
| --- | ---: | ---: | ---: | ---: | --- |
| Minimal | 20 | 20 | 0 | 0 | scored 20 |
| Neural MLP | 20 | 20 | 0 | 0 | scored 20 |
| 128ep Aligned Simple | 20 | 19 | 0 | 1 | scored 19, unsupported 1 |
| 128ep Aligned NN | 20 | 19 | 0 | 1 | not supported 1, scored 19 |
| 128ep Raw Simple | 20 | 20 | 2 | 0 | proxy scored 2, scored 18 |
| 128ep Raw NN | 20 | 20 | 2 | 0 | proxy scored 2, scored 18 |
| Qwen3-Omni v6 LoRA | 20 | 20 | 0 | 0 | scored 20 |
| Cosmos3-Super Reasoner | 20 | 19 | 0 | 1 | not evaluated 1, scored 19 |
| Cosmos3-Nano Future Window | 20 | 11 | 0 | 9 | not evaluated 9, scored 11 |

| # | Task | Min | NN | 128-S | 128-NN | 128-RS | 128-RN | Qwen3 | C3-S | C3-N |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01 | Action Recognition | score | score | score | score | score | score | score | score | score |
| 02 | Procedure Step Recognition | score | score | score | score | score | score | score | score | not evaluated |
| 03 | Action Boundary Detection | score | score | score | score | score | score | score | score | score |
| 04 | Next-Action Prediction | score | score | score | score | score | score | score | score | score |
| 05 | Hand Trajectory Forecasting | score | score | score | score | score | score | score | score | not evaluated |
| 06 | Contact State Prediction | score | score | score | score | score | score | score | score | score |
| 07 | Object Relevance Prediction | score | score | score | score | score | score | score | score | not evaluated |
| 08 | Language Grounding | score | score | score | score | score | score | score | score | not evaluated |
| 09 | Cross-Modal Retrieval | score | score | score | score | score | score | score | score | score |
| 10 | Cross-Modal Reconstruction | score | score | score | score | score | score | score | score | score |
| 11 | Temporal Order Verification | score | score | score | score | score | score | score | score | not evaluated |
| 12 | Multimodal Synchronization Detection | score | score | score | score | score | score | score | score | not evaluated |
| 13 | Long-Horizon Next-Action Forecasting | score | score | score | score | score | score | score | score | score |
| 14 | Long-Horizon Next-Subtask Forecasting | score | score | score | score | score | score | score | score | score |
| 15 | Interaction Text Prediction | score | score | score | score | proxy | proxy | score | not evaluated | not evaluated |
| 16 | Action-Object Relation Prediction | score | score | score | score | score | score | score | score | score |
| 17 | Future Object-Set Forecasting | score | score | score | score | score | score | score | score | score |
| 18 | IMU-to-Hand Pose Reconstruction | score | score | score | score | score | score | score | score | not evaluated |
| 19 | Camera-View Synchronization Retrieval | score | score | unsupported | not supported | proxy | proxy | score | score | not evaluated |
| 20 | Time-to-Next-Transition Regression | score | score | score | score | score | score | score | score | score |

Sources and raw values are in `docs/data/task_method_20_result_matrix.json` and `docs/data/unified_task_model_radar.json`.

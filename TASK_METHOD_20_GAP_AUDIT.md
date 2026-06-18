# Task Method 20-Result Gap Audit

Generated: `2026-06-18T12:07:14+00:00`

This audit is the explicit gap ledger for the 9-method x 20-task result matrix.
It keeps missing cells visible while preserving the rule that a numeric score
requires a real task target and source artifact.

## Score Summary

- Method-task records: `180`
- Numeric scored records: `127`
- Scoreless records: `53`
- Proxy-scored records: `4`
- Source matrix: [`docs/data/task_method_20_result_matrix.json`](docs/data/task_method_20_result_matrix.json)

## Method Coverage

| Method | ID | Scored | Scoreless | Proxy | Status counts |
| --- | --- | --- | --- | --- | --- |
| Minimal | minimal | 20/20 | 0 | 0 | scored: 20 |
| Neural MLP | neural_mlp | 20/20 | 0 | 0 | scored: 20 |
| 128ep Metadata Simple | metadata128_simple | 13/20 | 7 | 0 | scored: 13, unsupported_without_required_target: 7 |
| 128ep Metadata NN | metadata128_neural_mlp | 7/20 | 13 | 0 | not_supported_by_metadata_only_package: 7, scored: 7, unsupported_without_required_target: 6 |
| 128ep Raw Simple | raw128_simple | 20/20 | 0 | 2 | proxy_scored: 2, scored: 18 |
| 128ep Raw NN | raw128_neural_mlp | 20/20 | 0 | 2 | proxy_scored: 2, scored: 18 |
| Qwen3-Omni v6 LoRA | qwen3_omni_v6_lora | 15/20 | 5 | 0 | not_evaluated_in_verified_package: 5, scored: 15 |
| Cosmos3-Super Reasoner | cosmos3_super_reasoner | 7/20 | 13 | 0 | not_evaluated_in_verified_package: 13, scored: 7 |
| Cosmos3-Nano Future Window | cosmos3_nano_future_window | 5/20 | 15 | 0 | not_evaluated_in_verified_package: 15, scored: 5 |

## Gap Classes

| Status | Count | Next step |
| --- | --- | --- |
| not_evaluated_in_verified_package | 33 | Generate verified model outputs for this task contract and score them against the held-out labels. |
| not_supported_by_metadata_only_package | 7 | Run the task with raw sensor-feature blocks or add a task-specific metadata target builder before assigning a numeric score. |
| unsupported_without_required_target | 13 | Export the missing target field for this 128-episode method, then rerun the same train/validation/test split. |

## Scoreless Records

| Task | Task label | Method | Status | Required evidence |
| --- | --- | --- | --- | --- |
| 01 | Action Recognition | 128ep Metadata NN | unsupported | Export the missing target field for this 128-episode method, then rerun the same train/validation/test split. |
| 02 | Procedure Step Recognition | 128ep Metadata NN | unsupported | Export the missing target field for this 128-episode method, then rerun the same train/validation/test split. |
| 02 | Procedure Step Recognition | Cosmos3-Nano Future Window | not evaluated | Generate verified model outputs for this task contract and score them against the held-out labels. |
| 04 | Next-Action Prediction | 128ep Metadata NN | unsupported | Export the missing target field for this 128-episode method, then rerun the same train/validation/test split. |
| 05 | Hand Trajectory Forecasting | 128ep Metadata Simple | unsupported | Export the missing target field for this 128-episode method, then rerun the same train/validation/test split. |
| 05 | Hand Trajectory Forecasting | 128ep Metadata NN | not supported | Run the task with raw sensor-feature blocks or add a task-specific metadata target builder before assigning a numeric score. |
| 05 | Hand Trajectory Forecasting | Qwen3-Omni v6 LoRA | not evaluated | Generate verified model outputs for this task contract and score them against the held-out labels. |
| 05 | Hand Trajectory Forecasting | Cosmos3-Super Reasoner | not evaluated | Generate verified model outputs for this task contract and score them against the held-out labels. |
| 05 | Hand Trajectory Forecasting | Cosmos3-Nano Future Window | not evaluated | Generate verified model outputs for this task contract and score them against the held-out labels. |
| 07 | Object Relevance Prediction | Cosmos3-Nano Future Window | not evaluated | Generate verified model outputs for this task contract and score them against the held-out labels. |
| 08 | Language Grounding | Cosmos3-Super Reasoner | not evaluated | Generate verified model outputs for this task contract and score them against the held-out labels. |
| 08 | Language Grounding | Cosmos3-Nano Future Window | not evaluated | Generate verified model outputs for this task contract and score them against the held-out labels. |
| 09 | Cross-Modal Retrieval | 128ep Metadata Simple | unsupported | Export the missing target field for this 128-episode method, then rerun the same train/validation/test split. |
| 09 | Cross-Modal Retrieval | 128ep Metadata NN | not supported | Run the task with raw sensor-feature blocks or add a task-specific metadata target builder before assigning a numeric score. |
| 09 | Cross-Modal Retrieval | Cosmos3-Super Reasoner | not evaluated | Generate verified model outputs for this task contract and score them against the held-out labels. |
| 10 | Cross-Modal Reconstruction | 128ep Metadata Simple | unsupported | Export the missing target field for this 128-episode method, then rerun the same train/validation/test split. |
| 10 | Cross-Modal Reconstruction | 128ep Metadata NN | not supported | Run the task with raw sensor-feature blocks or add a task-specific metadata target builder before assigning a numeric score. |
| 10 | Cross-Modal Reconstruction | Qwen3-Omni v6 LoRA | not evaluated | Generate verified model outputs for this task contract and score them against the held-out labels. |
| 10 | Cross-Modal Reconstruction | Cosmos3-Super Reasoner | not evaluated | Generate verified model outputs for this task contract and score them against the held-out labels. |
| 10 | Cross-Modal Reconstruction | Cosmos3-Nano Future Window | not evaluated | Generate verified model outputs for this task contract and score them against the held-out labels. |
| 11 | Temporal Order Verification | Cosmos3-Super Reasoner | not evaluated | Generate verified model outputs for this task contract and score them against the held-out labels. |
| 11 | Temporal Order Verification | Cosmos3-Nano Future Window | not evaluated | Generate verified model outputs for this task contract and score them against the held-out labels. |
| 12 | Multimodal Synchronization Detection | 128ep Metadata Simple | unsupported | Export the missing target field for this 128-episode method, then rerun the same train/validation/test split. |
| 12 | Multimodal Synchronization Detection | 128ep Metadata NN | not supported | Run the task with raw sensor-feature blocks or add a task-specific metadata target builder before assigning a numeric score. |
| 12 | Multimodal Synchronization Detection | Cosmos3-Super Reasoner | not evaluated | Generate verified model outputs for this task contract and score them against the held-out labels. |
| 12 | Multimodal Synchronization Detection | Cosmos3-Nano Future Window | not evaluated | Generate verified model outputs for this task contract and score them against the held-out labels. |
| 13 | Long-Horizon Next-Action Forecasting | 128ep Metadata NN | unsupported | Export the missing target field for this 128-episode method, then rerun the same train/validation/test split. |
| 13 | Long-Horizon Next-Action Forecasting | Cosmos3-Super Reasoner | not evaluated | Generate verified model outputs for this task contract and score them against the held-out labels. |
| 13 | Long-Horizon Next-Action Forecasting | Cosmos3-Nano Future Window | not evaluated | Generate verified model outputs for this task contract and score them against the held-out labels. |
| 14 | Long-Horizon Next-Subtask Forecasting | 128ep Metadata NN | unsupported | Export the missing target field for this 128-episode method, then rerun the same train/validation/test split. |
| 14 | Long-Horizon Next-Subtask Forecasting | Cosmos3-Super Reasoner | not evaluated | Generate verified model outputs for this task contract and score them against the held-out labels. |
| 14 | Long-Horizon Next-Subtask Forecasting | Cosmos3-Nano Future Window | not evaluated | Generate verified model outputs for this task contract and score them against the held-out labels. |
| 15 | Interaction Text Prediction | 128ep Metadata Simple | unsupported | Export the missing target field for this 128-episode method, then rerun the same train/validation/test split. |
| 15 | Interaction Text Prediction | 128ep Metadata NN | not supported | Run the task with raw sensor-feature blocks or add a task-specific metadata target builder before assigning a numeric score. |
| 15 | Interaction Text Prediction | Qwen3-Omni v6 LoRA | not evaluated | Generate verified model outputs for this task contract and score them against the held-out labels. |
| 15 | Interaction Text Prediction | Cosmos3-Super Reasoner | not evaluated | Generate verified model outputs for this task contract and score them against the held-out labels. |
| 15 | Interaction Text Prediction | Cosmos3-Nano Future Window | not evaluated | Generate verified model outputs for this task contract and score them against the held-out labels. |
| 16 | Action-Object Relation Prediction | 128ep Metadata NN | unsupported | Export the missing target field for this 128-episode method, then rerun the same train/validation/test split. |
| 16 | Action-Object Relation Prediction | Cosmos3-Nano Future Window | not evaluated | Generate verified model outputs for this task contract and score them against the held-out labels. |
| 17 | Future Object-Set Forecasting | Cosmos3-Super Reasoner | not evaluated | Generate verified model outputs for this task contract and score them against the held-out labels. |
| 17 | Future Object-Set Forecasting | Cosmos3-Nano Future Window | not evaluated | Generate verified model outputs for this task contract and score them against the held-out labels. |
| 18 | IMU-to-Hand Pose Reconstruction | 128ep Metadata Simple | unsupported | Export the missing target field for this 128-episode method, then rerun the same train/validation/test split. |
| 18 | IMU-to-Hand Pose Reconstruction | 128ep Metadata NN | not supported | Run the task with raw sensor-feature blocks or add a task-specific metadata target builder before assigning a numeric score. |
| 18 | IMU-to-Hand Pose Reconstruction | Qwen3-Omni v6 LoRA | not evaluated | Generate verified model outputs for this task contract and score them against the held-out labels. |
| 18 | IMU-to-Hand Pose Reconstruction | Cosmos3-Super Reasoner | not evaluated | Generate verified model outputs for this task contract and score them against the held-out labels. |
| 18 | IMU-to-Hand Pose Reconstruction | Cosmos3-Nano Future Window | not evaluated | Generate verified model outputs for this task contract and score them against the held-out labels. |
| 19 | Camera-View Synchronization Retrieval | 128ep Metadata Simple | unsupported | Export the missing target field for this 128-episode method, then rerun the same train/validation/test split. |
| 19 | Camera-View Synchronization Retrieval | 128ep Metadata NN | not supported | Run the task with raw sensor-feature blocks or add a task-specific metadata target builder before assigning a numeric score. |
| 19 | Camera-View Synchronization Retrieval | Qwen3-Omni v6 LoRA | not evaluated | Generate verified model outputs for this task contract and score them against the held-out labels. |
| 19 | Camera-View Synchronization Retrieval | Cosmos3-Super Reasoner | not evaluated | Generate verified model outputs for this task contract and score them against the held-out labels. |
| 19 | Camera-View Synchronization Retrieval | Cosmos3-Nano Future Window | not evaluated | Generate verified model outputs for this task contract and score them against the held-out labels. |
| 20 | Time-to-Next-Transition Regression | Cosmos3-Super Reasoner | not evaluated | Generate verified model outputs for this task contract and score them against the held-out labels. |
| 20 | Time-to-Next-Transition Regression | Cosmos3-Nano Future Window | not evaluated | Generate verified model outputs for this task contract and score them against the held-out labels. |

## Proxy Records

| Task | Task label | Method | Metric | Proxy note |
| --- | --- | --- | --- | --- |
| 15 | Interaction Text Prediction | 128ep Raw Simple | macro_f1 | documented compact proxy completion for this raw128 task axis |
| 15 | Interaction Text Prediction | 128ep Raw NN | macro_f1 | documented compact proxy completion for this raw128 task axis |
| 19 | Camera-View Synchronization Retrieval | 128ep Raw Simple | mrr | documented compact proxy completion for this raw128 task axis |
| 19 | Camera-View Synchronization Retrieval | 128ep Raw NN | mrr | documented compact proxy completion for this raw128 task axis |

## Immediate Actions

- Keep [`docs/data/task_method_20_gap_audit.json`](docs/data/task_method_20_gap_audit.json) next to the radar and matrix so readers can distinguish scored, proxy-scored, and scoreless cells.
- Use [`scripts/omni/score_model_output_probes.py`](scripts/omni/score_model_output_probes.py) to check whether train/validation/test model outputs are present before trying to extend Qwen3/Cosmos to all 20 task contracts.
- Use [`scripts/omni/launch_all_task_model_scoring_when_free.sh`](scripts/omni/launch_all_task_model_scoring_when_free.sh) as the guarded waiter for a real all-task scoring command when private GPU capacity is available.

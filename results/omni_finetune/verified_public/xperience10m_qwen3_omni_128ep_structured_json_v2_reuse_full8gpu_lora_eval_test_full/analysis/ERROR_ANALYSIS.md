# Qwen3-Omni LoRA v2 strict-json Held-Out Error Analysis

This report is computed from the verified public package predictions. It contains only derived metrics and sanitized examples.

## Overall

- Prediction rows: `448`
- JSON validity from `metrics.json`: `0.9978`
- Parsed prediction rate from public rows: `0.9978`
- Action exact rate: `0.0290`
- Subtask exact rate: `0.0022`
- Contact exact rate: `0.7188`
- Object F1: `0.3016`

## Weakest Episode Groups

| group | samples | parsed_prediction_rate | action_exact_rate | object_f1 |
| --- | --- | --- | --- | --- |
| 8a8e1b3c-607e-4ada-b3fd-fa639727e92c__ep1 | 32 | 0.9688 | 0.0312 | 0.1677 |
| 9c553886-83c5-4dc4-be5c-dcb269b3a771__ep2 | 32 | 1.0000 | 0.0000 | 0.3745 |
| b9dd769b-e31a-4fdb-945e-5a60db6487b0__ep2 | 32 | 1.0000 | 0.0000 | 0.4016 |
| 5399ef86-4df9-49bc-809f-8f4f92f9e659__ep6 | 32 | 1.0000 | 0.0000 | 0.0286 |
| 877779cd-25f3-4293-a3c4-39067dd9558c__ep4 | 32 | 1.0000 | 0.0000 | 0.3587 |
| 1796b943-caad-43c6-b9bd-80b8d601f37d__ep1 | 32 | 1.0000 | 0.0000 | 0.1244 |
| a1012a57-385e-45a9-8a59-694a26fe92a5__ep1 | 32 | 1.0000 | 0.0312 | 0.5714 |
| ba045ed4-ef25-404d-b756-8dcbd45b18fa__ep2 | 32 | 1.0000 | 0.0312 | 0.1205 |

## Action Families

| group | samples | parsed_prediction_rate | action_exact_rate | subtask_exact_rate | object_f1 |
| --- | --- | --- | --- | --- | --- |
| cleaning | 8 | 0.8750 | 0.0000 | 0.0000 | 0.0455 |
| locomotion | 23 | 1.0000 | 0.0000 | 0.0000 | 0.1250 |
| small_object_sorting | 87 | 1.0000 | 0.0000 | 0.0000 | 0.2707 |
| other | 94 | 1.0000 | 0.0000 | 0.0000 | 0.2803 |
| phone_use | 51 | 1.0000 | 0.0196 | 0.0000 | 0.4132 |
| paper_cardboard_craft | 142 | 1.0000 | 0.0493 | 0.0070 | 0.3580 |
| retail_stocking | 38 | 1.0000 | 0.1053 | 0.0000 | 0.1582 |
| food_kitchen | 5 | 1.0000 | 0.2000 | 0.0000 | 0.1538 |

## Train-Seen Split

| group | samples | parsed_prediction_rate | action_exact_rate | next_action_exact_rate |
| --- | --- | --- | --- | --- |
| unseen_in_train | 317 | 0.9968 | 0.0126 | 0.0126 |
| seen_in_train | 131 | 1.0000 | 0.0687 | 0.0687 |

## Required-Modality State

| group | samples | parsed_prediction_rate | action_exact_rate | object_f1 |
| --- | --- | --- | --- | --- |
| rrd_missing_only_required_modalities_present | 448 | 0.9978 | 0.0290 | 0.3016 |

## Object Categories

| group | samples | object_precision | object_recall | object_f1 |
| --- | --- | --- | --- | --- |
| furniture_room | 96 | 0.3356 | 0.3155 | 0.3252 |
| other_object | 135 | 0.2000 | 0.2048 | 0.2024 |
| no_object_label | 2 | 0.0000 | 0.0000 | 0.0000 |
| cleaning | 8 | 0.0455 | 0.0476 | 0.0465 |
| craft_small_object | 106 | 0.2508 | 0.2645 | 0.2575 |
| tool_stationery | 138 | 0.4413 | 0.4422 | 0.4417 |
| food_kitchen | 56 | 0.3202 | 0.2889 | 0.3037 |
| phone_device | 162 | 0.3915 | 0.3719 | 0.3814 |

## Interpretation

The diagnostic pilot is dominated by invalid or weak structured outputs and exact-label failures. These tables identify where to tighten JSON constraints, action/subtask target formatting, object vocabularies, and missing-modality robustness before claiming stronger model quality.

Generated files:

- `error_analysis_summary.json`
- `episode_error_analysis.csv`
- `action_family_error_analysis.csv`
- `train_seen_error_analysis.csv`
- `missing_modality_error_analysis.csv`
- `object_category_error_analysis.csv`

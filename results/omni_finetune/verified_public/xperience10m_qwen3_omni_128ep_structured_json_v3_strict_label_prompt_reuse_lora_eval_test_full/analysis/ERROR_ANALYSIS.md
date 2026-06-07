# Qwen3-Omni strict-label v3 Held-Out Error Analysis

This report is computed from public-safe predictions and an episode manifest. It contains only derived metrics and sanitized examples.

## Overall

- Prediction rows: `448`
- JSON validity from `metrics.json`: `1.0000`
- Parsed prediction rate from public rows: `1.0000`
- Action exact rate: `0.0312`
- Subtask exact rate: `0.0022`
- Contact exact rate: `0.7210`
- Object F1: `0.3069`

## Weakest Episode Groups

| group | samples | parsed_prediction_rate | action_exact_rate | object_f1 |
| --- | --- | --- | --- | --- |
| a1012a57-385e-45a9-8a59-694a26fe92a5__ep1 | 32 | 1.0000 | 0.0000 | 0.5341 |
| 9c553886-83c5-4dc4-be5c-dcb269b3a771__ep2 | 32 | 1.0000 | 0.0000 | 0.3719 |
| ba045ed4-ef25-404d-b756-8dcbd45b18fa__ep2 | 32 | 1.0000 | 0.0000 | 0.1039 |
| 5399ef86-4df9-49bc-809f-8f4f92f9e659__ep6 | 32 | 1.0000 | 0.0000 | 0.0000 |
| 877779cd-25f3-4293-a3c4-39067dd9558c__ep4 | 32 | 1.0000 | 0.0000 | 0.3315 |
| 1796b943-caad-43c6-b9bd-80b8d601f37d__ep1 | 32 | 1.0000 | 0.0000 | 0.1347 |
| 8a8e1b3c-607e-4ada-b3fd-fa639727e92c__ep1 | 32 | 1.0000 | 0.0312 | 0.1928 |
| 33f7ae08-ac1d-4321-9cb9-eca79016b359__ep1 | 32 | 1.0000 | 0.0312 | 0.0774 |

## Action Families

| group | samples | parsed_prediction_rate | action_exact_rate | subtask_exact_rate | object_f1 |
| --- | --- | --- | --- | --- | --- |
| cleaning | 8 | 1.0000 | 0.0000 | 0.0000 | 0.0435 |
| locomotion | 23 | 1.0000 | 0.0000 | 0.0000 | 0.1471 |
| small_object_sorting | 87 | 1.0000 | 0.0000 | 0.0000 | 0.2716 |
| other | 94 | 1.0000 | 0.0106 | 0.0000 | 0.2831 |
| phone_use | 51 | 1.0000 | 0.0392 | 0.0000 | 0.4175 |
| paper_cardboard_craft | 142 | 1.0000 | 0.0493 | 0.0070 | 0.3559 |
| retail_stocking | 38 | 1.0000 | 0.0789 | 0.0000 | 0.2093 |
| food_kitchen | 5 | 1.0000 | 0.2000 | 0.0000 | 0.3333 |

## Train-Seen Split

| group | samples | parsed_prediction_rate | action_exact_rate | next_action_exact_rate |
| --- | --- | --- | --- | --- |
| unseen_in_train | 317 | 1.0000 | 0.0095 | 0.0095 |
| seen_in_train | 131 | 1.0000 | 0.0840 | 0.0840 |

## Required-Modality State

| group | samples | parsed_prediction_rate | action_exact_rate | object_f1 |
| --- | --- | --- | --- | --- |
| rrd_missing_only_required_modalities_present | 448 | 1.0000 | 0.0312 | 0.3069 |

## Object Categories

| group | samples | object_precision | object_recall | object_f1 |
| --- | --- | --- | --- | --- |
| no_object_label | 2 | 0.0000 | 0.0000 | 0.0000 |
| cleaning | 8 | 0.0500 | 0.0476 | 0.0488 |
| craft_small_object | 106 | 0.2604 | 0.2419 | 0.2508 |
| furniture_room | 96 | 0.3047 | 0.2681 | 0.2852 |
| tool_stationery | 138 | 0.4528 | 0.4280 | 0.4400 |
| other_object | 135 | 0.2288 | 0.2119 | 0.2200 |
| food_kitchen | 56 | 0.3875 | 0.2756 | 0.3221 |
| phone_device | 162 | 0.4435 | 0.3618 | 0.3985 |

## Interpretation

The diagnostic pilot is dominated by invalid or weak structured outputs and exact-label failures. These tables identify where to tighten JSON constraints, action/subtask target formatting, object vocabularies, and missing-modality robustness before claiming stronger model quality.

Generated files:

- `error_analysis_summary.json`
- `episode_error_analysis.csv`
- `action_family_error_analysis.csv`
- `train_seen_error_analysis.csv`
- `missing_modality_error_analysis.csv`
- `object_category_error_analysis.csv`

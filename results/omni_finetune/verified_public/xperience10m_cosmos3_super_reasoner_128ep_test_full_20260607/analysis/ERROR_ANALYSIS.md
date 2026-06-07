# Cosmos3-Super Reasoner base-weight Held-Out Error Analysis

This report is computed from the verified public package predictions. It contains only derived metrics and sanitized examples.

## Overall

- Prediction rows: `448`
- JSON validity from `metrics.json`: `0.5112`
- Parsed prediction rate from public rows: `0.5112`
- Action exact rate: `0.0089`
- Subtask exact rate: `0.0000`
- Contact exact rate: `0.3214`
- Object F1: `0.1370`

## Weakest Episode Groups

| group | samples | parsed_prediction_rate | action_exact_rate | object_f1 |
| --- | --- | --- | --- | --- |
| 9c553886-83c5-4dc4-be5c-dcb269b3a771__ep2 | 32 | 0.0938 | 0.0000 | 0.0325 |
| 5399ef86-4df9-49bc-809f-8f4f92f9e659__ep6 | 32 | 0.2500 | 0.0000 | 0.0000 |
| b6579cb5-0a71-4ca6-8808-1e2700be05c7__ep3 | 32 | 0.2812 | 0.0000 | 0.1401 |
| b9dd769b-e31a-4fdb-945e-5a60db6487b0__ep2 | 32 | 0.2812 | 0.0312 | 0.1439 |
| 877779cd-25f3-4293-a3c4-39067dd9558c__ep4 | 32 | 0.3125 | 0.0000 | 0.2182 |
| ba045ed4-ef25-404d-b756-8dcbd45b18fa__ep2 | 32 | 0.4375 | 0.0000 | 0.0000 |
| 1796b943-caad-43c6-b9bd-80b8d601f37d__ep1 | 32 | 0.4688 | 0.0000 | 0.0578 |
| ba18b7c1-21ff-45da-8452-41acce7fc8de__ep2 | 32 | 0.5312 | 0.0000 | 0.1728 |

## Action Families

| group | samples | parsed_prediction_rate | action_exact_rate | subtask_exact_rate | object_f1 |
| --- | --- | --- | --- | --- | --- |
| small_object_sorting | 87 | 0.3218 | 0.0000 | 0.0000 | 0.1299 |
| paper_cardboard_craft | 142 | 0.3239 | 0.0141 | 0.0000 | 0.0840 |
| other | 94 | 0.5106 | 0.0000 | 0.0000 | 0.0900 |
| food_kitchen | 5 | 0.6000 | 0.0000 | 0.0000 | 0.0000 |
| locomotion | 23 | 0.6957 | 0.0000 | 0.0000 | 0.0348 |
| cleaning | 8 | 0.7500 | 0.0000 | 0.0000 | 0.0000 |
| phone_use | 51 | 0.9020 | 0.0000 | 0.0000 | 0.3715 |
| retail_stocking | 38 | 0.9474 | 0.0526 | 0.0000 | 0.2222 |

## Train-Seen Split

| group | samples | parsed_prediction_rate | action_exact_rate | next_action_exact_rate |
| --- | --- | --- | --- | --- |
| unseen_in_train | 317 | 0.4890 | 0.0000 | 0.0095 |
| seen_in_train | 131 | 0.5649 | 0.0305 | 0.0229 |

## Required-Modality State

| group | samples | parsed_prediction_rate | action_exact_rate | object_f1 |
| --- | --- | --- | --- | --- |
| rrd_missing_only_required_modalities_present | 448 | 0.5112 | 0.0089 | 0.1370 |

## Object Categories

| group | samples | object_precision | object_recall | object_f1 |
| --- | --- | --- | --- | --- |
| tool_stationery | 138 | 0.2642 | 0.0852 | 0.1288 |
| craft_small_object | 106 | 0.3204 | 0.1065 | 0.1598 |
| paper_cardboard | 261 | 0.2971 | 0.1067 | 0.1570 |
| no_object_label | 2 | 0.0000 | 0.0000 | 0.0000 |
| furniture_room | 96 | 0.1950 | 0.0978 | 0.1303 |
| phone_device | 162 | 0.5245 | 0.1256 | 0.2027 |
| food_kitchen | 56 | 0.2424 | 0.0711 | 0.1100 |
| other_object | 135 | 0.1185 | 0.0595 | 0.0792 |

## Interpretation

The diagnostic pilot is dominated by invalid or weak structured outputs and exact-label failures. These tables identify where to tighten JSON constraints, action/subtask target formatting, object vocabularies, and missing-modality robustness before claiming stronger model quality.

Generated files:

- `error_analysis_summary.json`
- `episode_error_analysis.csv`
- `action_family_error_analysis.csv`
- `train_seen_error_analysis.csv`
- `missing_modality_error_analysis.csv`
- `object_category_error_analysis.csv`

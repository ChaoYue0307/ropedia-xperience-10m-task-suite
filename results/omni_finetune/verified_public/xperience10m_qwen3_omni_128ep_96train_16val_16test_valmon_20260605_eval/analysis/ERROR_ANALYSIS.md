# Qwen3-Omni Held-Out Error Analysis

This report is computed from the verified public package predictions. It contains only derived metrics and sanitized examples.

## Overall

- Prediction rows: `448`
- JSON validity from `metrics.json`: `0.8750`
- Parsed prediction rate from public rows: `0.8772`
- Action exact rate: `0.0246`
- Subtask exact rate: `0.0067`
- Contact exact rate: `0.6451`
- Object F1: `0.2230`

## Weakest Episode Groups

| group | samples | parsed_prediction_rate | action_exact_rate | object_f1 |
| --- | --- | --- | --- | --- |
| 1796b943-caad-43c6-b9bd-80b8d601f37d__ep1 | 32 | 0.5625 | 0.0000 | 0.0459 |
| 8a8e1b3c-607e-4ada-b3fd-fa639727e92c__ep1 | 32 | 0.7500 | 0.0312 | 0.0942 |
| 33f7ae08-ac1d-4321-9cb9-eca79016b359__ep1 | 32 | 0.8438 | 0.0000 | 0.0529 |
| b750fab3-7fbb-43a0-b451-c64c4d4a64da__ep1 | 32 | 0.8438 | 0.0000 | 0.2353 |
| ba18b7c1-21ff-45da-8452-41acce7fc8de__ep2 | 32 | 0.8438 | 0.0000 | 0.2836 |
| ba045ed4-ef25-404d-b756-8dcbd45b18fa__ep2 | 32 | 0.8438 | 0.0625 | 0.0746 |
| b9dd769b-e31a-4fdb-945e-5a60db6487b0__ep2 | 32 | 0.8750 | 0.0312 | 0.3265 |
| 4b02bb38-384a-438a-b5f9-6131d85c34b0__ep1 | 32 | 0.8750 | 0.0938 | 0.2830 |

## Action Families

| group | samples | parsed_prediction_rate | action_exact_rate | subtask_exact_rate | object_f1 |
| --- | --- | --- | --- | --- | --- |
| locomotion | 23 | 0.2609 | 0.0000 | 0.0000 | 0.0120 |
| food_kitchen | 5 | 0.6000 | 0.2000 | 0.0000 | 0.2727 |
| cleaning | 8 | 0.7500 | 0.0000 | 0.0000 | 0.0000 |
| other | 94 | 0.8511 | 0.0000 | 0.0000 | 0.1910 |
| phone_use | 51 | 0.9020 | 0.0588 | 0.0196 | 0.3501 |
| paper_cardboard_craft | 142 | 0.9225 | 0.0282 | 0.0141 | 0.2308 |
| small_object_sorting | 87 | 0.9655 | 0.0000 | 0.0000 | 0.2740 |
| retail_stocking | 38 | 0.9737 | 0.0789 | 0.0000 | 0.1564 |

## Train-Seen Split

| group | samples | parsed_prediction_rate | action_exact_rate | next_action_exact_rate |
| --- | --- | --- | --- | --- |
| unseen_in_train | 317 | 0.8454 | 0.0158 | 0.0158 |
| seen_in_train | 131 | 0.9542 | 0.0458 | 0.0458 |

## Required-Modality State

| group | samples | parsed_prediction_rate | action_exact_rate | object_f1 |
| --- | --- | --- | --- | --- |
| rrd_missing_only_required_modalities_present | 448 | 0.8772 | 0.0246 | 0.2230 |

## Object Categories

| group | samples | object_precision | object_recall | object_f1 |
| --- | --- | --- | --- | --- |
| furniture_room | 96 | 0.2534 | 0.2334 | 0.2430 |
| other_object | 135 | 0.1372 | 0.1643 | 0.1495 |
| food_kitchen | 56 | 0.2228 | 0.2000 | 0.2108 |
| cleaning | 8 | 0.0400 | 0.0476 | 0.0435 |
| phone_device | 162 | 0.3252 | 0.3132 | 0.3191 |
| paper_cardboard | 261 | 0.2227 | 0.3234 | 0.2638 |
| craft_small_object | 106 | 0.2266 | 0.2581 | 0.2413 |
| retail_container | 101 | 0.2028 | 0.1752 | 0.1880 |

## Interpretation

The diagnostic pilot is dominated by invalid or weak structured outputs and exact-label failures. These tables identify where to tighten JSON constraints, action/subtask target formatting, object vocabularies, and missing-modality robustness before claiming stronger model quality.

Generated files:

- `error_analysis_summary.json`
- `episode_error_analysis.csv`
- `action_family_error_analysis.csv`
- `train_seen_error_analysis.csv`
- `missing_modality_error_analysis.csv`
- `object_category_error_analysis.csv`

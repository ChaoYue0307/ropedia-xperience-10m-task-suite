# Two Evidence Lines

The public Xperience-10M suite has two result lines. Read them separately.

![Two evidence-line map](docs/assets/charts/two_evidence_line_map.svg)

Score formula: 2 single-episode methods x 20 tasks = 40 records; 7 selected-128 methods x 20 tasks = 140 records; total public matrix = 180/180 scored records.

| Line | Data unit | Score statement | Valid claim | Do not claim |
| --- | --- | --- | --- | --- |
| 1 sample episode | One public sample episode; 5,821 frames; 1,161 aligned 20-frame windows; 8,546 feature dimensions. | 40/40 direct scores from Minimal and Neural MLP heads. | Task construction, raw-file inspection, local reproducibility, and controlled single-episode baselines. | Multi-episode generalization. |
| 128 selected episodes | Selected held-out 96/16/16 split; 34,269 exported windows; public-safe processed features linked to official gated episode paths. | 140/140 selected-128 scores: 134 direct + 6 compact-proxy. | Same-split baseline/model comparison, Qwen3/Cosmos diagnostics, and scale-up planning. | Reading compact-proxy cells as direct raw-target measurements. |

## Result Ledger

| Line | Methods | Tasks | Scored records | Direct scores | Proxy scores |
| --- | --- | --- | --- | --- | --- |
| 1 sample episode | 2 | 20 | 40/40 | 40 | 0 |
| 128 selected episodes | 7 | 20 | 140/140 | 134 | 6 compact-proxy scores |
| Total public matrix | 9 | 20 | 180/180 | 174 | 6 |

## Result Files

| Purpose | Artifact |
| --- | --- |
| Two-line map figure | [`docs/assets/charts/two_evidence_line_map.svg`](docs/assets/charts/two_evidence_line_map.svg) |
| Unified 9-method x 20-task matrix | [`docs/data/task_method_20_result_matrix.json`](docs/data/task_method_20_result_matrix.json) |
| Two-line result summary | [`docs/data/two_evidence_line_result_summary.json`](docs/data/two_evidence_line_result_summary.json) |
| 1-episode radar data | [`docs/data/single_episode_task_model_radar.json`](docs/data/single_episode_task_model_radar.json) |
| 128-episode radar data | [`docs/data/episode128_task_model_radar.json`](docs/data/episode128_task_model_radar.json) |
| 128-episode feature index | [`docs/data/xperience10m_128_episode_feature_index.json`](docs/data/xperience10m_128_episode_feature_index.json) |
| Score evidence and proxy ledger | [`docs/data/task_method_20_gap_audit.json`](docs/data/task_method_20_gap_audit.json) |

## Interpretation Rule

Use the 1-episode line for task construction and reproducibility claims.
Use the 128-episode line for held-out comparison and model-branch claims.
Do not mix those claims without naming the evidence line.

## Reading Order

1. Choose the evidence line.
2. Open the matching radar.
3. Inspect the matrix row for method, task, metric, source artifact, and proxy flag.
4. Check compact-proxy cells before interpreting totals.

# Two Evidence Lines

The public Xperience-10M task suite has two result lines. Read them separately.

| Line | Data unit | Methods | Best use |
| --- | --- | --- | --- |
| 1 sample episode | One public sample episode; 5,821 frames; 1,161 aligned 20-frame windows; 8,546 feature dimensions. | Minimal heads and Neural MLP heads on all 20 tasks; 40/40 scored method-task records. | Inspect raw files, understand each task, rerun local baselines, and debug task quality. |
| 128 selected episodes | Selected held-out 96/16/16 split; 34,269 exported windows; public-safe processed features linked to official gated episode paths. | Metadata simple/NN, raw-feature simple/NN, Qwen3-Omni, Cosmos3-Super, Cosmos3-Nano; 140/140 scored 128-line records: 134 direct + 6 compact-proxy scores. | Compare same-split baselines and model branches; keep proxy flags visible when direct raw targets are unavailable. |

## Result Ledger

| Line | Methods | Tasks | Scored records | Direct scores | Proxy scores |
| --- | --- | --- | --- | --- | --- |
| 1 sample episode | 2 | 20 | 40/40 | 40 | 0 |
| 128 selected episodes | 7 | 20 | 140/140 | 134 | 6 compact-proxy scores |
| Total public matrix | 9 | 20 | 180/180 | 174 | 6 |

## Result Files

| Purpose | Artifact |
| --- | --- |
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

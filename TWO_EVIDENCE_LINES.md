# Two Evidence Lines

The public Xperience-10M suite has two evidence lines. Read them separately.

![Two evidence-line map](docs/assets/charts/two_evidence_line_map.svg)

Score formula: 2 single-episode methods x 20 tasks = 40 records; 7 selected-128 methods x 20 tasks = 140 records; total public matrix = 180/180 scored records.

| Line | Data unit | Score statement | Valid claim | Do not claim |
| --- | --- | --- | --- | --- |
| 1 sample episode | One public sample episode; 5,821 frames; 1,161 aligned 20-frame windows; 8,546 feature dimensions. | 40/40 direct scores from Minimal and Neural MLP heads. | Task construction, raw-file inspection, local reproducibility, and controlled single-episode baselines. | Multi-episode generalization. |
| 128 selected episodes | Selected held-out 96/16/16 split; 34,269 exported windows; public-safe processed features linked to official gated episode paths. | 140/140 selected-128 scores: 134 direct + 6 compact-proxy. | Same-split method comparison, Qwen3-Omni v6 LoRA diagnostics, Cosmos3-Super/Cosmos3-Nano diagnostics, and scale-up planning. | Reading compact-proxy cells as direct raw-target measurements. |

## Result Ledger

| Line | Methods | Tasks | Scored records | Direct scores | Proxy scores |
| --- | --- | --- | --- | --- | --- |
| 1 sample episode | 2 | 20 | 40/40 | 40 | 0 |
| 128 selected episodes | 7 | 20 | 140/140 | 134 | 6 compact-proxy scores |
| Total public matrix | 9 | 20 | 180/180 | 174 | 6 |

## Method Blocks

| Evidence line | Method block | Methods | Score statement | Read as |
| --- | --- | --- | --- | --- |
| 1 sample episode | Task-head baselines | Minimal; Neural MLP | 40/40 direct scores. | Task-lab reproducibility and simple-vs-neural behavior. |
| 128 selected episodes | Aligned baseline heads | Metadata simple/NN; raw-feature simple/NN | 80/80 scores: 74 direct + 6 compact-proxy. | Same-split metadata/raw-feature baseline comparison. |
| 128 selected episodes | Qwen3-Omni series | Qwen3-Omni v6 LoRA | 20/20 direct scores from verified selected-128 LoRA and task-specific probes. | Current trainable Qwen3-Omni diagnostic baseline on the selected-128 surface. |
| 128 selected episodes | Cosmos3 series | Cosmos3-Super Reasoner; Cosmos3-Nano Future Window | 40/40 direct scores from verified public-safe reasoner and future-window artifacts. | Cosmos3 reasoner and future-window diagnostics on the selected-128 surface. |

Qwen3 run v1-v6 is a LoRA/evaluation lineage inside the 128-episode line, not the project evidence-line numbering. The 20-task matrix uses Qwen3-Omni v6 LoRA; v5 remains the pinned prior release. Cosmos3-Super Forward-Dynamics LoRA is a separate adapter artifact and is not counted as a 20-task matrix method row.

## Result Files

| Purpose | Artifact |
| --- | --- |
| Two-line map figure | [`docs/assets/charts/two_evidence_line_map.svg`](docs/assets/charts/two_evidence_line_map.svg) |
| Unified 9-method x 20-task matrix | [`docs/data/task_method_20_result_matrix.json`](docs/data/task_method_20_result_matrix.json) |
| Two-line result summary | [`docs/data/two_evidence_line_result_summary.json`](docs/data/two_evidence_line_result_summary.json) |
| Qwen3-Omni v1-v6 run lineage | [`docs/data/qwen3_omni_run_lineage.json`](docs/data/qwen3_omni_run_lineage.json), [`QWEN3_OMNI_RUN_LINEAGE.md`](QWEN3_OMNI_RUN_LINEAGE.md) |
| 1-episode radar data | [`docs/data/single_episode_task_model_radar.json`](docs/data/single_episode_task_model_radar.json) |
| 128-episode radar data | [`docs/data/episode128_task_model_radar.json`](docs/data/episode128_task_model_radar.json) |
| 128-episode feature index | [`docs/data/xperience10m_128_episode_feature_index.json`](docs/data/xperience10m_128_episode_feature_index.json) |
| Score evidence and proxy ledger | [`docs/data/task_method_20_gap_audit.json`](docs/data/task_method_20_gap_audit.json) |

## Interpretation Rule

Use the 1-episode line for task construction and reproducibility claims.
Use the 128-episode line for held-out same-split comparison and model-diagnostic claims.
Do not mix those claims without naming the evidence line.

## Reading Order

1. Choose the evidence line.
2. Open the matching radar.
3. Inspect the matrix row for method, task, metric, source artifact, and proxy flag.
4. Check compact-proxy cells before interpreting totals.

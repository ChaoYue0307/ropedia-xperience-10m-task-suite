# Two Evidence-Line Result Summary

Generated: `2026-06-21T08:38:20+00:00`.

Source matrix: [`docs/data/task_method_20_result_matrix.json`](docs/data/task_method_20_result_matrix.json)

Interpretation rule: Use the 1-episode line for task construction and reproducibility claims. Use the 128-episode line for held-out comparison and model-branch claims.

## Read This First

The suite has two public result lines. Line 1 is the fully inspectable one-episode task lab. Line 2 is the 128-episode comparison surface for aligned baselines, Qwen3-Omni, and Cosmos branches. Do not mix the two when reading scores.

Score formula: 2 single-episode methods x 20 tasks = 40 records; 7 selected-128 methods x 20 tasks = 140 records; total public matrix = 180/180 scored records.

| Line | What the scores mean | Valid claim | Do not claim |
| --- | --- | --- | --- |
| 1 sample episode | 40/40 direct scores from Minimal and Neural MLP heads on the same 20 task contracts. | Supports task construction, file inspection, local reproducibility, and controlled single-episode baseline claims. | Do not use this line as evidence of multi-episode generalization. |
| 128 selected episodes | 140/140 selected-128 scores across seven method branches: 134 direct scores plus 6 documented compact-proxy scores. | Supports same-split comparison, model-branch diagnostics, and scale-up planning on public-safe processed artifacts. | Do not read compact-proxy cells as direct raw-target measurements. |

## Public Score Totals

- Lines: 2
- Tasks per method: 20
- Methods: 9
- Scored records: 180/180
- Direct scores: 174
- Compact-proxy scores: 6 documented cells

## Line Ledger And Entry Points

| Line | Methods | Tasks | Scored records | Direct scores | Proxy scores | Primary visuals | Source artifacts |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 sample episode | 2 | 20 | 40/40 | 40 | 0 | docs/assets/charts/two_evidence_line_map.svg<br>docs/assets/charts/single_episode_task_model_radar.svg | docs/data/single_episode_task_model_radar.json<br>docs/data/two_evidence_line_result_summary.json<br>results/episode_task_suite/summary_report.json<br>results/episode_task_suite/feature_manifest.json<br>docs/single_episode_explorer.html |
| 128 selected episodes | 7 | 20 | 140/140 | 134 | 6 | docs/assets/charts/two_evidence_line_map.svg<br>docs/assets/charts/episode128_task_model_radar.svg<br>docs/assets/charts/unified_task_model_radar.svg | docs/data/episode128_task_model_radar.json<br>docs/data/two_evidence_line_result_summary.json<br>docs/data/xperience10m_128_episode_feature_index.json<br>docs/data/omni_model_comparison.json<br>docs/data/task_method_20_gap_audit.json |

## Method Detail By Line

| Line | Method | Method detail | Scored records | Direct scores | Proxy scores |
| --- | --- | --- | --- | --- | --- |
| 1 sample episode | Minimal | Single-episode simple heads over the public sample split. | 20/20 | 20 | 0 |
| 1 sample episode | Neural MLP | Single-episode compact PyTorch MLP heads on the same 20 task contracts. | 20/20 | 20 | 0 |
| 128 selected episodes | 128ep Aligned Simple | 128-episode aligned simple baselines: JSONL metadata/text tasks plus staged sensor-block tasks where the processed target exists. | 20/20 | 19 | 1 |
| 128 selected episodes | 128ep Aligned NN | 128-episode aligned MLP baselines: JSONL metadata/text tasks plus staged sensor-block tasks where the processed target exists. | 20/20 | 19 | 1 |
| 128 selected episodes | 128ep Raw Simple | 128-episode 4430-dim sensor NPZ simple heads; tasks 15/19 use compact proxies. | 20/20 | 18 | 2 |
| 128 selected episodes | 128ep Raw NN | 128-episode 4430-dim sensor NPZ MLP heads; tasks 15/19 use compact proxies. | 20/20 | 18 | 2 |
| 128 selected episodes | Qwen3-Omni v6 LoRA | Verified held-out Qwen3-Omni v6 LoRA metrics, plus task 16 and any completed private-GPU future/retrieval/sensor-target probes scored from task-specific JSON. | 20/20 | 20 | 0 |
| 128 selected episodes | Cosmos3-Super Reasoner | Verified Cosmos3-Super base-weight Reasoner JSON-task evaluation, plus task 5/8/9/10/11/12/13/14/16/17/18/19/20 probes where public metrics exist. | 20/20 | 20 | 0 |
| 128 selected episodes | Cosmos3-Nano Future Window | Verified Cosmos3-Nano future-window compatibility metrics, plus model-output probes for tasks 2/5/7/8/10/11/12/13/14/15/16/17/18/19 and a derived task-20 boundary timing probe scored from held-out future-window artifacts. | 20/20 | 20 | 0 |

## Proxy-Scored Cells

| Task | Task label | Method | Metric | Reason |
| --- | --- | --- | --- | --- |
| 15 | Interaction Text Prediction | 128ep Raw Simple | macro_f1 | documented compact proxy completion for this raw128 task axis |
| 15 | Interaction Text Prediction | 128ep Raw NN | macro_f1 | documented compact proxy completion for this raw128 task axis |
| 19 | Camera-View Synchronization Retrieval | 128ep Aligned Simple | mrr | paired camera-view embeddings are absent from the 128 JSONL/feature export; metadata features retrieve the synchronized same-window depth/audio block as a documented compact synchronization proxy |
| 19 | Camera-View Synchronization Retrieval | 128ep Aligned NN | mrr | paired camera-view embeddings are absent from the 128 JSONL/feature export; metadata features retrieve the synchronized same-window depth/audio block as a documented compact synchronization proxy |
| 19 | Camera-View Synchronization Retrieval | 128ep Raw Simple | mrr | documented compact proxy completion for this raw128 task axis |
| 19 | Camera-View Synchronization Retrieval | 128ep Raw NN | mrr | documented compact proxy completion for this raw128 task axis |

## Reading Order

| Step | Reason |
| --- | --- |
| Choose the evidence line | Line 1 answers task-lab and reproducibility questions; line 2 answers selected-128 comparison questions. |
| Open the matching radar | Use the 1-episode radar for Minimal-vs-Neural behavior and the 128-episode radar for baseline/model-branch comparison. |
| Inspect the matrix row | Every numeric score is tied to a method, task, metric key, source artifact, and proxy flag. |
| Check proxy cells before interpreting totals | The six compact-proxy cells are numeric but are not direct raw-target measurements. |

## Reader Policy

- 1 sample episode: Use for task construction, raw-file inspection, local reproducibility, and controlled Minimal-vs-Neural baseline behavior.
- 128 selected episodes: Use for held-out comparison, metadata/raw-feature baselines, Qwen3/Cosmos branches, and scale-up decisions.
- Proxy scores: Proxy-scored cells stay numeric only when the source artifact and reason are attached; they should not be read as direct raw-target measurements.

# Two Evidence-Line Result Summary

Generated: `2026-06-21T07:36:39+00:00`.

Source matrix: [`docs/data/task_method_20_result_matrix.json`](docs/data/task_method_20_result_matrix.json)

Interpretation rule: Use the 1-episode line for task construction and reproducibility claims. Use the 128-episode line for held-out comparison and model-branch claims.

## Public Score Totals

- Lines: 2
- Tasks per method: 20
- Methods: 9
- Scored records: 180/180
- Direct scores: 174
- Compact-proxy scores: 6

## Line Ledger

| Line | Methods | Tasks | Scored records | Direct scores | Proxy scores | Method families |
| --- | --- | --- | --- | --- | --- | --- |
| 1 sample episode | 2 | 20 | 40/40 | 40 | 0 | Minimal, Neural MLP |
| 128 selected episodes | 7 | 20 | 140/140 | 134 | 6 | 128ep Aligned Simple, 128ep Aligned NN, 128ep Raw Simple, 128ep Raw NN, Qwen3-Omni v6 LoRA, Cosmos3-Super Reasoner, Cosmos3-Nano Future Window |

## Proxy-Scored Cells

| Task | Task label | Method | Metric | Reason |
| --- | --- | --- | --- | --- |
| 15 | Interaction Text Prediction | 128ep Raw Simple | macro_f1 | documented compact proxy completion for this raw128 task axis |
| 15 | Interaction Text Prediction | 128ep Raw NN | macro_f1 | documented compact proxy completion for this raw128 task axis |
| 19 | Camera-View Synchronization Retrieval | 128ep Aligned Simple | mrr | paired camera-view embeddings are absent from the 128 JSONL/feature export; metadata features retrieve the synchronized same-window depth/audio block as a documented compact synchronization proxy |
| 19 | Camera-View Synchronization Retrieval | 128ep Aligned NN | mrr | paired camera-view embeddings are absent from the 128 JSONL/feature export; metadata features retrieve the synchronized same-window depth/audio block as a documented compact synchronization proxy |
| 19 | Camera-View Synchronization Retrieval | 128ep Raw Simple | mrr | documented compact proxy completion for this raw128 task axis |
| 19 | Camera-View Synchronization Retrieval | 128ep Raw NN | mrr | documented compact proxy completion for this raw128 task axis |

## Reader Policy

- 1 sample episode: Use for task construction, raw-file inspection, local reproducibility, and controlled Minimal-vs-Neural baseline behavior.
- 128 selected episodes: Use for held-out comparison, metadata/raw-feature baselines, Qwen3/Cosmos branches, and scale-up decisions.
- Proxy scores: Proxy-scored cells stay numeric only when the source artifact and reason are attached; they should not be read as direct raw-target measurements.

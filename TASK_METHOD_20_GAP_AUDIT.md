# Task Method 20-Result Completion Audit

Generated: `2026-06-21T15:21:42+00:00`

This audit is the explicit completion ledger for the 9-method x 20-task result
matrix. The current public matrix is complete at 180/180 scored records while
preserving the rule that every numeric score needs a source artifact, and every
compact substitute target remains marked as a proxy.

## Score Summary

- Method-task records: `180`
- Numeric scored records: `180`
- Scoreless records: `0`
- Proxy-scored records: `6`
- Source matrix: [`docs/data/task_method_20_result_matrix.json`](docs/data/task_method_20_result_matrix.json)

## Method Coverage

| Method | ID | Scored | Scoreless | Proxy | Status counts |
| --- | --- | --- | --- | --- | --- |
| Minimal | minimal | 20/20 | 0 | 0 | scored: 20 |
| Neural MLP | neural_mlp | 20/20 | 0 | 0 | scored: 20 |
| 128ep Aligned Simple | metadata128_simple | 20/20 | 0 | 1 | proxy_scored: 1, scored: 19 |
| 128ep Aligned NN | metadata128_neural_mlp | 20/20 | 0 | 1 | proxy_scored: 1, scored: 19 |
| 128ep Raw Simple | raw128_simple | 20/20 | 0 | 2 | proxy_scored: 2, scored: 18 |
| 128ep Raw NN | raw128_neural_mlp | 20/20 | 0 | 2 | proxy_scored: 2, scored: 18 |
| Qwen3-Omni v6 LoRA | qwen3_omni_v6_lora | 20/20 | 0 | 0 | scored: 20 |
| Cosmos3-Super Reasoner | cosmos3_super_reasoner | 20/20 | 0 | 0 | scored: 20 |
| Cosmos3-Nano Future Window | cosmos3_nano_future_window | 20/20 | 0 | 0 | scored: 20 |

## Scoreless Classes

| Status | Count | Next step |
| --- | --- | --- |

## Scoreless Records

| Task | Task label | Method | Status | Required evidence |
| --- | --- | --- | --- | --- |

## Proxy Records

| Task | Task label | Method | Metric | Proxy note |
| --- | --- | --- | --- | --- |
| 15 | Interaction Text Prediction | 128ep Raw Simple | macro_f1 | documented compact proxy completion for this raw128 task axis |
| 15 | Interaction Text Prediction | 128ep Raw NN | macro_f1 | documented compact proxy completion for this raw128 task axis |
| 19 | Camera-View Synchronization Retrieval | 128ep Aligned Simple | mrr | paired camera-view embeddings are absent from the 128 JSONL/feature export; metadata features retrieve the synchronized same-window depth/audio block as a documented compact synchronization proxy |
| 19 | Camera-View Synchronization Retrieval | 128ep Aligned NN | mrr | paired camera-view embeddings are absent from the 128 JSONL/feature export; metadata features retrieve the synchronized same-window depth/audio block as a documented compact synchronization proxy |
| 19 | Camera-View Synchronization Retrieval | 128ep Raw Simple | mrr | documented compact proxy completion for this raw128 task axis |
| 19 | Camera-View Synchronization Retrieval | 128ep Raw NN | mrr | documented compact proxy completion for this raw128 task axis |

## Reproducibility Actions

- Keep [`docs/data/task_method_20_gap_audit.json`](docs/data/task_method_20_gap_audit.json) next to the radar and matrix so readers can distinguish direct scored rows from proxy-scored rows.
- Use [`scripts/omni/score_model_output_probes.py`](scripts/omni/score_model_output_probes.py) to rescore verified model outputs when stronger replacement artifacts arrive.
- Use [`scripts/omni/launch_all_task_model_scoring_when_free.sh`](scripts/omni/launch_all_task_model_scoring_when_free.sh) as the guarded waiter for future replacement scoring commands when private GPU capacity is available.

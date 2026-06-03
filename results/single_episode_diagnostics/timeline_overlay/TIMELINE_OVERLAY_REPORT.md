# Timeline Prediction Overlay Report

This report aligns existing prediction CSV files to the exported episode timeline. It does not rerun training.

## Task-Level Correctness

- Current Action Recognition: 10/343 correct (0.0292)
- Current Subtask Recognition: 20/344 correct (0.0581)
- Action Transition Detection: 322/348 correct (0.9253)
- Next-Action Prediction: 12/348 correct (0.0345)
- Contact State Prediction: 348/348 correct (1.0000)
- Relevant Object Prediction: 2/348 correct (0.0057)

## Files

- `timeline_overlay.csv`: prediction rows with frame positions.
- `timeline_overlay.svg`: visual overlay across the episode.

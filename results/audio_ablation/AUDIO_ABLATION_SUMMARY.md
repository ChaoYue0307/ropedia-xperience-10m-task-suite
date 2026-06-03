# Audio Ablation and Raw-Audio Upgrade

This report is generated from committed task-suite artifacts plus the local public-sample MP4 audio stream.
It measures whether audio changes each single-episode task under the same chronological split.

## Raw Audio Feature

- Source: `local_public_sample/fisheye_cam0.mp4`
- Has audio: `True`
- Sample rate: `16000`
- Window feature dim: `588`
- Feature: Per-window raw waveform STFT log-mel statistics plus delta and waveform envelope statistics.

## Task Deltas

| Task | Metric | Current audio | No audio | Current audio delta | Raw replaces audio | Raw replacement delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Current Action Recognition | macro_f1 | 0.0091 | 0.0088 | 0.0003 | 0.0013 | -0.0077 |
| Current Subtask Recognition | macro_f1 | 0.0113 | 0.0112 | 0.0001 | 0.0008 | -0.0104 |
| Action Transition Detection | macro_f1 | 0.4621 | 0.4687 | -0.0066 | 0.4792 | 0.0171 |
| Next-Action Prediction | macro_f1 | 0.0106 | 0.0107 | -0.0001 | 0.0060 | -0.0046 |
| Future Hand Motion Forecasting | mae | 4.4664 | 4.3038 | -0.1626 | 4.3059 | 0.1605 |
| Contact State Prediction | macro_f1 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| Relevant Object Prediction | micro_f1 | 0.1581 | 0.1479 | 0.0102 | 0.1787 | 0.0206 |
| Language-to-Time Grounding | mrr | 0.0321 | 0.0272 | 0.0049 | 0.0248 | -0.0072 |
| Cross-Modal Window Retrieval | mrr | 0.3751 | 0.3892 | -0.0141 | 0.3275 | -0.0476 |
| Sensor-to-Visual Reconstruction | mae | 9.7942 | 10.4467 | 0.6524 | 8.8307 | 0.9635 |
| Temporal Order Verification | macro_f1 | 0.5172 | 0.4943 | 0.0230 | 0.5302 | 0.0129 |
| Cross-Modal Misalignment Detection | macro_f1 | 0.4173 | 0.4226 | -0.0052 | 0.4438 | 0.0264 |

## Aggregate

- Mean current-audio delta: `0.041849794979543296`
- Tasks where current handcrafted audio improves the primary metric: `6`
- Mean raw-replacement delta vs current handcrafted audio: `0.09362598132150173`
- Tasks where raw log-mel replacement improves over current handcrafted audio: `6`

Positive deltas always mean better according to each task's primary metric. For MAE tasks, lower MAE is converted into a positive improvement.

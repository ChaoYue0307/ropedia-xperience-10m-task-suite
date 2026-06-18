# 128-Episode Sensor-Block Completion Tasks

This supplement fills task cells that cannot be produced by JSONL metadata alone but can be produced from the staged 4430-dim processed feature blocks on the staged GPU mirror.

| task | simple status | simple primary | neural status | neural primary |
| --- | --- | ---: | --- | ---: |
| Hand Trajectory Forecasting | pass | 8.8173 | pass | 0.4294 |
| Cross-Modal Retrieval | pass | 0.0026 | pass | 0.0026 |
| Cross-Modal Reconstruction | pass | -190.6611 | pass | -0.4348 |
| Multimodal Synchronization Detection | pass | 0.4998 | pass | 0.7774 |
| Imu To Hand Pose | pass | 0.2295 | pass | 0.2556 |

Still scoreless for this layer: `interaction_text_prediction` needs raw annotation interaction text, and `camera_view_sync_retrieval` needs paired per-camera feature embeddings.

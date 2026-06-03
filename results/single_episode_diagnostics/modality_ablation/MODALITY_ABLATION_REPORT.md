# Single-Episode Modality Ablation Report

This diagnostic reruns compact ridge heads on the exported one-episode feature matrix. It is useful for checking which real feature blocks can support each task on this episode, not for estimating dataset-wide generalization.

No synthetic labels are introduced. Derived proxy targets are marked in `target_variant`, and feature groups that overlap with the target source are marked in `target_source_overlap`.

## Best Computed Group Per Task

- Current Action Recognition: Language score=0.0278, macro_f1=0.0278, target overlap=false
- Current Subtask Recognition: Language score=0.0483, macro_f1=0.0483, target overlap=false
- Action Transition Detection: Language score=0.7052, macro_f1=0.7052, target overlap=false
- Next-Action Prediction: Language score=0.0419, macro_f1=0.0419, target overlap=false
- Future Hand Motion Forecasting: Inertial score=0.5679, mae=0.7608, target overlap=false
- Contact State Prediction: All Features score=1.0000, macro_f1=1.0000, target overlap=false
- Relevant Object Prediction: Language score=0.2302, micro_f1=0.2302, target overlap=true; best non-overlap: Depth score=0.2013, micro_f1=0.2013
- Language-to-Time Grounding: Language score=0.2453, mrr=0.2453, target overlap=true; best non-overlap: Motion Capture score=0.0306, mrr=0.0306
- Cross-Modal Window Retrieval: All Features score=0.9724, mrr=0.9724, target overlap=true; best non-overlap: Pose + SLAM score=0.4262, mrr=0.4262
- Sensor-to-Visual Reconstruction: Video score=0.6113, mae=0.6358, target overlap=true; best non-overlap: Pose + SLAM score=0.5359, mae=0.8659
- Temporal Order Verification: Pose + SLAM score=0.5259, macro_f1=0.5259, target overlap=false
- Cross-Modal Misalignment Detection: Video score=0.4949, macro_f1=0.4949, target overlap=false

## Files

- `ablation_metrics.csv`: every task/modality pair, including not-computed rows and reasons.
- `ablation_matrix.svg`: compact heatmap for manual inspection.
- `ablation_summary.json`: group dimensions and computed/not-computed counts.

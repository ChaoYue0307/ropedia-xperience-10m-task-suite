# Cross-Modal Alignment Stress Report

This diagnostic uses real held-out feature windows, then deliberately shifts the query modality in time at evaluation. The perturbation is derived; it is not treated as observed data.

## Zero-Shift Versus Worst Shift

- Audio: zero-shift MRR=0.0233; worst shift=40 windows, MRR=0.0132
- Inertial: zero-shift MRR=0.2840; worst shift=-20 windows, MRR=0.0199
- Language: zero-shift MRR=0.0310; worst shift=-40 windows, MRR=0.0158
- Motion Capture: zero-shift MRR=0.2553; worst shift=-10 windows, MRR=0.0183
- Motion + Pose + IMU: zero-shift MRR=0.3897; worst shift=-20 windows, MRR=0.0238
- Pose + SLAM: zero-shift MRR=0.4262; worst shift=-20 windows, MRR=0.0206

## Files

- `alignment_shift_metrics.csv`: MRR/rank metrics for each query group and time shift.
- `alignment_shift_curves.svg`: MRR curves across time shifts.
- `alignment_stress_summary.json`: perturbation definition and status.

# Single-Episode Diagnostics Index

These outputs are local diagnostics built from the existing one-episode Xperience-10M artifacts. They are designed for manual verification while waiting for full multi-episode data access.

## Generated Analyses

- `modality_ablation/`: compact ridge-head ablations across real feature blocks.
- `timeline_overlay/`: existing prediction CSVs aligned to the episode timeline.
- `alignment_stress/`: cross-modal retrieval under explicit time-shift perturbations.
- `provenance.json`: input hashes, feature dimensions, and source artifact identifiers.

## Validity Boundaries

- This is a single-episode diagnostic, not a full Xperience-10M benchmark.
- Rows marked `not_computed` are intentionally left blank when train labels or valid splits are unavailable.
- Rows marked `derived_perturbation` use real features with deliberate time shifts for stress testing.

## Counts

- Ablation rows: 108; computed: 108.
- Timeline overlay rows: 2079.
- Alignment stress rows: 54.
- Shared feature shape: 1161 windows x 8546 features.

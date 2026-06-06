# Verified Omni Fine-Tuning Result

- Backbone: `qwen3_omni_lora`
- Dataset run: `xperience10m_qwen3_omni_128ep_96train_16val_16test_valmon_20260605`
- Training run: `xperience10m_qwen3_omni_128ep_96train_16val_16test_valmon_20260605_lora`
- Evaluation run: `xperience10m_qwen3_omni_128ep_96train_16val_16test_valmon_20260605_eval`
- Validation status: `verified`
- Held-out eval split: `test`
- Held-out episodes: `14`
- Prediction rows: `448`

## Primary Metrics

- json_validity_rate: `0.875`
- action_macro_f1: `0.0026621494447581404`
- subtask_accuracy: `0.006696428571428571`
- transition_accuracy: `0.8504464285714286`
- next_action_accuracy: `0.024553571428571428`
- contact_accuracy: `0.6450892857142857`
- object_micro_f1: `0.22299431459254582`
- held_out_episode_count: `14`

Raw Xperience-10M files, base-model weights, adapter or checkpoint weights, full checkpoints, and large archives are not included.

## Error Analysis

The package includes a derived held-out error analysis under `analysis/`. It
groups the 448 public prediction rows by episode, coarse action family,
train-seen status, required-modality state, and object category.

Key readouts:

- Official JSON validity from `metrics.json`: `0.8750`
- Parsed prediction rate from public rows: `0.8772`
- Weakest action family by parsed prediction rate: `locomotion` with 23 rows and `0.2609`
- Train-seen split: seen labels have `0.0458` action exact rate; unseen labels have `0.0158`
- Required-modality state: all held-out rows have required modalities present, with only `visualization.rrd` absent

Use `analysis/ERROR_ANALYSIS.md` and
`analysis/error_analysis_summary.json` before planning the next
structured-output pass.

Use this package as the source for README, website, and Hugging Face updates.

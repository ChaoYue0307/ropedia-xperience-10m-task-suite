# Qwen3-Omni v5/v6 Verified Comparison

Generated: `2026-06-14`

This compares only the two dense multiscale Qwen3-Omni LoRA held-out packages on the same selected 128-episode setup. Both use 4,032 held-out test predictions from 14 exported test episodes.

| metric | v5 | v6 | v6 - v5 |
| --- | ---: | ---: | ---: |
| JSON validity | 1.000000 | 0.999008 | -0.000992 |
| Action macro-F1 | 0.002290 | 0.002883 | +0.000593 |
| Subtask accuracy | 0.011194 | 0.003731 | -0.007463 |
| Transition accuracy | 0.990823 | 0.989831 | -0.000992 |
| Next-action accuracy | 0.053619 | 0.043053 | -0.010565 |
| Contact accuracy | 0.786458 | 0.817708 | +0.031250 |
| Object micro-F1 | 0.316146 | 0.306498 | -0.009648 |

## Readout

v6 is the latest verified Qwen3-Omni LoRA branch and should be shown as the current Qwen row in generated comparisons. It improves action macro-F1 and contact accuracy. It does not dominate v5: v5 remains stronger on exact JSON validity, subtask accuracy, transition accuracy, next-action accuracy, and object micro-F1.

The public release policy is therefore:

- keep `ropedia-xperience-10m-v5` pinned to the previous stable v5 commit,
- publish v6 on `main`, GitHub Pages, HF Space, artifact dataset, and the Qwen LoRA model repo,
- create a separate `ropedia-xperience-10m-v6` tag only as an experimental/latest-Qwen release, not by moving the v5 tag.

## Sources

- v5 package: `results/omni_finetune/verified_public/xperience10m_qwen3_omni_128ep_multiscale_cap96_v5_full8gpu_lora_eval_test_full/verified_result_summary.json`
- v6 package: `results/omni_finetune/verified_public/xperience10m_qwen3_omni_128ep_multiscale_cap96_v6_rank64_lr5e5_full8gpu_lora_eval_test_full/verified_result_summary.json`
- machine-readable comparison: `docs/data/qwen3_v5_v6_comparison.json`

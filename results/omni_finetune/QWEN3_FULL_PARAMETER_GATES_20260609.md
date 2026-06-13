# Qwen3-Omni Full-Parameter Feasibility Gates

Generated: `2026-06-13T17:41:13+00:00`

The 2026-06-09 gates prove that Qwen3-Omni full-parameter FSDP can load, prepare, run backward/optimizer steps, and complete guarded pilots up to 128 optimizer steps on an 8-GPU remote worker. They do not prove a production full-parameter fine-tune, and they intentionally save no full checkpoints or public weights.

## Summary

- Status: `pass`
- Decision: `full_parameter_feasible_for_guarded_short_runs_not_promoted`
- Passed runs: `6`
- Preempted runs: `1`
- Review/missing runs: `0`
- Completed full-parameter optimizer steps: `489`
- Longest passed run: `xperience10m_qwen3_omni_128ep_fullparam_pilot256_after_qwen_v6_preemptible_8gpu_20260611` (256 steps)
- Checkpoint saved: `False`

## Runs

| run | status | steps | samples | final loss | epoch/train loss | policy | source |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| Full-Parameter 1-Step Feasibility Smoke | passed | 1 | 8 | 1.2726 | 1.2726 | no weights/checkpoints | `results/omni_finetune/xperience10m_qwen3_omni_128ep_fullparam_smoke_preemptible_8gpu_20260609/fullparam_feasibility_summary.json` |
| Full-Parameter 8-Step Short Train | passed | 8 | 64 | 1.1805 | 1.2190 | no weights/checkpoints | `results/omni_finetune/xperience10m_qwen3_omni_128ep_fullparam_shorttrain8_preemptible_8gpu_20260609/fullparam_shorttrain8_summary.json` |
| Full-Parameter 32-Step Pilot | passed | 32 | 256 | 0.2206 | 0.8451 | no weights/checkpoints | `results/omni_finetune/xperience10m_qwen3_omni_128ep_fullparam_pilot32_preemptible_8gpu_20260609/fullparam_pilot32_summary.json` |
| Full-Parameter 64-Step Pilot | passed | 64 | 512 | 0.0112 | 0.4434 | no weights/checkpoints | `results/omni_finetune/xperience10m_qwen3_omni_128ep_fullparam_pilot64_preemptible_8gpu_20260609/fullparam_pilot64_summary.json` |
| Full-Parameter 128-Step Opportunistic Pilot | preempted_for_qwen_v5_handoff | 0 | 1024 |  |  | no weights/checkpoints | `results/omni_finetune/xperience10m_qwen3_omni_128ep_fullparam_pilot128_preemptible_8gpu_20260609/fullparam_pilot128_summary.json` |
| Full-Parameter 128-Step Post-Qwen-v5 Pilot | passed | 128 | 1024 | 0.0137 | 0.2158 | no weights/checkpoints | `results/omni_finetune/xperience10m_qwen3_omni_128ep_fullparam_pilot128_after_qwen_v5_preemptible_8gpu_20260609/training_metadata.json` |
| Full-Parameter 256-Step Post-Qwen-v6 Pilot | passed | 256 | 2048 | 0.0096 | 0.1158 | no weights/checkpoints | `results/omni_finetune/xperience10m_qwen3_omni_128ep_fullparam_pilot256_after_qwen_v6_preemptible_8gpu_20260611/training_metadata.json` |

## Publication Policy

- Public summary allowed: `true`
- Publish full-parameter weights: `false`
- Publish full checkpoints: `false`
- Reason: All completed 2026-06-09 full-parameter runs used save_mode=none; the preempted pilot saved nothing. These are feasibility evidence only.

## Next Steps

- Keep the verified Qwen3-Omni LoRA adapter as the published production result for the 128-episode suite.
- For a production full-parameter run, add a sharded checkpoint/resume plan before any long training launch.
- Run a separate checkpointed full-parameter pilot only when GPUs are not needed by verified LoRA evaluation/publication work.

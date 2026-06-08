# Qwen3-Omni Full-Parameter Smoke Attempts

Date: 2026-06-08

These runs were bounded feasibility checks, not promoted model results. They did
not produce Qwen3 full-parameter checkpoints or public weights.

## Attempt 1

- Run id: `xperience10m_qwen3_omni_128ep_fullparam_smoke_8gpu_20260608`
- Trainer mode: `--tuning-mode full`
- Optimizer init: before `accelerator.prepare`
- Dataset: `xperience10m_qwen3_omni_128ep_96train_16val_16test_valmon_20260605_dataset/dataset.jsonl`
- Scope: 8 train samples, `--max-train-steps 1`, no validation, `--save-mode none`
- Observed progress: `setup_done`, `model_load_start`, `model_load_done`, `accelerator_prepare_start`
- Stop reason: manually stopped after prepare stayed CPU/RAM-bound with no `accelerator_prepare_done`, no `train_step`, and no meaningful GPU utilization.
- Peak observation before stop: host RAM about 467 GiB used, GPUs at about 2.5 GiB on rank 0 and 329 MiB on the other H20s.

## Attempt 2

- Run id: `xperience10m_qwen3_omni_128ep_fullparam_smoke_afterwrap_8gpu_20260608`
- Trainer mode: `--tuning-mode full`
- Optimizer init: after model prepare (`--optimizer-init after_model_prepare`)
- Dataset: same 128-episode structured JSON dataset
- Scope: 8 train samples, `--max-train-steps 1`, no validation, `--save-mode none`, 20-minute launcher timeout
- Observed progress: `setup_done`, `model_load_start`, `model_load_done`, `accelerator_prepare_start`
- Stop reason: manually stopped after prepare again stayed CPU/RAM-bound with no `accelerator_prepare_done`, no `train_step`, and no meaningful GPU utilization.
- Peak observation before stop: host RAM about 454 GiB used, GPUs at about 2.5 GiB on rank 0 and 329 MiB on the other H20s.

## Decision

Do not launch a production full-parameter Qwen3-Omni run with the current
trainer. The LoRA/FSDP path remains the active production path for Qwen v5. A
real full-parameter run needs a dedicated trainer/checkpoint plan, likely with
ZeRO-3 or a direct FSDP sharded-state workflow that proves model wrapping,
optimizer state allocation, backward, and checkpointing on a small smoke before
any long run is scheduled.

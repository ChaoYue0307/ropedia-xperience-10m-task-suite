# Model Output Probe Readiness

Generated: `2026-06-16T12:00:01+00:00`

This report checks whether verified model branches have the prediction files
needed to extend them to every 20-task contract. It is readiness evidence only;
it does not assign new task scores.

| Method | ID | Matrix scores | Status | Split files | Next step |
| --- | --- | --- | --- | --- | --- |
| Cosmos3-Nano Future Window | cosmos3_nano_future_window | 5/20 | missing_required_model_outputs | train: missing; validation: missing; test: missing | Collect or generate train, validation, and test prediction JSONL files first. |
| Cosmos3-Super Reasoner | cosmos3_super_reasoner | 6/20 | missing_required_model_outputs | train: missing; validation: missing; test: present | Collect or generate train, validation, and test prediction JSONL files first. |
| Qwen3-Omni v6 LoRA | qwen3_omni_v6_lora | 6/20 | missing_required_model_outputs | train: missing; validation: missing; test: present | Collect or generate train, validation, and test prediction JSONL files first. |

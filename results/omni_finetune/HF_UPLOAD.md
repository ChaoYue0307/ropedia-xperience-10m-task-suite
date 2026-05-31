# Hugging Face Upload (Model Artifact)

The current checkpoint available in this repo is the pilot run:

- `results/omni_finetune/adapter_lora/` (`xperience10m_qwen3_omni_32ep_lora`)
- Train windows: `128`
- Processes: `8`
- JSON output path: `results/omni_finetune/predictions_eval.jsonl`

Upload target layout:
- Source directory: `results/omni_finetune/hf_upload/`
- Upload script: `scripts/omni/upload_qwen3_omni_lora_to_hf.py`

Run (when network to huggingface.co is available):

```bash
HF_TOKEN=<your_token> python3 scripts/omni/upload_qwen3_omni_lora_to_hf.py \
  --repo-id cy0307/ropedia-qwen3-omni-lora-smoke \
  --source-dir results/omni_finetune/hf_upload \
  --message "Upload Xperience-10M Qwen3-Omni pilot LoRA"
```

If you want the repo private, add `--private`.

Note: this is a pilot artifact. The full 32-episode LoRA run is still blocked by
data availability; this artifact should not be reported as a full-scale result.

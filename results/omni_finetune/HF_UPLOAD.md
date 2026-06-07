# Hugging Face Upload (Qwen3-Omni LoRA Adapter)

The verified public result packages intentionally exclude adapter weights. LoRA
weights are uploaded separately as a model repository after the eval package and
audit pass.

Current intended target:

- Model repo: `cy0307/ropedia-qwen3-omni-lora-128ep`
- Source package: `results/omni_finetune/hf_upload_qwen3_128ep_full/`
- Package builder: `scripts/omni/prepare_qwen3_lora_hf_package.py`
- Upload script: `scripts/omni/upload_qwen3_omni_lora_to_hf.py`

Prepare the upload directory from the completed adapter and verified summary:

```bash
python3 scripts/omni/prepare_qwen3_lora_hf_package.py \
  --adapter-dir checkpoints/xperience10m_qwen3_omni_128ep_structured_json_v2_reuse_full8gpu_lora/adapter_lora \
  --verified-summary results/omni_finetune/verified_public/xperience10m_qwen3_omni_128ep_structured_json_v3_strict_label_prompt_reuse_lora_eval_test_full/verified_result_summary.json \
  --output-dir results/omni_finetune/hf_upload_qwen3_128ep_full \
  --repo-id cy0307/ropedia-qwen3-omni-lora-128ep
```

Upload when network and `HF_TOKEN` are available:

```bash
HF_TOKEN=<your_token> python3 scripts/omni/upload_qwen3_omni_lora_to_hf.py \
  --repo-id cy0307/ropedia-qwen3-omni-lora-128ep \
  --source-dir results/omni_finetune/hf_upload_qwen3_128ep_full \
  --message "Upload Xperience-10M 128-episode Qwen3-Omni LoRA adapter"
```

The prepared model repo includes PEFT adapter files, tokenizer/processor
sidecars when present, a generated model card, and `upload_manifest.json` with
file hashes. It must not include raw Xperience-10M MP4/HDF5/RRD files, Qwen base
weights, full FSDP checkpoints, or optimizer state.

Older smoke/pilot adapter material may still exist in
`results/omni_finetune/hf_upload/` and at
`cy0307/ropedia-qwen3-omni-lora-smoke`. Do not use those paths for the final
128-episode full-run publication.

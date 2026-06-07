# Cosmos3-Super Training Contract Audit

- Run id: `xperience10m_cosmos3_super_training_contract_audit_20260608`
- Dataset: `/home/cy/Ropedia/ropedia-episode-task-suite/results/omni_finetune/xperience10m_qwen3_omni_128ep_96train_16val_16test_valmon_20260605_dataset/dataset.jsonl`
- Rows: `3808`
- Rows with Cosmos action targets: `0`
- Valid Cosmos action targets: `0`
- Status: `blocked_missing_cosmos_action_targets`
- Weights updated: `False`

## Blockers

- dataset has no cosmos_action_target/cosmos3_action_target/action_target records; semantic JSON labels cannot be used as Cosmos continuous action latents

## Required Target Schema

```json
{
  "cosmos_action_target": {
    "mode": "policy|forward_dynamics|inverse_dynamics",
    "domain_name": "one Cosmos3 embodiment domain supported by CosmosActionCondition",
    "chunk_size": "positive integer action transition count",
    "raw_actions": "required for forward_dynamics; list[list[float]] with shape [T, raw_action_dim]",
    "video": "required for inverse_dynamics, or image/video conditioning for policy and forward_dynamics",
    "resolution_tier": "optional; one of 256, 480, 704, 720",
    "view_point": "optional; ego_view|third_person_view|wrist_view|concat_view"
  }
}
```

## Next Steps

- Export Cosmos-native action targets from Xperience annotations or mocap/pose/contact signals into the required cosmos_action_target schema.
- Run the one-sample action batch packer that calls Cosmos3OmniPipeline.prepare_latents and the static segment helpers, then records whether the exported target supervises vision or action tokens.
- For camera_pose forward_dynamics targets, use vision velocity/rectified-flow loss under action conditioning; add a policy/inverse target export before claiming supervised action-token prediction.
- Run a one-episode overfit before scheduling a 96/16/16 Super LoRA run; only publish a Cosmos model repo after new adapter/checkpoint weights exist.

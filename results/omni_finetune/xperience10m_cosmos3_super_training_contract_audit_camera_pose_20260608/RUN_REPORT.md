# Cosmos3-Super Training Contract Audit

- Run id: `xperience10m_cosmos3_super_training_contract_audit_camera_pose_20260608`
- Dataset: `/home/cy/Ropedia/ropedia-episode-task-suite/results/omni_finetune/xperience10m_cosmos3_camera_pose_targets_20260608/dataset_with_cosmos_actions.jsonl`
- Rows: `3808`
- Rows with Cosmos action targets: `3808`
- Valid Cosmos action targets: `3808`
- Status: `ready_for_cosmos3_super_forward_dynamics_lora`
- Weights updated: `False`

## Blockers

- None

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

- Run the one-sample action batch packer that calls Cosmos3OmniPipeline.prepare_latents and the static segment helpers, then records whether the current target supervises vision or action tokens.
- For the current camera_pose forward_dynamics target, implement a one-sample overfit with vision velocity/rectified-flow loss under action conditioning; add a policy/inverse target export before claiming supervised action-token prediction.
- Run a one-episode overfit before scheduling a 96/16/16 Super LoRA run; only publish a Cosmos model repo after new adapter/checkpoint weights exist.

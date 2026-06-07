# Cosmos3-Super Action Batch Packer

- Run id: `xperience10m_cosmos3_super_action_packer_schema_smoke_20260608`
- Row: `27c9fc42-2bb4-4737-b09c-08d2dd88aed4__ep4:qa:0`
- Mode: `forward_dynamics`
- Domain: `camera_pose`
- Raw action shape: `[8, 9]`
- Pipeline loaded: `False`
- Status: `pass`

## Loss Surface

- `vision_velocity_conditioned_on_camera_pose`
- Cosmos3 forward_dynamics consumes raw_actions as conditioning and predicts noisy vision tokens. It does not supervise preds_action for this target mode.

## Next Step

- Implement the one-sample overfit with a vision velocity/rectified-flow loss under camera-pose action conditioning.
- Add a separate policy or inverse-dynamics target export before claiming supervised action-token prediction.

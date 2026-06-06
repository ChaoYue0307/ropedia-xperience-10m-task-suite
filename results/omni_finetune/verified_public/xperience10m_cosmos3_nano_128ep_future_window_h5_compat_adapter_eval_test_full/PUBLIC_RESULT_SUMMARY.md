# Verified Omni Fine-Tuning Result

- Backbone: `cosmos_world_model`
- Dataset run: `xperience10m_cosmos3_nano_128ep_future_window_h5_compat`
- Training run: `xperience10m_cosmos3_nano_128ep_future_window_h5_compat_adapter`
- Evaluation run: `xperience10m_cosmos3_nano_128ep_future_window_h5_compat_adapter_eval_test_full`
- Validation status: `verified`
- Held-out eval split: `test`
- Held-out episodes: `14`
- Prediction rows: `378`

## Primary Metrics

- future_retrieval_mrr: `0.022138720585222767`
- future_retrieval_recall_at_5: `0.015873015873015872`
- temporal_consistency: `0.09523809523809523`
- feature_reconstruction_error: `3479.218317102503`
- transition_accuracy: `0.9682539682539683`
- contact_accuracy: `0.7433862433862434`
- held_out_episode_count: `14`

Raw Xperience-10M files, base-model weights, adapter or checkpoint weights, full checkpoints, and large archives are not included.

Use this package as the source for README, website, and Hugging Face updates.

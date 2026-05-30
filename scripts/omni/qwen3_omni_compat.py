#!/usr/bin/env python3
"""Runtime compatibility shims for local Qwen3-Omni Transformers builds."""

from __future__ import annotations


def patch_qwen3_omni_config() -> None:
    """Patch a Transformers 5.0.0 Qwen3-Omni config init ordering bug.

    The local build reads `self.use_sliding_window` inside
    `Qwen3OmniMoeTalkerCodePredictorConfig.__init__` before `PreTrainedConfig`
    has copied unknown keyword arguments onto the instance. ModelScope's Instruct
    config includes `use_sliding_window`, but the attribute is not available yet.
    Setting it before the original initializer keeps the upstream behavior while
    allowing local `from_pretrained()` to proceed.
    """

    try:
        from transformers.models.qwen3_omni_moe import configuration_qwen3_omni_moe as cfg
    except Exception:
        return

    cls = getattr(cfg, "Qwen3OmniMoeTalkerCodePredictorConfig", None)
    if cls is None or getattr(cls, "_ropedia_use_sliding_window_patch", False):
        return

    original_init = cls.__init__

    def patched_init(self, *args, **kwargs):
        self.use_sliding_window = bool(kwargs.get("use_sliding_window", False))
        original_init(self, *args, **kwargs)

    cls.__init__ = patched_init
    cls._ropedia_use_sliding_window_patch = True

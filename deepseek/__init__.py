"""A small, readable, from-scratch implementation of the DeepSeek-V3 architecture.

Modules:
    config      -- DeepSeekV3Config (scaled-down defaults; real values in comments)
    norm        -- RMSNorm
    rope        -- RotaryEmbedding + apply_rotary (decoupled RoPE)
    mlp         -- MLP (SwiGLU feed-forward / single expert)
    attention   -- MLA (Multi-head Latent Attention)
    moe         -- MoEGate + DeepSeekMoE (shared + routed experts, aux-loss-free)
    layer       -- DecoderLayer (dense or MoE transformer block)
    mtp         -- MTPModule (Multi-Token Prediction)
    model       -- DeepSeekV3Model (full stack + lm head + MTP)
"""

from .config import DeepSeekV3Config
from .model import DeepSeekV3Model

__all__ = ["DeepSeekV3Config", "DeepSeekV3Model"]

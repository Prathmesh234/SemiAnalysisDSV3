"""Rotary Position Embedding (RoPE).

In DeepSeek-V3's MLA, RoPE is applied only to a small *decoupled* slice of the
query/key (`qk_rope_head_dim`), not to the whole head. The helpers here are the
standard rotary embedding used by Llama-style models.
"""

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """Precomputes cos/sin tables for the decoupled RoPE dimension."""

    def __init__(self, dim: int, base: float = 10000.0, max_position: int = 2048):
        super().__init__()
        # inv_freq: (dim/2,)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_position = max_position

    def forward(self, seq_len: int, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))   # (seq_len, dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)             # (seq_len, dim)
        return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates the two halves of the last dimension: [-x2, x1]."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _deinterleave(x: torch.Tensor) -> torch.Tensor:
    """Reorder interleaved rotary dims [x0,x1,x2,x3,...] -> [x0,x2,...,x1,x3,...].

    DeepSeek-V3 does exactly this (`view(b,h,s,d//2,2).transpose(4,3).reshape`)
    before applying cos/sin, so the half-split `rotate_half` lands on the right
    frequency pairs. Matching it keeps us bit-compatible with the reference.
    """
    b, h, s, d = x.shape
    return x.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to `x` of shape (batch, heads, seq, rope_dim).

    cos/sin are (seq, rope_dim); broadcast over batch and head dims.
    """
    x = _deinterleave(x)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos + rotate_half(x) * sin

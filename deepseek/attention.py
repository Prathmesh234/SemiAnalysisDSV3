"""Multi-head Latent Attention (MLA) — DeepSeek-V3's attention.

The core idea: instead of storing full keys/values, MLA compresses them into a
small *latent* vector (`kv_lora_rank`) that is up-projected per head on the fly.
The query is likewise produced from a low-rank latent (`q_lora_rank`). This
shrinks the KV cache dramatically.

Each head's query/key is split into two parts:
  * a "nope" (content) part  -> NO position info        (qk_nope_head_dim)
  * a "rope" (decoupled) part -> carries RoPE position   (qk_rope_head_dim)
They are concatenated before the attention dot-product. The decoupled RoPE key
is shared across all heads (computed once, then broadcast).

For simplicity this implements full-sequence (no KV-cache) causal attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DeepSeekV3Config
from .norm import RMSNorm
from .rope import RotaryEmbedding, apply_rotary


class MLA(nn.Module):
    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = config.qk_head_dim          # nope + rope
        self.v_head_dim = config.v_head_dim
        self.kv_lora_rank = config.kv_lora_rank

        # --- Query path: hidden -> latent -> per-head q (nope + rope) -------
        self.q_down_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.q_up_proj = nn.Linear(
            config.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False
        )

        # --- Key/Value path: hidden -> [latent | shared rope key] ----------
        self.kv_down_proj = nn.Linear(
            config.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=False
        )
        self.kv_norm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        # Up-projects the latent into per-head [k_nope | value].
        self.kv_up_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        # --- Output projection ---------------------------------------------
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, config.hidden_size, bias=False)

        self.rotary = RotaryEmbedding(
            self.qk_rope_head_dim, base=config.rope_theta,
            max_position=config.max_position_embeddings,
        )
        self.softmax_scale = self.qk_head_dim ** -0.5

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, T, _ = hidden_states.shape

        # ---- Query: down -> norm -> up -> split into nope/rope ------------
        q = self.q_up_proj(self.q_norm(self.q_down_proj(hidden_states)))
        q = q.view(B, T, self.num_heads, self.qk_head_dim).transpose(1, 2)
        q_nope, q_rope = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )  # (B, H, T, nope), (B, H, T, rope)

        # ---- Key/Value: down -> split latent & shared rope key ------------
        kv = self.kv_down_proj(hidden_states)
        c_kv, k_rope = torch.split(
            kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        # Shared decoupled key across heads: (B, 1, T, rope)
        k_rope = k_rope.view(B, T, 1, self.qk_rope_head_dim).transpose(1, 2)

        kv = self.kv_up_proj(self.kv_norm(c_kv))
        kv = kv.view(B, T, self.num_heads, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)
        k_nope, value = torch.split(
            kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )  # (B, H, T, nope), (B, H, T, v)

        # ---- Apply RoPE to the decoupled parts only -----------------------
        cos, sin = self.rotary(T, hidden_states.device, hidden_states.dtype)
        q_rope = apply_rotary(q_rope, cos, sin)
        k_rope = apply_rotary(k_rope, cos, sin)
        k_rope = k_rope.expand(B, self.num_heads, T, self.qk_rope_head_dim)

        # ---- Assemble full query/key and run attention --------------------
        query = torch.cat([q_nope, q_rope], dim=-1)   # (B, H, T, qk_head_dim)
        key = torch.cat([k_nope, k_rope], dim=-1)     # (B, H, T, qk_head_dim)

        scores = torch.matmul(query, key.transpose(-1, -2)) * self.softmax_scale
        causal = torch.triu(
            torch.full((T, T), float("-inf"), device=scores.device), diagonal=1
        )
        scores = scores + causal
        attn = F.softmax(scores.float(), dim=-1).to(query.dtype)

        out = torch.matmul(attn, value)               # (B, H, T, v_head_dim)
        out = out.transpose(1, 2).reshape(B, T, self.num_heads * self.v_head_dim)
        return self.o_proj(out)

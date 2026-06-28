"""A single DeepSeek-V3 decoder layer (pre-norm transformer block).

The only thing that varies between layers is whether the feed-forward block is a
dense MLP or a DeepSeekMoE. The first `first_k_dense_replace` layers are dense;
the rest are MoE.

    h = h + MLA(input_layernorm(h))
    h = h + FFN(post_attention_layernorm(h))     # FFN = dense MLP or MoE
"""

import torch
import torch.nn as nn

from .config import DeepSeekV3Config
from .norm import RMSNorm
from .attention import MLA
from .mlp import MLP
from .moe import DeepSeekMoE


class DecoderLayer(nn.Module):
    def __init__(self, config: DeepSeekV3Config, layer_idx: int):
        super().__init__()
        self.self_attn = MLA(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.is_moe = layer_idx >= config.first_k_dense_replace
        if self.is_moe:
            self.mlp = DeepSeekMoE(config)
        else:
            self.mlp = MLP(config.hidden_size, config.intermediate_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(self.input_layernorm(hidden_states))
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states

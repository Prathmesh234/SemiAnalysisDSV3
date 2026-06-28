"""Multi-Token Prediction (MTP) module.

A training-time objective: in addition to predicting the next token, the model
predicts a few *additional* future tokens via sequential MTP modules. Each module
reuses the main model's token embedding and output head (shared), and runs one
full transformer block.

For the k-th future token, given the previous depth's hidden state `h_prev` (for
position i) and the embedding of the (i+k)-th token:

    proj_in = M_k · [ RMSNorm(h_prev) ; RMSNorm(emb(t_{i+k})) ]   # concat -> 2*hidden -> hidden
    h_k     = TransformerBlock(proj_in)
    logits  = shared_head(final_norm(h_k))

This module is dropped for plain inference; we include it so the architecture is
complete and to show the MTP forward pass.
"""

import torch
import torch.nn as nn

from .config import DeepSeekV3Config
from .norm import RMSNorm
from .layer import DecoderLayer


class MTPModule(nn.Module):
    def __init__(self, config: DeepSeekV3Config, layer_idx: int):
        super().__init__()
        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.embed_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # M_k: concat(prev_hidden, token_embed) -> hidden
        self.proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        # A standard transformer block (uses MoE since these are deep layers).
        self.block = DecoderLayer(config, layer_idx)

    def forward(self, prev_hidden: torch.Tensor, token_embeds: torch.Tensor) -> torch.Tensor:
        """prev_hidden, token_embeds: (B, T, hidden). Returns (B, T, hidden)."""
        x = torch.cat(
            [self.hidden_norm(prev_hidden), self.embed_norm(token_embeds)], dim=-1
        )
        x = self.proj(x)
        return self.block(x)

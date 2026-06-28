"""The full DeepSeek-V3 model (scaled down, architecture-faithful).

Pipeline:
    tokens -> embed -> [DecoderLayer x N] -> RMSNorm -> lm_head -> logits

The model also exposes the Multi-Token Prediction (MTP) heads, which share the
token embedding and the output head with the main model. Call `forward(...,
return_mtp=True)` to also get the extra-token predictions.
"""

import torch
import torch.nn as nn

from .config import DeepSeekV3Config
from .norm import RMSNorm
from .layer import DecoderLayer
from .mtp import MTPModule


class DeepSeekV3Model(nn.Module):
    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            DecoderLayer(config, i) for i in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # MTP modules predict extra future tokens; they share embed + lm_head.
        self.mtp_modules = nn.ModuleList(
            MTPModule(config, layer_idx=config.num_hidden_layers + k)
            for k in range(config.num_mtp_modules)
        )

    def backbone(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run embedding + decoder stack, return final pre-head hidden states."""
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states

    def forward(self, input_ids: torch.Tensor, return_mtp: bool = False):
        """input_ids: (B, T) -> logits (B, T, vocab).

        If `return_mtp`, also returns a list of (B, T, vocab) logits, one per MTP
        module, each predicting one additional future token.
        """
        hidden_states = self.backbone(input_ids)
        logits = self.lm_head(self.norm(hidden_states))

        if not return_mtp:
            return logits

        mtp_logits = []
        prev_hidden = hidden_states
        for k, mtp in enumerate(self.mtp_modules, start=1):
            # Module k predicts the token k steps further ahead: feed the
            # embedding of the input shifted left by k (zero-padded at the end).
            shifted = torch.roll(input_ids, shifts=-k, dims=1)
            shifted[:, -k:] = 0
            token_embeds = self.embed_tokens(shifted)
            prev_hidden = mtp(prev_hidden, token_embeds)
            mtp_logits.append(self.lm_head(self.norm(prev_hidden)))

        return logits, mtp_logits

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

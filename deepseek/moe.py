"""DeepSeekMoE — fine-grained Mixture of Experts with shared experts.

Two design choices that distinguish DeepSeek-V3:

1. **Shared + routed experts.** A small set of *shared* experts always runs for
   every token (capturing common knowledge), plus a large pool of *routed*
   experts of which only `num_experts_per_tok` are selected per token.

2. **Auxiliary-loss-free load balancing.** Routing affinities are produced by a
   `sigmoid` gate. A per-expert bias (`e_score_correction_bias`) is added *only*
   for the top-k selection decision — the gate weights actually used to combine
   experts come from the unbiased sigmoid scores. The bias is nudged during
   training to balance load, so no auxiliary loss is needed (here it is just a
   learnable/zero buffer since we only do a forward pass).

Routing is also *group-limited*: experts are partitioned into `n_group` groups,
each token first picks the `topk_group` strongest groups, then picks its experts
from within those groups only.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DeepSeekV3Config
from .mlp import MLP


class MoEGate(nn.Module):
    """Computes which routed experts each token goes to, and with what weight."""

    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor

        # Router weights: hidden -> per-expert affinity logit.
        self.weight = nn.Parameter(torch.empty(self.n_routed_experts, config.hidden_size))
        nn.init.normal_(self.weight, std=0.02)
        # Aux-loss-free balancing bias (selection-only; not trained by grad here).
        self.register_buffer(
            "e_score_correction_bias", torch.zeros(self.n_routed_experts)
        )

    def forward(self, x: torch.Tensor):
        # x: (n_tokens, hidden)
        logits = F.linear(x.float(), self.weight.float())   # (n, n_experts)
        scores = logits.sigmoid()                           # V3 uses sigmoid

        # ---- bias added for SELECTION only --------------------------------
        scores_for_choice = scores + self.e_score_correction_bias

        # ---- group-limited routing ----------------------------------------
        n = x.shape[0]
        experts_per_group = self.n_routed_experts // self.n_group
        grouped = scores_for_choice.view(n, self.n_group, experts_per_group)
        # Group strength = sum of its top-2 experts.
        group_scores = grouped.topk(min(2, experts_per_group), dim=-1)[0].sum(-1)
        group_idx = group_scores.topk(self.topk_group, dim=-1)[1]      # (n, topk_group)
        group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1.0)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(n, self.n_group, experts_per_group)
            .reshape(n, self.n_routed_experts)
        )
        masked = scores_for_choice.masked_fill(score_mask == 0, float("-inf"))

        # ---- top-k expert selection ---------------------------------------
        topk_idx = masked.topk(self.top_k, dim=-1)[1]        # (n, top_k)
        # Gate weights come from the UNBIASED sigmoid scores.
        topk_weight = scores.gather(1, topk_idx)

        if self.top_k > 1 and self.norm_topk_prob:
            topk_weight = topk_weight / (topk_weight.sum(-1, keepdim=True) + 1e-20)
        topk_weight = topk_weight * self.routed_scaling_factor
        return topk_idx, topk_weight.to(x.dtype)


class DeepSeekMoE(nn.Module):
    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.gate = MoEGate(config)
        self.experts = nn.ModuleList(
            MLP(config.hidden_size, config.moe_intermediate_size)
            for _ in range(config.n_routed_experts)
        )
        # Shared expert(s): always applied, intermediate scaled by count.
        self.shared_experts = MLP(
            config.hidden_size,
            config.moe_intermediate_size * config.n_shared_experts,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, T, H = hidden_states.shape
        x = hidden_states.view(-1, H)                # (n_tokens, hidden)
        topk_idx, topk_weight = self.gate(x)         # (n, k), (n, k)

        # Combine the selected routed experts (simple, readable loop over k).
        out = torch.zeros_like(x)
        flat_idx = topk_idx.view(-1)                 # (n*k,)
        flat_w = topk_weight.view(-1, 1)             # (n*k, 1)
        token_ids = torch.arange(x.shape[0], device=x.device).repeat_interleave(self.top_k)
        for e, expert in enumerate(self.experts):
            mask = flat_idx == e
            if not mask.any():
                continue
            sel_tokens = token_ids[mask]
            out.index_add_(0, sel_tokens, expert(x[sel_tokens]) * flat_w[mask])

        # Shared expert runs for every token and is added ungated.
        out = out + self.shared_experts(x)
        return out.view(B, T, H)

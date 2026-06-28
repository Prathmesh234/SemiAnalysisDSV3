"""Configuration for the (scaled-down) DeepSeek-V3 model.

The real DeepSeek-V3 is 671B parameters: hidden_size=7168, 61 layers, 256
routed experts, 128 attention heads, etc. Those numbers are kept here as
comments for reference. The defaults below are a small, runnable version that
keeps every architectural feature (MLA, MoE, MTP) intact so you can read and
run the code on a laptop CPU.
"""

from dataclasses import dataclass, field


@dataclass
class DeepSeekV3Config:
    # --- Vocab / embedding -------------------------------------------------
    vocab_size: int = 1024              # real: 129280

    # --- Transformer backbone ---------------------------------------------
    hidden_size: int = 256              # real: 7168
    num_hidden_layers: int = 4          # real: 61 (we use 1 dense + 3 MoE)
    num_attention_heads: int = 8        # real: 128

    # --- Multi-head Latent Attention (MLA) --------------------------------
    q_lora_rank: int = 96               # real: 1536  (query down-projection dim)
    kv_lora_rank: int = 64              # real: 512   (key/value latent dim)
    qk_nope_head_dim: int = 32          # real: 128   (per-head content dim)
    qk_rope_head_dim: int = 16          # real: 64    (per-head decoupled RoPE dim)
    v_head_dim: int = 32                # real: 128   (per-head value dim)

    # --- Mixture of Experts (DeepSeekMoE) ---------------------------------
    intermediate_size: int = 512        # real: 18432 (dense FFN hidden size)
    moe_intermediate_size: int = 128    # real: 2048  (per-expert hidden size)
    n_routed_experts: int = 8           # real: 256
    n_shared_experts: int = 1           # real: 1
    num_experts_per_tok: int = 2        # real: 8     (top-k routed experts)
    first_k_dense_replace: int = 1      # real: 3     (first N layers are dense)
    n_group: int = 2                    # real: 8     (expert groups for routing)
    topk_group: int = 1                 # real: 4     (groups kept per token)
    routed_scaling_factor: float = 2.5  # real: 2.5
    norm_topk_prob: bool = True

    # --- Multi-Token Prediction (MTP) -------------------------------------
    num_mtp_modules: int = 1            # real: 1 (predict 1 extra token)

    # --- Normalisation / RoPE ---------------------------------------------
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    max_position_embeddings: int = 2048

    def __post_init__(self):
        assert self.num_attention_heads > 0
        assert self.first_k_dense_replace <= self.num_hidden_layers

    @property
    def qk_head_dim(self) -> int:
        """Total per-head query/key dim = content (nope) + decoupled RoPE."""
        return self.qk_nope_head_dim + self.qk_rope_head_dim

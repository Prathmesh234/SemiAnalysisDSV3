"""Configuration for the DeepSeek-V3 model.

These are the real DeepSeek-V3 (671B) defaults, taken from the official
`config.json` (deepseek-ai/DeepSeek-V3). Instantiating a full model from these
needs many GPUs — they describe the production architecture, not a CPU demo.
"""

from dataclasses import dataclass


@dataclass
class DeepSeekV3Config:
    # --- Vocab / embedding -------------------------------------------------
    vocab_size: int = 129280

    # --- Transformer backbone ---------------------------------------------
    hidden_size: int = 7168
    num_hidden_layers: int = 61
    num_attention_heads: int = 128

    # --- Multi-head Latent Attention (MLA) --------------------------------
    q_lora_rank: int = 1536             # query down-projection (latent) dim
    kv_lora_rank: int = 512             # key/value latent dim
    qk_nope_head_dim: int = 128         # per-head content (no-position) dim
    qk_rope_head_dim: int = 64          # per-head decoupled RoPE dim
    v_head_dim: int = 128               # per-head value dim

    # --- Mixture of Experts (DeepSeekMoE) ---------------------------------
    intermediate_size: int = 18432      # dense FFN hidden size
    moe_intermediate_size: int = 2048   # per-expert hidden size
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8        # top-k routed experts
    first_k_dense_replace: int = 3      # first N layers are dense FFN
    n_group: int = 8                    # expert groups for routing
    topk_group: int = 4                 # groups kept per token
    routed_scaling_factor: float = 2.5
    norm_topk_prob: bool = True

    # --- Multi-Token Prediction (MTP) -------------------------------------
    num_mtp_modules: int = 1            # num_nextn_predict_layers

    # --- Normalisation / RoPE ---------------------------------------------
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    max_position_embeddings: int = 163840   # 4096 base, extended via YaRN

    def __post_init__(self):
        assert self.num_attention_heads > 0
        assert self.first_k_dense_replace <= self.num_hidden_layers

    @property
    def qk_head_dim(self) -> int:
        """Total per-head query/key dim = content (nope) + decoupled RoPE."""
        return self.qk_nope_head_dim + self.qk_rope_head_dim

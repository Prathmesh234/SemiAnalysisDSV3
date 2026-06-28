# DeepSeek-V3 (from scratch, scaled down)

A small, readable PyTorch implementation of the DeepSeek-V3 architecture
([technical report](https://arxiv.org/abs/2412.19437), reference code
[`modeling_deepseek.py`](https://huggingface.co/deepseek-ai/DeepSeek-V3)).

Every architectural feature is present; only the sizes are shrunk so it runs on a
laptop CPU. The real config values are kept as comments in `config.py`.

## Run

```bash
uv run run_deepseek.py
```

## Modules

| File | What it implements |
|------|--------------------|
| `config.py`    | `DeepSeekV3Config` — small defaults, real values in comments |
| `norm.py`      | **RMSNorm** (fp32, no bias) |
| `rope.py`      | **Rotary embedding** for the decoupled RoPE slice |
| `mlp.py`       | **SwiGLU** FFN — used for dense layers and as each expert |
| `attention.py` | **MLA** — Multi-head Latent Attention |
| `moe.py`       | **DeepSeekMoE** — sigmoid gate, aux-loss-free bias, grouped routing, shared+routed experts |
| `layer.py`     | **DecoderLayer** — pre-norm block (dense or MoE FFN) |
| `mtp.py`       | **MTP** — Multi-Token Prediction module |
| `model.py`     | **DeepSeekV3Model** — embed → layers → norm → head (+ MTP) |

## The three things that make it DeepSeek-V3

**1. MLA (Multi-head Latent Attention).** Keys/values are compressed to a small
latent (`kv_lora_rank`) and up-projected per head on the fly; the query comes
from its own latent (`q_lora_rank`). Each head's query/key = a content (`nope`)
part with no position info + a small decoupled `rope` part that carries RoPE.
This is what shrinks the KV cache.

**2. DeepSeekMoE.** Always-on *shared* experts + a large pool of *routed* experts,
of which only `num_experts_per_tok` run per token. Routing uses a **sigmoid**
gate with **auxiliary-loss-free** load balancing: a per-expert bias is added
*only* for the top-k selection decision, while the combination weights come from
the unbiased scores. Routing is group-limited (`n_group`/`topk_group`).

**3. MTP (Multi-Token Prediction).** Extra heads predict additional future tokens
during training, sharing the embedding and output head. Dropped at inference.

## Layer layout

The first `first_k_dense_replace` layers use a dense SwiGLU FFN; the rest are MoE
— matching the real model (3 dense + 58 MoE there; here 1 dense + 3 MoE, since
"just do a couple of each variant").

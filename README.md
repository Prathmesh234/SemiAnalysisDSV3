# SemiAnalysisDSV3

A small, readable, from-scratch PyTorch implementation of the **DeepSeek-V3**
model architecture — MLA attention, DeepSeekMoE, and Multi-Token Prediction —
scaled down so it runs on a laptop CPU. The real config values are preserved as
comments alongside the small defaults.

## Quick start

```bash
uv sync
uv run run_deepseek.py               # build the model + run a forward pass
uv run tests/test_architecture.py    # verify architecture invariants
```

## Layout

- `deepseek/` — the model, one module per component (see `deepseek/README.md`)
- `run_deepseek.py` — demo forward pass
- `tests/test_architecture.py` — behavioral invariant checks (causality, MLA, MoE routing, RoPE)

## What it implements

| Feature | Where |
|---------|-------|
| Multi-head Latent Attention (MLA) | `deepseek/attention.py` |
| DeepSeekMoE (shared + routed experts, aux-loss-free gating) | `deepseek/moe.py` |
| Multi-Token Prediction (MTP) | `deepseek/mtp.py` |
| RMSNorm / RoPE / SwiGLU | `deepseek/{norm,rope,mlp}.py` |

Based on the DeepSeek-V3 technical report ([arXiv:2412.19437](https://arxiv.org/abs/2412.19437))
and the reference `modeling_deepseek.py`.

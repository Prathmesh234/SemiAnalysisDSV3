# SemiAnalysisDSV3

A small, readable, from-scratch implementation of the **DeepSeek-V3** model
architecture — MLA attention, DeepSeekMoE, and Multi-Token Prediction — in a
single Jupyter notebook.

## Notebook

[`deepseek_v3.ipynb`](deepseek_v3.ipynb) — the whole model, one component per
section, runnable top to bottom:

1. Config (real DeepSeek-V3 values)
2. RMSNorm
3. RoPE (decoupled rotary)
4. SwiGLU MLP / expert
5. Multi-head Latent Attention (MLA)
6. DeepSeekMoE (shared + routed experts, aux-loss-free gating)
7. Decoder layer (dense / MoE)
8. Multi-Token Prediction (MTP)
9. Full model
10. Sanity check — forward pass on a small config

The config holds the real DeepSeek-V3 (671B) values; the final cell shrinks them
with `dataclasses.replace` so the forward pass runs on CPU.

## Run

```bash
uv sync
uv run jupyter lab deepseek_v3.ipynb   # or open in VS Code / Colab
```

Based on the DeepSeek-V3 technical report ([arXiv:2412.19437](https://arxiv.org/abs/2412.19437))
and the reference `modeling_deepseek.py`.

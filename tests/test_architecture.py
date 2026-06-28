"""Behavioral invariant tests for the DeepSeek-V3 implementation.

These don't check weight values (the model is random) — they check that the
*architecture* behaves the way the design dictates:

  1. Causality           — position t's output never depends on tokens > t.
  2. MLA shapes/scale     — latent compression and head dims are wired correctly.
  3. MoE routing          — exactly top-k experts per token, group-limit respected,
                            gate weights normalized * routed_scaling_factor, shared
                            expert always contributes.
  4. RoPE relative-shift  — apply_rotary preserves the q·k relative-position identity.
  5. Dense/MoE layout     — first_k_dense_replace layers dense, rest MoE.

Run:  uv run tests/test_architecture.py
"""

import torch

from deepseek import DeepSeekV3Config, DeepSeekV3Model
from deepseek.moe import MoEGate
from deepseek.rope import RotaryEmbedding, apply_rotary


def test_causality():
    """Changing token at position p must not change logits at positions < p."""
    torch.manual_seed(0)
    cfg = DeepSeekV3Config()
    model = DeepSeekV3Model(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 12))

    with torch.no_grad():
        base = model(ids)
        p = 8
        ids2 = ids.clone()
        ids2[0, p] = (ids2[0, p] + 1) % cfg.vocab_size  # perturb position p
        changed = model(ids2)

    # Positions before p must be identical; position p onward may differ.
    diff_before = (base[:, :p] - changed[:, :p]).abs().max().item()
    diff_after = (base[:, p:] - changed[:, p:]).abs().max().item()
    assert diff_before < 1e-5, f"causality broken: {diff_before}"
    assert diff_after > 1e-6, "perturbation had no effect at/after p (mask too strong)"
    print(f"[causality]      ok  (max diff before p = {diff_before:.2e}, after = {diff_after:.2e})")


def test_mla_shapes():
    """MLA must compress KV to the latent and emit hidden-sized output."""
    torch.manual_seed(0)
    cfg = DeepSeekV3Config()
    from deepseek.attention import MLA
    mla = MLA(cfg).eval()
    x = torch.randn(2, 7, cfg.hidden_size)

    # KV down-projection width = latent + decoupled rope key (shared across heads).
    kv_width = mla.kv_down_proj.out_features
    assert kv_width == cfg.kv_lora_rank + cfg.qk_rope_head_dim, kv_width
    # Q/K score dim is nope+rope; value/output dim is v_head_dim.
    assert mla.qk_head_dim == cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
    assert abs(mla.softmax_scale - cfg.qk_head_dim ** -0.5) < 1e-12

    with torch.no_grad():
        out = mla(x)
    assert out.shape == x.shape, out.shape
    print(f"[mla shapes]     ok  (kv_down={kv_width}=latent{cfg.kv_lora_rank}+rope{cfg.qk_rope_head_dim}, "
          f"qk_head_dim={mla.qk_head_dim}, scale={mla.softmax_scale:.4f})")


def test_moe_routing():
    """Exactly top-k experts chosen, within topk_group groups, weights scaled."""
    torch.manual_seed(0)
    cfg = DeepSeekV3Config()
    gate = MoEGate(cfg)
    # Give the balancing bias non-trivial values to exercise selection-only effect.
    gate.e_score_correction_bias.copy_(torch.randn(cfg.n_routed_experts))
    x = torch.randn(20, cfg.hidden_size)

    idx, w = gate(x)
    assert idx.shape == (20, cfg.num_experts_per_tok)
    # exactly top_k distinct experts per token
    for row in idx:
        assert len(set(row.tolist())) == cfg.num_experts_per_tok
    # group-limit: all chosen experts live in <= topk_group groups
    per_group = cfg.n_routed_experts // cfg.n_group
    for row in idx:
        groups = {(e.item() // per_group) for e in row}
        assert len(groups) <= cfg.topk_group, groups
    # weights normalized then scaled by routed_scaling_factor
    sums = w.sum(-1)
    assert torch.allclose(sums, torch.full_like(sums, cfg.routed_scaling_factor), atol=1e-4), sums[:3]

    # gate weights must come from UNBIASED sigmoid scores, not biased ones
    logits = torch.nn.functional.linear(x.float(), gate.weight.float())
    unbiased = logits.sigmoid()
    pre_norm = unbiased.gather(1, idx)
    expected = pre_norm / pre_norm.sum(-1, keepdim=True) * cfg.routed_scaling_factor
    assert torch.allclose(w, expected.to(w.dtype), atol=1e-4), "gate weights not from unbiased scores"
    print(f"[moe routing]    ok  (top-{cfg.num_experts_per_tok} in <= {cfg.topk_group} group(s), "
          f"sum=routed_scaling={cfg.routed_scaling_factor}, weights from unbiased sigmoid)")


def test_moe_shared_expert_always_on():
    """The shared expert contributes for every token, ungated."""
    torch.manual_seed(0)
    cfg = DeepSeekV3Config()
    from deepseek.moe import DeepSeekMoE
    moe = DeepSeekMoE(cfg).eval()
    x = torch.randn(1, 4, cfg.hidden_size)

    with torch.no_grad():
        full = moe(x)
        # Zero out all routed-expert contributions by forcing gate weights to 0.
        saved = moe.gate.routed_scaling_factor
        moe.gate.routed_scaling_factor = 0.0
        shared_only = moe(x)
        moe.gate.routed_scaling_factor = saved
        shared_ref = moe.shared_experts(x.view(-1, cfg.hidden_size)).view_as(x)

    assert torch.allclose(shared_only, shared_ref, atol=1e-5), "shared expert path mismatch"
    assert not torch.allclose(full, shared_only, atol=1e-4), "routed experts contributed nothing"
    print("[moe shared]     ok  (shared expert always added; routed experts add on top)")


def test_rope_relative_shift():
    """RoPE identity: <rope(q,m), rope(k,n)> depends only on (m-n)."""
    torch.manual_seed(0)
    dim = 16
    rot = RotaryEmbedding(dim, base=10000.0)
    cos, sin = rot(10, torch.device("cpu"), torch.float32)
    q = torch.randn(1, 1, 10, dim)
    k = torch.randn(1, 1, 10, dim)
    qr = apply_rotary(q, cos, sin)
    kr = apply_rotary(k, cos, sin)
    # Compare dot product of (pos 5, pos 2) vs (pos 8, pos 5): same relative offset 3.
    d1 = (qr[0, 0, 5] * kr[0, 0, 2]).sum()
    # Use the SAME underlying vectors shifted by 3 positions to isolate the offset.
    q2 = q.clone(); k2 = k.clone()
    q2[0, 0, 8] = q[0, 0, 5]
    k2[0, 0, 5] = k[0, 0, 2]
    qr2 = apply_rotary(q2, cos, sin)
    kr2 = apply_rotary(k2, cos, sin)
    d2 = (qr2[0, 0, 8] * kr2[0, 0, 5]).sum()
    assert torch.allclose(d1, d2, atol=1e-4), f"rope relative-position broken: {d1} vs {d2}"
    print(f"[rope]           ok  (relative-position identity holds: {d1.item():.4f} == {d2.item():.4f})")


def test_layer_layout():
    """First first_k_dense_replace layers dense, the rest MoE."""
    cfg = DeepSeekV3Config()
    model = DeepSeekV3Model(cfg)
    kinds = [l.is_moe for l in model.layers]
    expected = [i >= cfg.first_k_dense_replace for i in range(cfg.num_hidden_layers)]
    assert kinds == expected, kinds
    print(f"[layer layout]   ok  ({cfg.first_k_dense_replace} dense + "
          f"{cfg.num_hidden_layers - cfg.first_k_dense_replace} MoE)")


if __name__ == "__main__":
    test_causality()
    test_mla_shapes()
    test_moe_routing()
    test_moe_shared_expert_always_on()
    test_rope_relative_shift()
    test_layer_layout()
    print("\nAll architecture invariants hold. ✅")

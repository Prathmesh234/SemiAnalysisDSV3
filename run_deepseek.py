"""Demo: build the scaled DeepSeek-V3 model and run a forward pass.

    uv run run_deepseek.py
"""

import torch

from deepseek import DeepSeekV3Config, DeepSeekV3Model


def main():
    torch.manual_seed(0)

    config = DeepSeekV3Config()
    model = DeepSeekV3Model(config)
    model.eval()

    print("=== Scaled DeepSeek-V3 ===")
    print(f"hidden_size            : {config.hidden_size}")
    print(f"layers                 : {config.num_hidden_layers} "
          f"({config.first_k_dense_replace} dense + "
          f"{config.num_hidden_layers - config.first_k_dense_replace} MoE)")
    print(f"attention heads        : {config.num_attention_heads} "
          f"(MLA: q_lora={config.q_lora_rank}, kv_lora={config.kv_lora_rank})")
    print(f"routed/shared experts  : {config.n_routed_experts}/{config.n_shared_experts} "
          f"(top-{config.num_experts_per_tok})")
    print(f"MTP modules            : {config.num_mtp_modules}")
    print(f"parameters             : {model.num_parameters():,}")

    # Layer types, to show the dense -> MoE transition.
    kinds = ["MoE" if layer.is_moe else "dense" for layer in model.layers]
    print(f"layer FFN types        : {kinds}")

    batch, seq = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (batch, seq))

    with torch.no_grad():
        logits, mtp_logits = model(input_ids, return_mtp=True)

    print("\n=== Forward pass ===")
    print(f"input_ids        : {tuple(input_ids.shape)}")
    print(f"main logits      : {tuple(logits.shape)}  "
          f"(expected {(batch, seq, config.vocab_size)})")
    for k, ml in enumerate(mtp_logits, start=1):
        print(f"MTP[{k}] logits   : {tuple(ml.shape)}  (predicts token +{k})")

    # Sanity checks.
    assert logits.shape == (batch, seq, config.vocab_size)
    assert torch.isfinite(logits).all(), "logits contain NaN/Inf"
    next_token = logits[:, -1].argmax(-1)
    print(f"\nargmax next-token: {next_token.tolist()}")
    print("OK — forward pass produced finite logits.")


if __name__ == "__main__":
    main()

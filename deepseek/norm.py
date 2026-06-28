"""RMSNorm (Root Mean Square Layer Normalization).

DeepSeek-V3 uses RMSNorm everywhere instead of LayerNorm. It normalises by the
root-mean-square of the activations (no mean subtraction, no bias):

    y = x / sqrt(mean(x^2) + eps) * weight
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in fp32 for numerical stability, then cast back.
        dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x.to(dtype))

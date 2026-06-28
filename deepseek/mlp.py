"""SwiGLU feed-forward network.

Used both as the dense FFN in the first layers and as each individual expert in
the MoE layers (just with a different `intermediate_size`). DeepSeek-V3 uses the
gated SiLU (SwiGLU) variant:

    down_proj( silu(gate_proj(x)) * up_proj(x) )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

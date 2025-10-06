# cs336_basics/swiglu.py
from __future__ import annotations
import torch
from torch import nn
from cs336_basics.linear import Linear


class SwiGLU(nn.Module):
    """
    SwiGLU feed-forward:
        a = W1 x
        b = W3 x
        u = SiLU(a) * b            # SiLU(a) = a * sigmoid(a)
        y = W2 u
    Shapes:
        W1: [d_ff, d_model]
        W3: [d_ff, d_model]
        W2: [d_model, d_ff]
    """

    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)  # gate input
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)  # value input
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)  # down projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.w1(x)  # [..., d_ff]
        b = self.w3(x)  # [..., d_ff]
        u = (a * torch.sigmoid(a)) * b  # SwiGLU = SiLU(a) ⊙ b
        return self.w2(u)  # [..., d_model]

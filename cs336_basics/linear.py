# cs336_basics/linear.py
from __future__ import annotations
import torch
from torch import nn
from einops import einsum  # expressive, shape-safe contractions


class Linear(nn.Module):
    """
    A minimal Linear (no bias) implemented with einops.

    Stores weights as W with shape [out_features, in_features] (not transposed).
    Forward computes: y[..., o] = sum_i x[..., i] * W[o, i]
    """

    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.W = nn.Parameter(torch.empty(out_features, in_features, **factory_kwargs))
        nn.init.trunc_normal_(self.W, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Using einops.einsum for clarity over shapes:
        # "... i, o i -> ... o" means: contract the shared 'i' dim, keep batch "...", produce 'o'.
        return einsum(x, self.W, "... i, o i -> ... o")

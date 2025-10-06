from __future__ import annotations
import torch
from torch import nn


def rmsnorm_fn(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Functional RMSNorm (scale-only) that preserves dtype.

    y = x * weight / sqrt(mean(x^2, dim=-1) + eps)
    """
    in_dtype = x.dtype
    x32 = x.to(torch.float32)
    rms = torch.sqrt(x32.pow(2).mean(dim=-1, keepdim=True) + float(eps))
    y32 = (x32 / rms) * weight.to(torch.float32)
    return y32.to(in_dtype)


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Norm (no bias).

    y = x * weight / sqrt(mean(x^2, dim=-1) + eps)
    The normalization is over the last dim (d_model).
    """

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = float(eps)
        # Learnable scale, one per hidden feature
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rmsnorm_fn(x, self.weight, self.eps)

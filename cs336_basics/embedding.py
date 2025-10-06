from __future__ import annotations
import torch
from torch import nn


class Embedding(nn.Module):
    """
    Minimal embedding layer.

    Stores an embedding matrix `weight` of shape [num_embeddings, embedding_dim].
    Forward returns `weight[token_ids]`, so the final dimension is embedding_dim.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, **factory_kwargs))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: Long tensor of arbitrary leading shape (...), values in [0, num_embeddings)
        # returns: (..., embedding_dim)
        return self.weight[token_ids]

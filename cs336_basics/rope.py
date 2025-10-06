from __future__ import annotations
import torch
from torch import nn


class RotaryPositionalEmbedding(nn.Module):
    """
    RoPE for queries/keys.
    x has shape (..., seq_len, d_k). We rotate last-dim in pairs (even, odd).

    Frequencies:
        freq[i] = theta^(-2i/d_k),  i = 0..(d_k/2-1)
    Angles for position p:  p * freq
    """

    # Explicitly annotate buffers for static type checkers
    cos_cached: torch.Tensor
    sin_cached: torch.Tensor

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError(f"RoPE requires even d_k, got {d_k}")
        self.d_k = d_k
        self.max_seq_len = int(max_seq_len)

        # Precompute cos/sin tables of shape [max_seq_len, d_k//2]
        i = torch.arange(0, d_k // 2, device=device, dtype=torch.float32)
        inv_freq = theta ** (-2.0 * i / d_k)  # [d_k//2]
        # positions 0..max_seq_len-1
        pos = torch.arange(self.max_seq_len, device=device, dtype=torch.float32)  # [S]
        angles = torch.einsum("s,f->sf", pos, inv_freq)  # [S, d_k//2]

        self.register_buffer("cos_cached", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cached", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        x: (..., seq_len, d_k)
        token_positions: (..., seq_len)  (same leading dims as x except last)
        returns: (..., seq_len, d_k)
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # Get cos/sin for the given positions: (..., seq_len, d_k//2)
        cos = self.cos_cached[token_positions]  # gather with broadcasting of leading dims
        sin = self.sin_cached[token_positions]

        # Split last dim into pairs (even, odd)
        x_even = x[..., 0::2]  # (..., seq_len, d_k//2)
        x_odd = x[..., 1::2]

        # If x has extra leading dims (e.g., heads), insert the exact number of singleton
        # dims just before the sequence dimension so shapes align as
        # (..., [extra dims], seq_len, d_k//2). This works for token_positions of shape
        # [T] or [B, T] and x shapes like [B, T, D] or [B, H, T, D].
        extra_leading = x.ndim - (token_positions.ndim + 1)
        if extra_leading > 0:
            insert_dim = token_positions.ndim - 1  # position of seq_len in token_positions
            for _ in range(extra_leading):
                cos = cos.unsqueeze(insert_dim)
                sin = sin.unsqueeze(insert_dim)

        # Rotate: [x_even, x_odd] -> [x_even*cos - x_odd*sin, x_odd*cos + x_even*sin]
        out_even = x_even * cos - x_odd * sin
        out_odd = x_odd * cos + x_even * sin

        # Interleave back to (..., seq_len, d_k)
        out = torch.empty_like(x)
        out[..., 0::2] = out_even
        out[..., 1::2] = out_odd
        return out.to(in_dtype)

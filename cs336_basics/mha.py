from __future__ import annotations
import math
import torch


def causal_mha_self_attn(
    x: torch.Tensor,  # [..., T, d_model]
    Wq: torch.Tensor,  # [d_model, d_model]  (out, in)
    Wk: torch.Tensor,  # [d_model, d_model]
    Wv: torch.Tensor,  # [d_model, d_model]
    Wo: torch.Tensor,  # [d_model, d_model]
    num_heads: int,
) -> torch.Tensor:
    *lead, T, d_model = x.shape
    H = num_heads
    assert d_model % H == 0, "d_model must be divisible by num_heads"
    d_k = d_model // H

    # upcast for numerical stability
    x32 = x.to(torch.float32)
    Wq32, Wk32, Wv32, Wo32 = (
        Wq.to(torch.float32),
        Wk.to(torch.float32),
        Wv.to(torch.float32),
        Wo.to(torch.float32),
    )

    # Project to model width with W[out,in] -> x @ W.T  ==> "... t m, o m -> ... t o"
    Qm = torch.einsum("... t m, o m -> ... t o", x32, Wq32)  # [..., T, d_model]
    Km = torch.einsum("... t m, o m -> ... t o", x32, Wk32)
    Vm = torch.einsum("... t m, o m -> ... t o", x32, Wv32)

    # split into heads: [..., H, T, d_k]
    Q = Qm.reshape(*lead, T, H, d_k).transpose(-3, -2)
    K = Km.reshape(*lead, T, H, d_k).transpose(-3, -2)
    V = Vm.reshape(*lead, T, H, d_k).transpose(-3, -2)

    # scaled dot-product attention with causal mask
    scores = torch.einsum("... h q k, ... h t k -> ... h q t", Q, K) / math.sqrt(d_k)
    causal = torch.ones(T, T, dtype=torch.bool, device=scores.device).tril()
    scores = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)

    scores = scores - torch.amax(scores, dim=-1, keepdim=True)
    attn = torch.exp(scores) * causal
    attn = attn / torch.sum(attn, dim=-1, keepdim=True)

    # context per head -> concat -> output proj
    ctx = torch.einsum("... h q t, ... h t v -> ... h q v", attn, V)  # [..., H, T, d_k]
    ctx = ctx.transpose(-3, -2).reshape(*lead, T, d_model)  # [..., T, d_model]

    # output projection: W_O[ out,in ] -> ctx @ W_O.T
    y = torch.einsum("... t m, o m -> ... t o", ctx, Wo32)  # [..., T, d_model]
    return y.to(x.dtype)

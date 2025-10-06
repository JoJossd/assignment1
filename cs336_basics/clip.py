from collections.abc import Iterable
import torch


def run_gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
) -> None:
    """
    Clip the global L2 norm of all parameter gradients to <= max_l2_norm.
    Modifies .grad in-place. Uses eps=1e-6 for numerical stability.
    """
    eps = 1e-6
    params = [p for p in parameters if p is not None and p.grad is not None]
    if not params:
        return
    if max_l2_norm <= 0:
        return  # nothing sensible to do

    # Compute global L2 norm across all grads, in fp32 for stability
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return
    total_sq = torch.zeros((), dtype=torch.float32, device=grads[0].device)
    for g in grads:
        total_sq += g.detach().to(torch.float32).pow(2).sum()
    total_norm = total_sq.sqrt()

    if total_norm <= max_l2_norm:
        return  # already within the budget

    scale = max_l2_norm / (total_norm + eps)
    for p in params:
        if p.grad is not None:
            p.grad.mul_(scale)

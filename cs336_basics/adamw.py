import math
import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    """
    Minimal AdamW (decoupled weight decay) following the assignment’s algorithm.

    Args:
        params: iterable of parameters to optimize.
        lr (float): learning rate α.
        betas (Tuple[float, float]): (β1, β2).
        eps (float): ε for numerical stability.
        weight_decay (float): λ (decoupled).
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        if lr <= 0.0:
            raise ValueError("lr must be positive")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError("betas must be in [0,1)")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        defaults = dict(lr=float(lr), betas=tuple(betas), eps=float(eps), weight_decay=float(weight_decay))
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                # state init
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                state["step"] += 1
                t = state["step"]

                # m ← β1 m + (1-β1) g
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                # v ← β2 v + (1-β2) g^2
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # α_t = α * sqrt(1-β2^t) / (1-β1^t)
                bias_correction1 = 1.0 - beta1**t
                bias_correction2 = 1.0 - beta2**t
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                # θ ← θ - α_t * m / (sqrt(v)+ε)
                # Perform parameter updates under no_grad to avoid in-place ops on leaf tensors
                with torch.no_grad():
                    denom = exp_avg_sq.sqrt().add_(eps)
                    p.addcdiv_(exp_avg, denom, value=-step_size)

                    # Decoupled weight decay: θ ← θ - α λ θ
                    if wd != 0.0:
                        p.add_(p, alpha=-lr * wd)

        return loss

import math


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    Cosine LR with linear warmup.

    Warm-up (t < T_w):      α_t = (t / T_w) * α_max
    Cosine (T_w ≤ t ≤ T_c): α_t = α_min + 0.5*(1 + cos( (t-T_w)/(T_c-T_w) * π ))*(α_max-α_min)
    Post  (t > T_c):        α_t = α_min
    """
    t = it
    a_max = float(max_learning_rate)
    a_min = float(min_learning_rate)
    Tw = int(warmup_iters)
    Tc = int(cosine_cycle_iters)

    # Warm-up (guard Tw==0 just in case)
    if t < Tw:
        return a_max * (t / max(1, Tw))

    # Cosine phase (guard Tc==Tw to avoid div-by-zero in degenerate case)
    if t <= Tc:
        if Tc == Tw:
            return a_min
        frac = (t - Tw) / (Tc - Tw)
        return a_min + 0.5 * (1.0 + math.cos(math.pi * frac)) * (a_max - a_min)

    # After cosine
    return a_min

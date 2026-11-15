"""GRPO advantages with CLM's dual channel (docs/data_and_training.md §3.3–§3.4).

    A[i, t] = (r_i − mean_g r) · loss_mask[t]  +  w_eff · a_eff_i · ctrl_mask[t]
    a_eff_i = clip((C̄_succ − C_i) / C̄_succ, −1, 1)  for successful i in groups with ≥ 2 successes, else 0
"""

from __future__ import annotations


def group_advantages(rewards: list[float], costs: list[float], success_at: float = 1.0) -> tuple[list[float], list[float]]:
    g = len(rewards)
    mean_r = sum(rewards) / g
    a_task = [r - mean_r for r in rewards]
    succ = [i for i, r in enumerate(rewards) if r >= success_at]
    a_eff = [0.0] * g
    if len(succ) >= 2:
        c_bar = sum(costs[i] for i in succ) / len(succ)
        if c_bar > 0:
            for i in succ:
                a_eff[i] = max(-1.0, min(1.0, (c_bar - costs[i]) / c_bar))
    return a_task, a_eff


def token_advantages(rec: dict, a_task: float, a_eff: float, w_eff: float) -> list[float]:
    return [a_task * lm + w_eff * a_eff * cm for lm, cm in zip(rec["loss_mask"], rec["ctrl_mask"])]


def informative(rewards: list[float]) -> bool:
    """Dynamic sampling: a group whose rewards are all equal gives no task signal."""
    return max(rewards) - min(rewards) > 1e-9

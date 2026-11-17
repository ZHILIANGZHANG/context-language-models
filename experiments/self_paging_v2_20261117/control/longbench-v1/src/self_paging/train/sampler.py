"""Prompt sampling for GRPO: pool weights, on-line adaptation to within-group variance, and a
budget curriculum (docs/rl_design.md §3).

* **Buckets** are (budget ratio, decision count, delayed decisions capped at 3). Rows without the
  ``n_delayed`` field (pools written before ``data/rl_strata.py``) are annotated once at start-up.
* **Signal** of a finished group: the standard deviation of task + hygiene scores, relative to
  ``std_ref`` and capped at 1. A group with equal scores that still carries signal (counterfactual
  credit, or an efficiency difference between clean successes) counts ``weak``. Pass rate is not used:
  partial scores (0.25, 0.5, 0.75) carry signal, and all-correct groups can carry an efficiency signal.
* **Weight** of a row: its pool ``weight`` × the EMA of its bucket's signal (unseen buckets start at 1).
  A fraction ``explore`` of draws ignores the adaptation and uses the pool weights only, so a bucket that
  went quiet keeps at least that share of attention and can come back.
* **Curriculum**: ``ratio_start`` maps a budget ratio to the first step at which it may be drawn, e.g.
  ``{0.25: 30}`` trains on the looser budgets first.

Each rank keeps its own statistics (ranks see different groups; nothing is all-reduced).
"""

from __future__ import annotations

import math
import random
from collections import Counter


def bucket_of(row: dict) -> tuple:
    return (float(row["ratio"]), int(row.get("n_decisions") or row["t1_config"]["n_decisions"]),
            min(int(row.get("n_delayed", 0)), 3))


def group_signal(group: list[dict], std_ref: float = 0.25, weak: float = 0.25,
                 has_other_signal: bool = False) -> float:
    scores = [r["reward"] + r.get("hygiene", 0.0) for r in group]
    mean = sum(scores) / len(scores)
    std = math.sqrt(sum((s - mean) ** 2 for s in scores) / len(scores))
    sig = min(1.0, std / std_ref) if std_ref > 0 else float(std > 0)
    return max(sig, weak) if (sig == 0.0 and has_other_signal) else sig


class PromptSampler:
    def __init__(self, pool: list[dict], adaptive: bool = True, explore: float = 0.1, ema: float = 0.3,
                 std_ref: float = 0.25, weak: float = 0.25, ratio_start: dict | None = None):
        if not pool:
            raise ValueError("empty prompt pool")
        if any("n_delayed" not in r for r in pool):
            from ..data.rl_strata import annotate

            cache: dict = {}
            pool = [r if "n_delayed" in r else annotate(r, cache=cache) for r in pool]
        self.pool = pool
        self.adaptive, self.explore, self.ema, self.std_ref, self.weak = adaptive, explore, ema, std_ref, weak
        self.ratio_start = {float(k): int(v) for k, v in (ratio_start or {}).items()}
        self.buckets = [bucket_of(r) for r in pool]
        self.base = [float(r.get("weight", 1.0)) for r in pool]
        self.signal: dict[tuple, float] = {}
        self.seen: Counter = Counter()
        self.drawn: Counter = Counter()

    def allowed(self, step: int) -> list[int]:
        idx = [i for i, r in enumerate(self.pool) if self.ratio_start.get(float(r["ratio"]), 0) <= step]
        if not idx:
            raise ValueError(f"curriculum leaves no prompts at step {step}")
        return idx

    def multiplier(self, b: tuple) -> float:
        return self.signal.get(b, 1.0) if self.adaptive else 1.0

    def sample(self, rng: random.Random, step: int = 0) -> dict:
        idx = self.allowed(step)
        w = [self.base[i] for i in idx]
        if self.adaptive and rng.random() >= self.explore:
            wa = [x * self.multiplier(self.buckets[i]) for x, i in zip(w, idx)]
            if sum(wa) > 0:
                w = wa
        i = rng.choices(idx, weights=w, k=1)[0]
        self.drawn[self.buckets[i]] += 1
        return self.pool[i]

    def update(self, row: dict, group: list[dict], has_other_signal: bool = False) -> float:
        b = bucket_of(row)
        sig = group_signal(group, self.std_ref, self.weak, has_other_signal)
        old = self.signal.get(b)
        self.signal[b] = sig if old is None else (1 - self.ema) * old + self.ema * sig
        self.seen[b] += 1
        return sig

    def summary(self, step: int = 0) -> dict:
        idx = self.allowed(step)
        mass = {}
        for i in idx:
            mass[self.buckets[i]] = mass.get(self.buckets[i], 0.0) + self.base[i]
        total = sum(mass.values())
        adapt = sum(m * self.multiplier(b) for b, m in mass.items())
        # effective share of draws per bucket kind under the current mixture
        p = {b: (self.explore if self.adaptive else 1.0) * m / total
             + ((1 - self.explore) * m * self.multiplier(b) / adapt if self.adaptive and adapt > 0 else 0.0)
             for b, m in mass.items()}
        quiet = [b for b in self.signal if self.signal[b] < 0.1]
        return {"sampler_buckets_seen": len(self.signal), "sampler_buckets_quiet": len(quiet),
                "sampler_quiet_mass": round(sum(p.get(b, 0.0) for b in quiet), 4),
                "sampler_p_delayed": round(sum(v for b, v in p.items() if b[2] > 0), 4),
                "sampler_ratios": sorted({b[0] for b in mass})}

"""Illustrative (not measured) cost model of context-management strategies on a deep-research-like workload.

Usage: OBS=4000 T=80 python clm_cost_model.py   (observation tokens per step, number of task steps)
Strategies: full = no management with an unlimited window; summary = harness summary at the budget;
clm_batch = one large model-directed compaction at 75% of the budget; clm_eager = each observation
shrunk to a note right after it is used, plus a rare fold of old turns.

Each call sends a prompt (list of segments) and generates `gen` tokens. Costs per call:
  FLOPs (CLM's own metric, kv_cache_flops.py): 2*N_body*(new_prefill + gen)
        + 4*layers*width*((L^2 - matched^2)/2 + gen*L)        [prefix caching at segment granularity]
  API with prompt caching (Anthropic-style): matched*read + new*write + gen*out
  CLM's local USD estimate (summarize_compute_cost): new*in + gen*out      [cached prefix = free]
  No caching: L*in + gen*out
Segments are (id, tokens); the cached prefix is the longest run of identical leading segments
shared with the previous call (vLLM/SGLang reuse it across calls)."""
import itertools

N_BODY, LAYERS, WIDTH = 24.3532e9, 16, 6144          # Qwen3.6-27B, as in kv_cache_flops.py
IN, WRITE, READ, OUT = 3.0e-6, 3.75e-6, 0.30e-6, 15.0e-6  # $/token: input, cache write, cache read, output

import os
P, GEN, NOTE, EDIT_GEN, SUMMARY = 2000, 300, 150, 200, 1500
OBS = int(os.environ.get("OBS", 4000))
T, BUDGET = int(os.environ.get("T", 80)), 28672
_ids = itertools.count()


def seg(n):
    return (next(_ids), n)


class Meter:
    def __init__(self):
        self.prev, self.flops, self.api, self.local, self.naive, self.calls, self.Ls = [], 0.0, 0.0, 0.0, 0.0, 0, []

    def call(self, prompt, gen):
        L = sum(n for _, n in prompt)
        matched = 0
        for a, b in zip(prompt, self.prev):
            if a != b:
                break
            matched += a[1]
        new = L - matched
        self.flops += 2 * N_BODY * (new + gen) + 4 * LAYERS * WIDTH * ((L * L - matched * matched) / 2 + gen * L)
        self.api += matched * READ + new * WRITE + gen * OUT
        self.local += new * IN + gen * OUT
        self.naive += L * IN + gen * OUT
        self.prev, self.calls = list(prompt), self.calls + 1
        self.Ls.append(L)


def collapse(ctx, keep, gen_summary):
    """Fold every turn before the last `keep` segments into one summary segment."""
    return [ctx[0], seg(gen_summary)] + ctx[-keep:]


def run(strategy):
    m, ctx = Meter(), [seg(P)]
    for t in range(T):
        L = sum(n for _, n in ctx)
        if strategy == "summary" and L + GEN + OBS > BUDGET:
            m.call(ctx, SUMMARY)                       # harness asks for a summary of everything
            ctx = [ctx[0], seg(SUMMARY)]
        if strategy == "clm_batch" and L + GEN + OBS > 0.75 * BUDGET:
            # one editing turn, chosen by the model: old observations -> notes, and if that is
            # not enough, fold everything but the last step into a summary
            body = [s if s[1] != OBS else seg(NOTE) for s in ctx[1:-2]]
            ctx = [ctx[0]] + body + ctx[-2:]
            gen = EDIT_GEN
            if sum(n for _, n in ctx) + GEN + OBS > 0.5 * BUDGET:
                ctx, gen = collapse(ctx, 2, SUMMARY), EDIT_GEN + SUMMARY
            m.call(m.prev or ctx, gen)                 # the editing call sees the pre-edit prompt
        if strategy == "clm_eager" and L + GEN + OBS > 0.75 * BUDGET:
            ctx = collapse(ctx, 2, SUMMARY)            # rare fold of old turns
            m.call(m.prev or ctx, EDIT_GEN + SUMMARY)
        m.call(ctx, GEN)                               # the task step
        ctx = ctx + [seg(GEN), seg(OBS)]
        if strategy == "clm_eager" and t > 0:
            # in the same command as the next action: shrink the previous observation to a note
            ctx[len(ctx) - 3] = seg(NOTE)
    return m


rows = []
for s in ("full", "summary", "clm_batch", "clm_eager"):
    m = run(s)
    rows.append((s, m))
base = dict(rows)["summary"]
print(f"{'strategy':10s} {'calls':>5s} {'meanL':>7s} {'maxL':>7s} {'PFLOPs':>8s} {'API$':>7s} {'CLMlocal$':>9s} {'naive$':>7s}   ratios vs summary (FLOPs / API / local / naive)")
for s, m in rows:
    print(f"{s:10s} {m.calls:5d} {sum(m.Ls)/len(m.Ls):7.0f} {max(m.Ls):7d} {m.flops/1e15:8.2f} {m.api:7.3f} {m.local:9.3f} {m.naive:7.3f}   "
          f"{m.flops/base.flops:.2f} / {m.api/base.api:.2f} / {m.local/base.local:.2f} / {m.naive/base.naive:.2f}")

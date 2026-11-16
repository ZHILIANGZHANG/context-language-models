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


# -- reward v2 (docs/rl_design.md §4) ------------------------------------------------------------
# per-event penalty and cap for each hygiene counter; the total is capped at HYGIENE_CAP
HYGIENE = {"forced_evictions": (0.02, 0.10), "truncated_pages": (0.03, 0.10),
           "invalid": (0.05, 0.10), "kv_errors": (0.02, 0.05)}
HYGIENE_CAP = 0.25


def hygiene_penalty(meta: dict) -> float:
    """Small, capped penalty for the mechanisms through which information gets lost
    (system evictions, truncated pages) and for protocol errors. Always ≤ 0."""
    total = sum(min(cap, per * meta.get(k, 0)) for k, (per, cap) in HYGIENE.items())
    return -min(HYGIENE_CAP, total)


def counterfactual_credit(rec: dict, env, scale: float = 0.5) -> tuple[list[float], dict]:
    """Token-level credit for the KV calls that made a T1 decision fail or succeed.

    For every decision, V = resident environment text in the turn that called ``store``:
      * wrong and rule_read(V) != gold: every environment page p with rule_read(V ∪ {p}) == gold
        (or, if no single page suffices, every missing page of the minimal sufficient set) whose
        last removal was the model's own ``kv_out`` → −c on that kv_out turn's control tokens;
      * right: every page q brought back by the model's ``kv_in`` with rule_read(V − {q}) != gold
        → +c on that kv_in turn's control tokens;
      * wrong with rule_read(V) == gold is a reasoning error: no KV credit.
    c = scale / number of decisions. Ground truth is used only to compute the reward, never shown
    to the model. Model notes are not parsed, which is why blame is applied only to wrong answers.
    """
    from ..envs.t1_warehouse import rule_read

    T = len(rec["tokens"])
    credit = [0.0] * T
    stats = {"blamed_kv_out": 0, "credited_kv_in": 0, "reasoning_errors": 0}
    ep = env.ep
    if not ep.decisions:
        return credit, stats
    c = scale / len(ep.decisions)
    snaps = [set(s) for s in rec["snapshots"]]
    epid = {int(k): v for k, v in rec["env_page"].items()}
    kv_log = rec["meta"].get("kv_log", [])
    ctrl_by_turn: dict[int, list[int]] = {}
    for i, (p, m) in enumerate(zip(rec["page_of"], rec["ctrl_mask"])):
        if m:
            ctrl_by_turn.setdefault(p, []).append(i)

    def text(es):
        return [ep.pages[e] for e in sorted(es)]

    def add(turn_pid, value):
        for i in ctrl_by_turn.get(turn_pid, []):
            credit[i] += value

    decisions = {d.page: d for d in ep.decisions}
    for call in rec["meta"].get("env_calls", []):
        if call["name"] != "store" or call.get("decision_page") not in decisions:
            continue
        d = decisions[call["decision_page"]]
        v = call["version"]
        resident = {e for e, pid in epid.items() if e <= d.page and pid in snaps[v]}
        visible_ok = bool(resident) and rule_read(text(resident)) == d.answer
        right = env.answers.get(d.page) == d.answer
        before = [k for k in kv_log if k["turn"] is not None and k["turn"] < call["turn"] and k["ok"]]
        if not right:
            if visible_ok:
                stats["reasoning_errors"] += 1
                continue
            absent = [e for e in epid if e <= d.page and e not in resident]
            fixers = [e for e in absent if rule_read(text(resident | {e})) == d.answer]
            if not fixers:
                fixers = [e for e in ep.need[d.page] if e in absent]
            for e in fixers:
                outs = [k for k in before if epid[e] in k["pages"] and k["op"] in ("kv_out", "forced_out", "kv_drop")]
                if outs and outs[-1]["op"] != "forced_out":  # system evictions are hygiene, not credit
                    add(outs[-1]["turn_pid"], -c)
                    stats["blamed_kv_out"] += 1
        else:
            for k in before:
                if k["op"] != "kv_in":
                    continue
                for pid in k["pages"]:
                    e = next((x for x, y in epid.items() if y == pid), None)
                    if e is not None and e in resident and rule_read(text(resident - {e})) != d.answer:
                        add(k["turn_pid"], c)
                        stats["credited_kv_in"] += 1
    return credit, stats


def advantages_v2(group: list[dict], w_eff: float, w_credit: float, success_at: float = 1.0) -> list[list[float]]:
    """Reward v2 advantages for one GRPO group. Each record needs ``reward`` (task score),
    ``hygiene`` (≤ 0), ``cost`` and ``credit`` (per-token list, may be all zeros).

      R_i       = reward_i + hygiene_i
      A_task_i  = R_i − mean_g R                       (no std normalisation)
      a_eff_i   = efficiency among *clean* successes (reward ≥ success_at and hygiene == 0)
      A[i, t]   = A_task_i·loss[t] + w_eff·a_eff_i·ctrl[t] + w_credit·credit_i[t]
    """
    R = [r["reward"] + r.get("hygiene", 0.0) for r in group]
    mean = sum(R) / len(R)
    clean = [i for i, r in enumerate(group) if r["reward"] >= success_at and r.get("hygiene", 0.0) == 0.0]
    a_eff = [0.0] * len(group)
    if len(clean) >= 2:
        c_bar = sum(group[i]["cost"] for i in clean) / len(clean)
        if c_bar > 0:
            for i in clean:
                a_eff[i] = max(-1.0, min(1.0, (c_bar - group[i]["cost"]) / c_bar))
    out = []
    for i, r in enumerate(group):
        cr = r.get("credit") or [0.0] * len(r["loss_mask"])
        out.append([(R[i] - mean) * lm + w_eff * a_eff[i] * cm + w_credit * cv * lm
                    for lm, cm, cv in zip(r["loss_mask"], r["ctrl_mask"], cr)])
    return out


def has_signal(advs: list[list[float]]) -> bool:
    """Keep a group if any token has a non-zero advantage (task, efficiency or credit), not only
    when task rewards differ: an all-success group can still carry an efficiency signal."""
    return any(abs(a) > 1e-12 for adv in advs for a in adv)

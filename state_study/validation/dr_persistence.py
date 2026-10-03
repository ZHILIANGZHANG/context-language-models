"""Secondary analysis of delayed-relevance traces (read-only use of their package and data).

For explicit-state arms, replay the environment with the executed actions and the model's own
state patches; for each shelf whose believed content departs from reality, record how long the
discrepancy lasts, which way it points, and what ends it. For every arm, measure how errors cluster
in time from the per-step correctness flags.
"""
import glob, json, re, sys, statistics as st
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True  # keep the submodule free of __pycache__
DR = Path(__file__).resolve().parents[2] / "third_party" / "references" / "delayed-relevance" / "src"
sys.path.insert(0, str(DR))
from dr.envs.warehouse import Warehouse          # noqa: E402
from dr.runner import NO_OP                      # noqa: E402
from dr.runtimes.skillstate import SkillStateRuntime, _merge_into  # noqa: E402
from dr.types import Action                      # noqa: E402

TR = sys.argv[1]  # directory that contains the extracted results/ (see README)

def belief(state):
    out = {}
    for k, v in (state.get("shelf_contents") or {}).items():
        try:
            i = int(k)
        except (TypeError, ValueError):
            continue
        if isinstance(v, dict):
            out[i] = (str(v.get("sku")),)
        elif isinstance(v, str) and v:
            out[i] = (v,)
        else:
            out[i] = None
    return out

def reality(env):
    return {i: ((c[0],) if c else None) for i, c in env.shelves.items()}

def kind(b, r):
    if b is None and r is None:
        return None
    if b is None:
        return "phantom_vacancy"      # believes empty, actually occupied  -> invites a Store (loud: rejected)
    if r is None:
        return "phantom_occupancy"    # believes occupied, actually empty  -> suppresses a Store (silent)
    return None if b[0] == r[0] else "wrong_sku"

def load(f):
    rows = [json.loads(l) for l in open(f) if l.strip()]
    head = next((r["condiciones"] for r in rows if r.get("kind") == "run_header"), None)
    return head, [r for r in rows if r.get("kind") != "run_header"]

def state_episodes(f):
    head, steps = load(f)
    if head is None or head.get("runtime") != "skillstate":
        return None
    env = Warehouse(horizon=len(steps), seed=head["seed"], apendice_b=head.get("apendice_b", False),
                    sin_telemetria=head.get("sin_telemetria", False), ruido=head.get("ruido", 0))
    env.reset()
    fields = set(env.schema_fields())
    state, open_d, done = {}, {}, []
    for p in steps:
        obs = env.observe()
        b, r = belief(state), reality(env)
        # close discrepancies that were fixed by the previous patch
        for s in list(open_d):
            if kind(b.get(s), r[s]) is None:
                d = open_d.pop(s); d["end"] = p["step"]; d["dur"] = d["end"] - d["start"]; done.append(d)
        # failures during open discrepancies, attributed when the executed action is the believed-optimal one
        if p.get("actionable") and not p.get("correct") and open_d:
            for d in open_d.values():
                d["fails"] += 1
        for resp in p["raw"]["respuestas"]:
            parsed = SkillStateRuntime._parse(resp)
            if parsed is not None and all(k in fields for k in parsed[0]):
                _merge_into(state, parsed[0], deep=True)
                break
        act = Action.parse(p["ejecutado"]) if p.get("ejecutado") else None
        env.apply(act if act is not None else NO_OP)
        b2, r2 = belief(state), reality(env)
        nxt = p["step"] + 1
        for s in r2:
            k = kind(b2.get(s), r2[s])
            if k and s not in open_d:
                open_d[s] = {"file": Path(f).name, "shelf": s, "kind": k, "start": nxt, "fails": 0,
                             "origin_action_ok": bool(p.get("correct"))}
        # how does the NEXT observation look for open discrepancies (rejection / mention)?
        if env.step_index < len(steps):
            text = env.observe().text
            for s, d in open_d.items():
                if "ACTION REJECTED" in text and re.search(rf"shelf {s}\b", text):
                    d["rejected_feedback"] = True
    for d in open_d.values():
        d["end"] = None; d["dur"] = len(steps) - d["start"]; d["censored"] = True; done.append(d)
    return done

def runs(flags):
    out, cur = [], 0
    for ok in flags:
        if not ok:
            cur += 1
        elif cur:
            out.append(cur); cur = 0
    if cur:
        out.append(cur)
    return out

def error_dynamics(pattern):
    ee = eo = oe = oo = 0
    all_runs, eps, err_eps, n = [], 0, 0, 0
    per_ep_err = []
    for f in sorted(glob.glob(pattern)):
        _, steps = load(f)
        flags = [bool(p.get("correct")) for p in steps if p.get("actionable")]
        if not flags:
            continue
        eps += 1; n += len(flags)
        e = flags.count(False); per_ep_err.append(e); err_eps += e > 0
        for a, b in zip(flags, flags[1:]):
            if not a and not b: ee += 1
            elif not a and b: eo += 1
            elif a and not b: oe += 1
            else: oo += 1
        all_runs += runs(flags)
    if not eps:
        return None
    pe = sum(per_ep_err) / n
    p_e_e = ee / (ee + eo) if ee + eo else float("nan")
    p_e_o = oe / (oe + oo) if oe + oo else float("nan")
    return dict(episodes=eps, steps=n, err_rate=round(pe, 4), p_err_after_err=round(p_e_e, 3),
                p_err_after_ok=round(p_e_o, 4), lift=round(p_e_e / p_e_o, 1) if p_e_o else None,
                mean_run=round(st.mean(all_runs), 2) if all_runs else 0, max_run=max(all_runs or [0]),
                eps_with_err=err_eps, errs_per_ep=sorted(per_ep_err, reverse=True)[:8])

if __name__ == "__main__":
    print("== error dynamics from per-step correctness (actionable steps) ==")
    for T in (50, 200):
        for model in ("claude-haiku-4-5", "gemini-3-flash-preview"):
            for rt in ("react", "memory", "stateful", "skillstate"):
                r = error_dynamics(f"{TR}/results/adj_T{T}_{model}_{rt}_s*.jsonl")
                if r:
                    print(f"T={T:3d} {model:22s} {rt:10s} {r}")
    print("\n== L2 retroactive-correction probe ==")
    for model in ("claude-haiku-4-5", "claude-sonnet-5", "gemini-3-flash-preview"):
        for rt in ("react", "skillstate"):
            r = error_dynamics(f"{TR}/results/l2v2_{model}_{rt}_*.jsonl")
            if r:
                print(f"{model:22s} {rt:10s} {r}")

def summarize_discrepancies(pattern, label):
    eps = []
    for f in sorted(glob.glob(pattern)):
        r = state_episodes(f)
        if r is not None:
            eps += r
    by = defaultdict(list)
    for d in eps:
        by[d["kind"]].append(d)
    print(f"\n-- {label}: {len(eps)} belief-reality discrepancies")
    for k, ds in sorted(by.items()):
        durs = [d["dur"] for d in ds]
        cens = sum(1 for d in ds if d.get("censored"))
        fails = sum(d["fails"] for d in ds)
        rej = sum(1 for d in ds if d.get("rejected_feedback"))
        print(f"   {k:18s} n={len(ds):4d}  median_dur={st.median(durs):6.1f}  mean_dur={st.mean(durs):6.1f}  "
              f"max={max(durs):4d}  still_open_at_end={cens:3d}  failures_during={fails:4d}  got_rejection_feedback={rej}")
    return eps

if __name__ == "__main__":
    print("\n== belief-reality discrepancies in explicit-state (SKILL.state) runs ==")
    summarize_discrepancies(f"{TR}/results/adj_T200_claude-haiku-4-5_skillstate_s*.jsonl", "Haiku T=200")
    summarize_discrepancies(f"{TR}/results/adj_T50_claude-haiku-4-5_skillstate_s*.jsonl", "Haiku T=50")
    summarize_discrepancies(f"{TR}/results/l2v2_claude-sonnet-5_skillstate_*.jsonl", "Sonnet L2")
    summarize_discrepancies(f"{TR}/results/l2v2_claude-haiku-4-5_skillstate_*.jsonl", "Haiku L2")

"""L1 replay: real Haiku ReAct prefixes from the delayed-relevance L1 traces (quarantine notice
40 steps before the first decision it affects).

Single call at the decision step:
  BASE   as recorded
  CHECK  one generic pre-action sentence in the spec (names no event type and no shelf)
  JIT    the env's distilled STANDING NOTICE line attached to the decision observation (oracle)
Two calls (W answers the notice, D answers the decision):
  PIN    the spec offers agent-written 'PIN:' lines, re-shown above every later observation
  FACT   the notice states an occupancy fact (shelf occupied by a maintenance rig) instead of a rule
Recorded replies between notice and decision are kept as they are (they never mention the notice).

    python dr_l1.py build   --traces T --out OUT      # BASE/CHECK/JIT -> OUT/A, OUT/blind.json
    python dr_l1.py build-w --traces T --out OUT      # PIN/FACT step W -> OUT/W
    python dr_l1.py build-d --traces T --out OUT      # after OUT/outW -> OUT/D, OUT/blind_D.json
    python dr_l1.py score   --out OUT
T is the results/ directory extracted from the delayed-relevance traces (see README)."""
import argparse, collections, hashlib, json, random, re
from pathlib import Path

from dr_common import history_lines, import_dr, load_steps, react_prompt

SALT_1, SALT_2 = "l1fix-5e07", "l1pin-c3a1"
CHECK = ("\nBefore committing an action, look back through the whole history for earlier events that "
         "change which action is correct now, including events that needed no action when they arrived.\n")
PIN = ("\nPINNED NOTES\nIf an event establishes a rule or restriction that could affect later actions, add one "
       "line to your reply that starts with 'PIN:' and states it. Every PIN line you write is shown to you "
       "again with each later event, under the heading PINNED NOTES.\n")
SEEDS, REPS = (0, 3, 6), range(4)


def trace(t, seed, rep):
    return load_steps(t, f"l1v3_T50_claude-haiku-4-5_mt8192_estricto_v3_react_k40_s{seed}_r{rep}.jsonl")


def spec(W, seed, extra=""):
    return W(horizon=50, seed=seed, latent_k=40, latent_estricto=True).spec() + extra


def positions(st):
    n = next(i for i, r in enumerate(st) if "facility_notice" in r["observation"])
    d = next(i for i, r in enumerate(st) if r.get("es_el_paso"))
    return n, d, int(re.search(r"\| shelf=(\d+) ", st[n]["observation"]).group(1))


def fact(text):
    old = "| status=quarantined | reason=scheduled_maintenance | effective=immediately | do_not_store=true | expires=none"
    assert old in text
    return text.replace(old, "| status=occupied | occupant=maintenance_rig | reason=scheduled_maintenance "
                              "| effective=immediately | expires=none")


def pins(text):
    return [l.strip() for l in text.splitlines() if re.match(r"\s*\**\s*PIN\s*:", l)]


def shuffled(ids, seed):
    q = list(ids)
    random.Random(seed).shuffle(q)
    print(len(q)); print(" ".join(q))


def build(a, W):
    (a.out / "A").mkdir(parents=True, exist_ok=True)
    bm = {}
    for cell in ("BASE", "CHECK", "JIT"):
        for seed in SEEDS:
            for rep in REPS:
                st = trace(a.traces, seed, rep)
                _, d, shelf = positions(st)
                obs = st[d]["observation"]
                if cell == "JIT":
                    obs = f"STANDING NOTICE | shelf={shelf} | status=quarantined | do_not_store=true\n" + obs
                key = f"{cell}-s{seed}-r{rep}"
                b = hashlib.sha1((SALT_1 + key).encode()).hexdigest()[:10]
                (a.out / "A" / f"{b}.txt").write_text(react_prompt(spec(W, seed, CHECK if cell == "CHECK" else ""),
                                                                   history_lines(st[:d]), st[d]["step"], obs))
                bm[b] = dict(key=key, cell=cell, seed=seed, rep=rep, step=st[d]["step"], shelf=shelf,
                             esperado=st[d]["esperado"], ejecutado_api=st[d]["ejecutado"])
    (a.out / "blind.json").write_text(json.dumps(bm, indent=1))
    shuffled(bm, 7)


def two_call(a, W, stage):
    (a.out / stage).mkdir(parents=True, exist_ok=True)
    bm = {}
    for cell in ("PIN", "FACT"):
        for seed in SEEDS:
            for rep in REPS:
                st = trace(a.traces, seed, rep)
                n, d, shelf = positions(st)
                key = f"{cell}-s{seed}-r{rep}"
                w = hashlib.sha1((SALT_2 + "W" + key).encode()).hexdigest()[:10]
                sp = spec(W, seed, PIN if cell == "PIN" else "")
                note = fact(st[n]["observation"]) if cell == "FACT" else st[n]["observation"]
                hist = history_lines(st[:n])
                if stage == "W":
                    (a.out / "W" / f"{w}.txt").write_text(react_prompt(sp, hist, st[n]["step"], note))
                    bm[w] = dict(key=key)
                    continue
                reply = (a.out / "outW" / f"{w}.txt").read_text().strip()
                p_lines = pins(reply) if cell == "PIN" else []
                block = ("PINNED NOTES:\n" + "\n".join(p_lines) + "\n") if p_lines else ""
                hist += [f"Observation: [step {st[n]['step']}] {note}", f"Reasoning & Action: {reply}"]
                hist += [l.replace(f"Observation: [step {r['step']}] ", f"Observation: [step {r['step']}] {block}", 1)
                         if l.startswith("Observation:") else l
                         for r in st[n + 1:d] for l in history_lines([r])]
                dd = hashlib.sha1((SALT_2 + "D" + key).encode()).hexdigest()[:10]
                (a.out / "D" / f"{dd}.txt").write_text(react_prompt(sp, hist, st[d]["step"], block + st[d]["observation"]))
                bm[dd] = dict(key=key, cell=cell, seed=seed, rep=rep, w=w, pins=p_lines,
                              esperado=st[d]["esperado"], shelf=shelf)
    if stage == "D":
        (a.out / "blind_D.json").write_text(json.dumps(bm, indent=1))
    shuffled(bm, stage)


def score(a, Action):
    for name, out_dir in (("blind.json", "outA"), ("blind_D.json", "outD")):
        f = a.out / name
        if not f.exists():
            continue
        res = collections.defaultdict(list)
        for b, it in sorted(json.loads(f.read_text()).items(), key=lambda x: (x[1]["cell"], x[1]["seed"], x[1]["rep"])):
            act = Action.parse((a.out / out_dir / f"{b}.txt").read_text())
            s = act.args.get("shelf") if act is not None and act.name == "Store" else None
            exp = Action.parse(it["esperado"]).args["shelf"]
            res[it["cell"]].append("ok" if s == exp else ("deaf" if s == it["shelf"] else f"other:{s}"))
        for cell, r in res.items():
            c = collections.Counter(r)
            print(f"{cell:5s} correct {c['ok']}/{len(r)}  stored-on-blocked-shelf {c['deaf']}  other {[x for x in r if x.startswith('other')]}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["build", "build-w", "build-d", "score"])
    p.add_argument("--traces", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--src", default=None)
    a = p.parse_args()
    W, Action = import_dr(a.src)
    {"build": lambda: build(a, W), "build-w": lambda: two_call(a, W, "W"), "build-d": lambda: two_call(a, W, "D"),
     "score": lambda: score(a, Action)}[a.stage]()

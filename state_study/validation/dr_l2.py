"""L2 replay: real Haiku ReAct prefixes from the delayed-relevance L2 traces, changing only the notice.

Cells  ORIG  notice as recorded (names the step-0 put-away, whose pallet was already shipped)
       COH   same text, corrects_step pointing at the put-away that holds the shelf (see dr_coherence)
       INT   ORIG plus one spec sentence asking to update occupancy on every event
Call A answers the notice (step 10); call B answers the next inbound pallet (step 11).

    python dr_l2.py build-a --traces T --out OUT
    python dr_l2.py build-b --traces T --out OUT     # after OUT/outA/<id>.txt are written
    python dr_l2.py score   --out OUT
T is the results/ directory extracted from the delayed-relevance traces (see README)."""
import argparse, collections, hashlib, json, random
from pathlib import Path

from dr_common import history_lines, import_dr, load_steps, react_prompt

SALT = "l2fix-91c2"
INTEGRATE = ("\nBefore choosing an action, update your record of shelf occupancy using every event, "
             "including events that need no action.\n")
OCCUPANT = {4: 4, 6: 4, 10: 8}   # step of the put-away holding shelf 0 when the notice arrives
DEAF = {4: 2, 6: 2, 10: 1}       # shelf an agent that ignores the notice stores on at step 11
CELLS, SEEDS, REPS = ("ORIG", "COH", "INT"), (4, 6, 10), range(4)


def trace(t, seed, rep):
    return load_steps(t, f"l2v2_claude-haiku-4-5_react_s{seed}_r{rep}_k10_mt8192.jsonl")


def notice(steps, cell, seed):
    text = steps[10]["observation"]
    assert "correction_notice" in text and "corrects_step=0 " in text
    return text.replace("corrects_step=0 ", f"corrects_step={OCCUPANT[seed]} ") if cell == "COH" else text


def spec(W, seed, cell):
    s = W(horizon=50, seed=seed, invalidation_k=10).spec()
    return s + INTEGRATE if cell == "INT" else s


def norm(a_text):
    t = a_text.strip()
    k = t.find("Reasoning")
    return t[k:] if k > 0 else t


def build_a(a, W):
    (a.out / "A").mkdir(parents=True, exist_ok=True)
    bm = {}
    for cell in CELLS:
        for seed in SEEDS:
            for rep in REPS:
                key = f"{cell}-s{seed}-r{rep}"
                b = hashlib.sha1((SALT + key).encode()).hexdigest()[:10]
                st = trace(a.traces, seed, rep)
                (a.out / "A" / f"{b}.txt").write_text(react_prompt(spec(W, seed, cell), history_lines(st[:10]), 10,
                                                                   notice(st, cell, seed)))
                bm[b] = dict(key=key, cell=cell, seed=seed, rep=rep)
    (a.out / "blind_A.json").write_text(json.dumps(bm, indent=1))
    q = list(bm)
    random.Random(len(q)).shuffle(q)
    print(len(q)); print(" ".join(q))


def build_b(a, W):
    bmA = json.loads((a.out / "blind_A.json").read_text())
    (a.out / "B").mkdir(exist_ok=True)
    bm = {}
    for aid, it in sorted(bmA.items()):
        f = a.out / "outA" / f"{aid}.txt"
        if not f.exists():
            continue
        b = hashlib.sha1((SALT + "B" + aid).encode()).hexdigest()[:10]
        st = trace(a.traces, it["seed"], it["rep"])
        lines = history_lines(st[:10]) + [f"Observation: [step 10] {notice(st, it['cell'], it['seed'])}",
                                         f"Reasoning & Action: {norm(f.read_text())}"]
        (a.out / "B" / f"{b}.txt").write_text(react_prompt(spec(W, it["seed"], it["cell"]), lines, 11,
                                                           st[11]["observation"]))
        bm[b] = dict(it, a_id=aid)
    (a.out / "blind_B.json").write_text(json.dumps(bm, indent=1))
    q = list(bm)
    random.Random(len(q)).shuffle(q)
    print(len(q)); print(" ".join(q))


def score(a, Action):
    bm = json.loads((a.out / "blind_B.json").read_text())
    res, per = collections.defaultdict(list), collections.defaultdict(list)
    for b, it in sorted(bm.items(), key=lambda x: (x[1]["cell"], x[1]["seed"], x[1]["rep"])):
        act = Action.parse((a.out / "outB" / f"{b}.txt").read_text())
        s = act.args.get("shelf") if act is not None and act.name == "Store" else None
        box = "ok" if s == 0 else ("deaf" if s == DEAF[it["seed"]] else f"other:{s}")
        res[it["cell"]].append(box)
        per[(it["cell"], it["seed"])].append(box)
    for cell in CELLS:
        c = collections.Counter(res[cell])
        print(f"{cell:5s} correct {c['ok']}/{len(res[cell])}  deaf {c['deaf']}  other {[x for x in res[cell] if x.startswith('other')]}"
              + "   by seed: " + ", ".join(f"s{s} {per[(cell, s)].count('ok')}/{len(per[(cell, s)])}" for s in SEEDS))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["build-a", "build-b", "score"])
    p.add_argument("--traces", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--src", default=None)
    a = p.parse_args()
    W, Action = import_dr(a.src)
    {"build-a": lambda: build_a(a, W), "build-b": lambda: build_b(a, W), "score": lambda: score(a, Action)}[a.stage]()

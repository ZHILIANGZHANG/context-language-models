"""Self-conditioning probe. After S_T is emptied (forward pick or retroactive void), the log contains
k later arrivals that the agent itself handled as if S_T were still full: it stored them on higher
shelves, optionally with THOUGHT lines restating 'S_T holds ...'. The decision asks for the next
arrival's shelf; the correct answer is still S_T (it is still empty)."""
from __future__ import annotations
import json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import vdepth as V
from retro import RULE, _line, render as _render_unused  # noqa: F401


def generate(seed, form="retro", k=8, thoughts=True, gap=30, n_pre=150):
    rs = random.Random(f"ec-struct-{seed}"); rt = random.Random(f"ec-text-{seed}")
    H = 40; T = rs.randint(3, 6); items = {}; status = {}; tl = []
    def item(s):
        items.setdefault(s, f"I-{rs.randint(100, 999)}{s:03d}"); return items[s]
    def tele():
        tl.append(V.Ev("telemetry", None, None,
            f"Telemetry: zone={rt.choice('ABC')} temp={rt.uniform(14, 24):.1f}C humidity={rt.randint(30, 60)}% "
            f"forklift_F{rt.randint(1, 6)}_battery={rt.randint(20, 99)}% dock_queue={rt.randint(0, 7)} "
            f"conveyor_speed={rt.uniform(0.8, 1.6):.2f}m/s badge_scans={rt.randint(5, 80)} "
            f"hvac_mode={rt.choice(['eco', 'normal', 'boost'])}."))
    def store(s, it, thought=None):
        e = V.Ev("store", s, it, f"Inbound: item {it} arrived at dock {rt.choice('ABC')}{rt.randint(1, 4)} "
                 f"(supplier={rt.choice(V._SUP)}, carrier={rt.choice(V._CAR)}, weight={rt.randint(2, 90)}kg, "
                 f"po=PO-{rt.randint(10000, 99999)}).", action=f"STORE {it} -> S{s}", result=f"OK: {it} stored on S{s}.")
        e.thought = thought; tl.append(e); status[s] = it
    def dispatch(s):
        tl.append(V.Ev("dispatch", s, status[s], f"Outbound: order SO-{rt.randint(10000, 99999)} picked item "
                       f"{status[s]} from S{s} (customer=C-{rt.randint(100, 999)}, carrier={rt.choice(V._CAR)}); "
                       f"S{s} is now EMPTY.")); status[s] = None
    for s in range(1, T + H + 1):
        store(s, item(s))
        if rs.random() < 0.6: tele()
    while len(tl) < n_pre:
        if rs.random() < 0.25:
            dispatch(rs.choice([s for s in range(T + 4, T + H + 1) if status.get(s)]))
        else:
            tele()
    dispatch(T)
    vit = f"I-{rs.randint(100, 999)}{500 + T:03d}"
    store(T, vit); voided = tl[-1]
    for _ in range(rs.randint(4, 8)): tele()
    if form == "forward":
        tl.append(V.Ev("dispatch", T, vit, f"Outbound: order SO-{rt.randint(10000, 99999)} picked item {vit} from "
                       f"S{T} (customer=C-{rt.randint(100, 999)}, carrier={rt.choice(V._CAR)}); S{T} is now EMPTY."))
    else:
        tl.append(V.Ev("correction", T, vit, f"Audit correction AC-{rt.randint(1000, 9999)}: the storage recorded at "
                       f"step {{STEP}} ({vit} -> S{T}) never completed; the pallet was returned to the vendor. "
                       f"S{T} is EMPTY."))
    key = tl[-1]; status[T] = None
    # k later arrivals handled by the agent as if S_T were still occupied (the closed-loop failure)
    rows = list(range(gap))
    slots = sorted(random.Random(f"ec-slots-{seed}").sample(rows, k)) if k else []
    for i in rows:
        if i in slots:
            free = min(s for s in range(1, T + H + 1) if status.get(s) is None and s != T)
            it = f"I-{rs.randint(100, 999)}{600 + i:03d}"
            th = (f"THOUGHT: S1-S{T} are occupied (S{T} holds {vit}); the lowest empty shelf is S{free}."
                  if thoughts else None)
            store(free, it, th)
        elif rs.random() < 0.15:
            occ = [s for s in range(T + 6, T + H + 1) if status.get(s)]
            dispatch(rs.choice(occ))
        else:
            tele()
    for i, e in enumerate(tl, start=1):
        e.step = i
    if form != "forward":
        key.text = key.text.replace("{STEP}", str(voided.step))
    voided.superseded_by = key.step
    answer = min(s for s in status if status[s] is None)
    deaf = min(s for s in status if status[s] is None and s != T)
    assert answer == T
    dec = f"I-{rs.randint(100, 999)}999"
    tl.append(V.Ev("decision", None, dec, f"Inbound: item {dec} arrived at dock A1 (supplier={rs.choice(V._SUP)}, "
                   f"carrier={rs.choice(V._CAR)}, weight={rs.randint(2, 90)}kg).", step=len(tl) + 1))
    return {"events": tl, "answer": T, "deaf": deaf, "status": status}


def render(sc, cond):
    hist = sc["events"][:-1]
    st = {"shelves": {f"S{s}": (v if v else "EMPTY") for s, v in sorted(sc["status"].items())}}
    if cond in ("raw", "tomb"):
        body = "HISTORY LOG:\n" + "\n".join(_line(e, cond) for e in hist)
    elif cond == "state":
        body = "CURRENT STATE:\n" + json.dumps(st, indent=1)
    elif cond == "log_state":
        body = ("HISTORY LOG:\n" + "\n".join(_line(e, "raw") for e in hist) +
                "\n\nCURRENT STATE:\n" + json.dumps(st, indent=1))
    d = sc["events"][-1]
    return (f"{RULE}\n\nYou are the warehouse agent. Your working context follows.\n\n"
            f"=== CONTEXT ===\n{body}\n=== END CONTEXT ===\n\n[step {d.step}] {d.text}\n"
            f"Which shelf do you store item {d.item} on?\n"
            'Reply with JSON only, exactly of the form {"shelf": "S<number>"}.')

"""Form-of-supersession probe. Shelf T is stored (ACTION/RESULT OK), k later arrivals are handled
(optionally with THOUGHT lines that restate 'S_T holds ...'), then S_T becomes EMPTY either by a
forward event (an outbound pick) or by a retroactive audit that voids the storage. The decision is
the first arrival after a gap with no arrivals. Same seed -> same log except for the manipulated lines."""
from __future__ import annotations
import json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import vdepth as V

RULE = ("Warehouse rule: when an item arrives, store it on the lowest-numbered shelf that is "
        "currently EMPTY. Audit corrections override earlier records; the most recent information "
        "about a shelf is its current status.")


def generate(seed, form="retro", thoughts=False, k=8, gap=40, n_pre=150):
    rs = random.Random(f"rt-struct-{seed}"); rt = random.Random(f"rt-text-{seed}")
    H = 40; T = rs.randint(3, 6)
    items = {}
    def item(s):
        items.setdefault(s, f"I-{rs.randint(100, 999)}{s:03d}"); return items[s]
    tele = lambda: V.Ev("telemetry", None, None,
        f"Telemetry: zone={rt.choice('ABC')} temp={rt.uniform(14, 24):.1f}C humidity={rt.randint(30, 60)}% "
        f"forklift_F{rt.randint(1, 6)}_battery={rt.randint(20, 99)}% dock_queue={rt.randint(0, 7)} "
        f"conveyor_speed={rt.uniform(0.8, 1.6):.2f}m/s badge_scans={rt.randint(5, 80)} "
        f"hvac_mode={rt.choice(['eco', 'normal', 'boost'])}.")
    tl = []
    status = {}
    def store(s, it, thought=None):
        txt = (f"Inbound: item {it} arrived at dock {rt.choice('ABC')}{rt.randint(1, 4)} "
               f"(supplier={rt.choice(V._SUP)}, carrier={rt.choice(V._CAR)}, weight={rt.randint(2, 90)}kg, "
               f"po=PO-{rt.randint(10000, 99999)}).")
        e = V.Ev("store", s, it, txt, action=f"STORE {it} -> S{s}", result=f"OK: {it} stored on S{s}.")
        e.thought = thought
        tl.append(e); status[s] = it
    def dispatch(s):
        tl.append(V.Ev("dispatch", s, status[s], f"Outbound: order SO-{rt.randint(10000, 99999)} picked item "
                       f"{status[s]} from S{s} (customer=C-{rt.randint(100, 999)}, carrier={rt.choice(V._CAR)}); "
                       f"S{s} is now EMPTY.")); status[s] = None
    # 1. prefix: fill S1..S_{T+H}, with telemetry and some dispatches above T+3 (refilled later)
    for s in range(1, T + H + 1):
        store(s, item(s))
        if rs.random() < 0.6: tl.append(tele())
    while len(tl) < n_pre:
        u = rs.random()
        if u < 0.25:
            occ = [s for s in range(T + 4, T + H + 1) if status.get(s)]
            dispatch(rs.choice(occ))
        else:
            tl.append(tele())
    # 2. T is emptied and refilled by the record that will be voided
    dispatch(T)
    voided_item = f"I-{rs.randint(100, 999)}{500 + T:03d}"
    store(T, voided_item)
    voided = tl[-1]
    # 3. k arrivals handled after it; they go to the lowest empty shelf above T
    for j in range(k):
        free = min(s for s in range(1, T + H + 1) if status.get(s) is None)
        it = f"I-{rs.randint(100, 999)}{600 + j:03d}"
        th = (f"THOUGHT: S1-S{T} are occupied (S{T} holds {voided_item}); the lowest empty shelf is S{free}."
              if thoughts else None)
        store(free, it, th)
        for _ in range(rs.randint(1, 3)): tl.append(tele())
    # 4. S_T becomes EMPTY: forward pick, or retroactive void of the storage record
    if form == "forward":
        tl.append(V.Ev("dispatch", T, voided_item, f"Outbound: order SO-{rt.randint(10000, 99999)} picked item "
                       f"{voided_item} from S{T} (customer=C-{rt.randint(100, 999)}, carrier={rt.choice(V._CAR)}); "
                       f"S{T} is now EMPTY."))
    else:
        tl.append(V.Ev("correction", T, voided_item, "Audit correction AC-{0}: the storage recorded at step "
                       "{1} ({2} -> S{3}) never completed; the pallet was returned to the vendor. S{3} is EMPTY."
                       .format(rt.randint(1000, 9999), "{STEP}", voided_item, T)))
    key = tl[-1]
    status[T] = None
    # 5. gap without arrivals; dispatches only above T
    for _ in range(gap):
        if rs.random() < 0.2:
            occ = [s for s in range(T + 1, T + H + 1) if status.get(s)]
            dispatch(rs.choice(occ))
        else:
            tl.append(tele())
    for i, e in enumerate(tl, start=1):
        e.step = i
    key.text = key.text.replace("{STEP}", str(voided.step))
    voided.superseded_by = key.step
    answer = min(s for s in status if status[s] is None)
    deaf = min(s for s in status if status[s] is None and s != T)
    assert answer == T and deaf != T
    dec = f"I-{rs.randint(100, 999)}999"
    tl.append(V.Ev("decision", None, dec, f"Inbound: item {dec} arrived at dock A1 (supplier={rs.choice(V._SUP)}, "
                   f"carrier={rs.choice(V._CAR)}, weight={rs.randint(2, 90)}kg).", step=len(tl) + 1))
    return {"events": tl, "answer": T, "deaf": deaf, "status": status, "voided_step": voided.step, "key_step": key.step}


def _line(e, mode):
    s = f"[step {e.step}] {e.text}"
    th = getattr(e, "thought", None)
    if th: s += f"\n    {th}"
    if e.action: s += f"\n    ACTION: {e.action}\n    RESULT: {e.result}"
    if mode == "tomb" and e.superseded_by is not None:
        s += f"\n    [VOID: this storage never completed, see step {e.superseded_by}]"
    return s


def render(sc, cond):
    hist = sc["events"][:-1]
    if cond in ("raw", "tomb"):
        body = "HISTORY LOG:\n" + "\n".join(_line(e, cond) for e in hist)
    elif cond == "state":
        st = {"shelves": {f"S{s}": (v if v else "EMPTY") for s, v in sorted(sc["status"].items())}}
        body = "CURRENT STATE:\n" + json.dumps(st, indent=1)
    elif cond == "log_state":
        st = {"shelves": {f"S{s}": (v if v else "EMPTY") for s, v in sorted(sc["status"].items())}}
        body = "HISTORY LOG:\n" + "\n".join(_line(e, "raw") for e in hist) + "\n\nCURRENT STATE:\n" + json.dumps(st, indent=1)
    d = sc["events"][-1]
    return (f"{RULE}\n\nYou are the warehouse agent. Your working context follows.\n\n"
            f"=== CONTEXT ===\n{body}\n=== END CONTEXT ===\n\n[step {d.step}] {d.text}\n"
            f"Which shelf do you store item {d.item} on?\n"
            'Reply with JSON only, exactly of the form {"shelf": "S<number>"}.')

"""Version-depth probe: the decision depends on the current status of a few low shelves whose
status flipped f times earlier in the log. Everything else (event count, token count, position of
each shelf's latest version, the answer) is held fixed across f, so any change in accuracy is due to
the number of superseded versions of decision-relevant entities.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field

RULE = ("Warehouse rule: when an item arrives, store it on the lowest-numbered shelf that is "
        "currently EMPTY. Inspections, returns, cycle counts and outbound picks change or confirm "
        "a shelf's status; the most recent event about a shelf is its current status.")

_SUP = ["Arcadia Foods", "Borealis Parts", "Cobalt Medical", "Delta Textiles", "Equinox Labs",
        "Fjord Marine", "Granite Tools", "Helix Pharma"]
_CAR = ["DHL", "UPS", "Maersk", "FedEx", "Kuehne+Nagel"]
_REASON = ["moisture check", "label audit", "seal check", "weight recheck", "temperature log review"]


@dataclass
class Ev:
    kind: str            # store | out | ret | cycle | dispatch | telemetry | decision
    shelf: int | None
    item: str | None
    text: str = ""
    step: int = 0
    action: str | None = None
    result: str | None = None
    superseded_by: int | None = None


@dataclass
class VD:
    seed: int
    f: int
    answer: int
    n_rel: int
    events: list = field(default_factory=list)
    shelves: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


def generate(seed: int, f: int, *, H: int = 30, flip_budget: int = 168, tele_flip: int = 60,
             settle: int = 40, tele_setup: int = 20, irrelevant: bool = True,
             tele_mult: float = 1.0) -> VD:
    """f flip pairs (out + return) on each of the shelves S1..Sa; Sa is finally dispatched."""
    rs = random.Random(f"vd-struct-{seed}")          # structure shared by every f
    a = rs.randint(4, 7)
    rel = list(range(1, a + 1))
    high = list(range(a + 1, a + 1 + H))
    items = {s: f"I-{rs.randint(100, 999)}{s:03d}" for s in rel + high}
    # final-touch position of every relevant shelf except Sa (same across f)
    final_pos = {s: rs.uniform(0.55, 0.95) for s in rel if s != a}
    rp = random.Random(f"vd-pos-{seed}-{f}")          # positions of flips (depend on f)
    rt = random.Random(f"vd-text-{seed}")             # text details (shared stream; draws vary)

    def telemetry():
        return Ev("telemetry", None, None,
                  f"Telemetry: zone={rt.choice('ABC')} temp={rt.uniform(14, 24):.1f}C "
                  f"humidity={rt.randint(30, 60)}% forklift_F{rt.randint(1, 6)}_battery={rt.randint(20, 99)}% "
                  f"dock_queue={rt.randint(0, 7)} conveyor_speed={rt.uniform(0.8, 1.6):.2f}m/s "
                  f"badge_scans={rt.randint(5, 80)} hvac_mode={rt.choice(['eco', 'normal', 'boost'])}.")

    timeline: list[Ev] = []
    # 1. setup: oracle stores items on S1..S_{a+H} in order, with telemetry interleaved
    setup_tele = int(tele_setup * tele_mult)
    order = [("store", s) for s in rel + high] + [("telemetry", None)] * setup_tele
    head = order[:2]
    rest = order[2:]
    rs2 = random.Random(f"vd-setup-{seed}-{tele_mult}")
    rs2.shuffle(rest)
    # keep stores in shelf order (the rule fills the lowest empty shelf first)
    stores = iter(s for k, s in order if k == "store")
    for k, _ in head + rest:
        if k == "telemetry":
            timeline.append(telemetry())
        else:
            s = next(stores)
            it = items[s]
            txt = (f"Inbound: item {it} arrived at dock {rt.choice('ABC')}{rt.randint(1, 4)} "
                   f"(supplier={rt.choice(_SUP)}, carrier={rt.choice(_CAR)}, weight={rt.randint(2, 90)}kg, "
                   f"po=PO-{rt.randint(10000, 99999)}).")
            timeline.append(Ev("store", s, it, txt, action=f"STORE {it} -> S{s}",
                               result=f"OK: {it} stored on S{s}."))

    # 2. flip phase: positions are floats in [0, 1); sorted later
    flips: list[tuple[float, Ev]] = []

    def out_ev(s):
        t = rt.randint(1000, 9999)
        return Ev("out", s, items[s], f"Inspection QA-{t}: item {items[s]} on S{s} sent to QA lab "
                                      f"(reason={rt.choice(_REASON)}); S{s} is now EMPTY.")

    def ret_ev(s):
        return Ev("ret", s, items[s], f"Inspection: item {items[s]} returned from QA lab to S{s} "
                                      f"(inspector=E-{rt.randint(10, 99)}); S{s} is OCCUPIED again.")

    def cycle_ev(s):
        return Ev("cycle", s, items[s], f"Cycle count CC-{rt.randint(1000, 9999)}: S{s} verified, "
                                        f"holds {items[s]} (OCCUPIED).")

    n_rel_events = 0
    for s in rel:
        end = final_pos.get(s, 0.95)
        if f == 0:
            if s != a:
                flips.append((end, cycle_ev(s)))
                n_rel_events += 1
            continue
        pts = sorted(rp.uniform(0.02, end) for _ in range(2 * f - 1)) + [end]
        if s == a:  # Sa's flips can sit anywhere before the settle phase
            pts = sorted(rp.uniform(0.02, 0.98) for _ in range(2 * f))
        for i, p in enumerate(pts):
            flips.append((p, out_ev(s) if i % 2 == 0 else ret_ev(s)))
        n_rel_events += 2 * f
    n_irr = flip_budget - n_rel_events
    if not irrelevant:
        n_irr_pairs = 0
    else:
        n_irr_pairs = max(0, n_irr // 2)
    # irrelevant pairs on high shelves, non-overlapping per shelf
    per_shelf: dict[int, list[float]] = {s: [] for s in high}
    for _ in range(n_irr_pairs):
        s = rp.choice(high)
        per_shelf[s] += [rp.uniform(0.0, 1.0), rp.uniform(0.0, 1.0)]
    for s, pts in per_shelf.items():
        pts.sort()
        for i, p in enumerate(pts):
            flips.append((p, out_ev(s) if i % 2 == 0 else ret_ev(s)))
    n_tele = int(tele_flip * tele_mult) + (flip_budget - n_rel_events - 2 * n_irr_pairs)
    for _ in range(n_tele):
        flips.append((rp.uniform(0.0, 1.0), telemetry()))
    flips.sort(key=lambda x: x[0])
    timeline += [e for _, e in flips]

    # 3. settle: Sa dispatched, then only high-shelf events and telemetry
    def dispatch_ev(s):
        return Ev("dispatch", s, items[s],
                  f"Outbound: order SO-{rt.randint(10000, 99999)} picked item {items[s]} from S{s} "
                  f"(customer=C-{rt.randint(100, 999)}, carrier={rt.choice(_CAR)}); S{s} is now EMPTY.")

    timeline.append(dispatch_ev(a))
    late = [dispatch_ev(s) for s in rs.sample(high, 4)]
    filler = [telemetry() for _ in range(int((settle - 4) * tele_mult))]
    rs3 = random.Random(f"vd-settle-{seed}-{tele_mult}")
    tail = late + filler
    rs3.shuffle(tail)
    timeline += tail

    # number the steps and mark superseded entries
    for i, e in enumerate(timeline, start=1):
        e.step = i
    last: dict[int, Ev] = {}
    for e in timeline:
        if e.shelf is None:
            continue
        prev = last.get(e.shelf)
        if prev is not None and e.kind in ("out", "ret", "dispatch"):
            prev.superseded_by = e.step
        last[e.shelf] = e
    state = {}
    for s in rel + high:
        e = last[s]
        state[s] = None if e.kind in ("out", "dispatch") else items[s]
    assert all(state[s] is not None for s in rel if s != a) and state[a] is None
    dec_item = f"I-{rs.randint(100, 999)}{999:03d}"
    timeline.append(Ev("decision", None, dec_item,
                       f"Inbound: item {dec_item} arrived at dock A1 (supplier={rs.choice(_SUP)}, "
                       f"carrier={rs.choice(_CAR)}, weight={rs.randint(2, 90)}kg).", step=len(timeline) + 1))
    sc = VD(seed=seed, f=f, answer=a, n_rel=a, events=timeline, shelves=state)
    sc.meta = {"versions_per_rel_shelf": 2 * f + 1, "n_events": len(timeline) - 1,
               "n_rel_flip_events": n_rel_events, "n_irr_flip_events": 2 * n_irr_pairs}
    return sc


def _line(e: Ev, mode: str) -> str:
    s = f"[step {e.step}] {e.text}"
    if e.action:
        s += f"\n    ACTION: {e.action}\n    RESULT: {e.result}"
    if mode == "tomb" and e.superseded_by is not None:
        s += f"\n    [SUPERSEDED: S{e.shelf} changed again at step {e.superseded_by}]"
    return s


def render(sc: VD, cond: str) -> str:
    hist = sc.events[:-1]
    if cond in ("raw", "tomb"):
        body = "HISTORY LOG:\n" + "\n".join(_line(e, cond) for e in hist)
    elif cond == "latest":
        body = "HISTORY LOG (superseded entries removed):\n" + "\n".join(
            _line(e, "raw") for e in hist if e.superseded_by is None)
    elif cond == "state":
        st = {"shelves": {f"S{s}": (v if v is not None else "EMPTY") for s, v in sorted(sc.shelves.items())}}
        body = "CURRENT STATE:\n" + json.dumps(st, indent=1)
    else:
        raise ValueError(cond)
    d = sc.events[-1]
    return (f"{RULE}\n\nYou are the warehouse agent. Your working context follows.\n\n"
            f"=== CONTEXT ===\n{body}\n=== END CONTEXT ===\n\n"
            f"[step {d.step}] {d.text}\nWhich shelf do you store item {d.item} on?\n"
            'Reply with JSON only, exactly of the form {"shelf": "S<number>"}.')

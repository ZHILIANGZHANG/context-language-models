"""Closed-loop warehouse environment for deferred-relevance tests with subagents (written from scratch).

A subagent plays a whole episode through this CLI, one command per event, so the history it acts on
is its own transcript (its own reasoning, no other model's text). At step NOTICE a facility notice
quarantines shelf S (do_not_store=true). The first step at which honoring the notice changes the
correct action is DECISION, 40 steps later: an inbound pallet arrives while S is the lowest empty
shelf, so the correct action skips S. In every generated script S is never touched in between.

Conditions (the agent only ever sees the procedure text and the events):
  BASE  the procedure never mentions facility notices
  RULE  the procedure has a rule for them (the notice is "in slot" for the procedure)
  PIN   BASE + the act command takes --pin '<note>'; pinned notes are shown above every later event
  REM   BASE + oracle reminder: every event after the notice starts with a STANDING NOTICE line

Physically impossible actions are rejected with a reason and the same event is asked again, up to
MAX_ATTEMPTS times. Storing on the quarantined shelf is physically possible, so it is never rejected.

    python wh_env.py --root DIR new EP --seed N --cond BASE     # by the experimenter
    python wh_env.py --root DIR start EP                        # by the agent: procedure + first event
    python wh_env.py --root DIR act EP 'Store({...})' [--pin 'text']
    python wh_env.py --root DIR score                           # by the experimenter
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

N_SHELVES, HORIZON, NOTICE, DECISION, MAX_ATTEMPTS = 100, 50, 8, 48, 3
CONDS = ("BASE", "RULE", "PIN", "REM")
SKUS = [f"SKU-{c}" for c in "ABCDEFGHJKLMN"]
CARRIERS = ["NORDLINE", "OAKLINE", "MERIDIAN", "HALVARD", "BLUEPEAK"]
SITES = ["Lodz-03", "Porto-11", "Lyon-07", "Gdansk-02", "Turin-05"]
SOURCES = ["hvac_zone_b", "humidity_north", "dock_door_3", "battery_bank_2", "scanner_hub_1"]


# ---------------------------------------------------------------- generation and ground truth
def _lowest_empty(shelves, skip=None):
    return next(i for i in range(N_SHELVES) if shelves[i] is None and i != skip)


def _draft(rng):
    """Event skeletons (kind, sku, units, lot) with outbound SKUs valid under the oracle."""
    shelves, events = [None] * N_SHELVES, []
    for t in range(HORIZON):
        held = sorted({v for v in shelves if v})
        u = rng.random()
        if t == NOTICE:
            kind = "notice"
        elif t == DECISION:
            kind = "inbound"
        elif u < 0.45:
            kind = "inbound"
        elif u < 0.72 and held:
            kind = "outbound"
        else:
            kind = "telemetry"
        ev = {"kind": kind}
        if kind == "inbound":
            ev.update(sku=rng.choice(SKUS), units=rng.randint(1, 20), lot=f"L-{rng.randint(1000, 9999)}")
            shelves[_lowest_empty(shelves)] = ev["sku"]
        elif kind == "outbound":
            ev["sku"] = rng.choice(held)
            shelves[shelves.index(ev["sku"])] = None
        events.append(ev)
    return events


def expected(shelves, ev, quarantined):
    """Correct action given the true shelves and the shelf under quarantine (None before the notice)."""
    if ev["kind"] == "inbound":
        s = _lowest_empty(shelves, skip=quarantined)
        return {"name": "Store", "args": {"shelf": s, "sku": ev["sku"], "units": ev["units"], "lot": ev["lot"]}}
    if ev["kind"] == "outbound" and ev["sku"] in shelves:
        return {"name": "Ship", "args": {"shelf": shelves.index(ev["sku"]), "sku": ev["sku"]}}
    return {"name": "Wait", "args": {}}


def check(shelves, ev, act):
    """None if the action is physically possible for this event, else the rejection reason."""
    if act is None:
        return "the action could not be parsed"
    if act["name"] == "Wait":
        return None
    s = act["args"].get("shelf")
    if not isinstance(s, int) or isinstance(s, bool) or not 0 <= s < N_SHELVES:
        return "shelf must be an integer from 0 to 99"
    if act["name"] == "Store":
        if ev["kind"] != "inbound":
            return "there is no arriving pallet to store"
        if shelves[s] is not None:
            return f"shelf {s} is occupied"
        if [act["args"].get(k) for k in ("sku", "units", "lot")] != [ev["sku"], ev["units"], ev["lot"]]:
            return "sku, units and lot must be copied from the event"
        return None
    if ev["kind"] != "outbound":
        return "there is no order to ship"
    if act["args"].get("sku") != ev["sku"]:
        return "sku must be the ordered SKU"
    if shelves[s] != ev["sku"]:
        return f"shelf {s} does not hold {ev['sku']}"
    return None


def apply(shelves, act):
    """Apply an accepted action to the true shelves."""
    if act["name"] == "Store":
        shelves[act["args"]["shelf"]] = act["args"]["sku"]
    elif act["name"] == "Ship":
        shelves[act["args"]["shelf"]] = None


def generate(seed):
    """Rejection-sample a script where the quarantine first matters exactly at DECISION."""
    rng = random.Random(f"wh-env-{seed}")
    for _ in range(10000):
        events = _draft(rng)
        shelves = [None] * N_SHELVES
        for ev in events[:DECISION]:
            apply(shelves, expected(shelves, ev, None))
        s = _lowest_empty(shelves)
        events[NOTICE]["shelf"] = s
        shelves, ok = [None] * N_SHELVES, True
        for t, ev in enumerate(events[:DECISION + 1]):
            q = s if t > NOTICE else None
            a, b = expected(shelves, ev, q), expected(shelves, ev, None)
            if (a != b) != (t == DECISION):
                ok = False
                break
            apply(shelves, a)
        if ok:
            return {"seed": seed, "events": events, "quarantined": s}
    raise RuntimeError(f"seed {seed}: no script found")


# ---------------------------------------------------------------- rendering
def _meta(rng):
    return (f"\n  audit_id=AUD-{rng.randint(100000, 999999)} | recorded_at=2026-09-{rng.randint(10, 28)}T"
            f"{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:00Z | source={rng.choice(['WMS-1', 'WMS-2', 'EDGE-4'])} "
            f"| schema={rng.randint(3, 9)}.{rng.randint(0, 9)} | latency_ms={rng.randint(5, 900)} | retries={rng.randint(0, 3)}"
            f"\n  reviewer=auditor_{rng.randint(100, 999)} | review={rng.choice(['auto_ok', 'sampled_ok', 'not_sampled'])} "
            f"| retention_years={rng.randint(3, 10)} | tag={rng.choice(['iso9001', 'iso14001', 'none'])} "
            f"| batch=ARC-{rng.randint(1000, 9999)} | queue={rng.choice(['in.q', 'out.q', 'tel.q'])} "
            f"| partition={rng.randint(0, 15)} | offset={rng.randint(10000, 999999)}"
            f"\n  correlation=CG-{rng.randint(10000, 99999)} | region={rng.choice(['north', 'south', 'east', 'west'])} "
            f"| ack={rng.choice(['pending', 'received'])} | replay={rng.randint(0, 2)} | env=production | end_of_record=true")


def render(ev, t, seed):
    rng = random.Random(f"wh-render-{seed}-{t}")
    if ev["kind"] == "inbound":
        head = (f"EVENT inbound_pallet | pallet_id=PAL-{rng.randint(1000, 9999)} | sku={ev['sku']} | units={ev['units']} "
                f"| lot={ev['lot']} | carrier={rng.choice(CARRIERS)} | dock={rng.randint(1, 9)} "
                f"| temperature_c={rng.uniform(2, 9):.1f} | supplier_site={rng.choice(SITES)} | gross_kg={rng.uniform(80, 900):.1f}")
    elif ev["kind"] == "outbound":
        head = (f"EVENT outbound_order | order_id=ORD-{rng.randint(1000, 9999)} | sku={ev['sku']} "
                f"| units_requested={rng.randint(1, 20)} | customer={rng.choice(['ACME', 'BRIGHTFOLD', 'KESTREL'])} "
                f"| service_level={rng.choice(['economy', 'standard', 'expedited'])} | label_printed=true")
    elif ev["kind"] == "notice":
        head = (f"EVENT facility_notice | notice_id=FAC-{rng.randint(1000, 9999)} | shelf={ev['shelf']} | status=quarantined "
                f"| reason=scheduled_maintenance | effective=immediately | do_not_store=true | expires=none "
                f"| raised_by=facilities_{rng.randint(100, 999)} | work_order=WO-{rng.randint(10000, 99999)}")
    else:
        head = (f"EVENT telemetry | source={rng.choice(SOURCES)} | value={rng.uniform(10, 70):.1f} "
                f"| status=within_band | window=15m | requires_action=false")
    return f"[step {t}] " + head + _meta(rng)


SPEC = """WAREHOUSE CONTROLLER PROCEDURE
The warehouse has 100 shelves, numbered 0 to 99. A shelf holds exactly one pallet or nothing. All
shelves are empty at the start. You receive one event per step and must answer it with exactly one
action.

EVENT FORMAT
An event is a pipe-separated list of key=value fields that starts with the event type. Most fields
are operational metadata (carrier, dock, audit trail, queue offsets and so on). Look at the type
first, then use the fields this procedure names.

EVENT TYPES
  inbound_pallet - a pallet has arrived and must be put away. Fields used: sku, units, lot.
  outbound_order - an order must be shipped from stock. Field used: sku.
  telemetry      - an equipment reading. It never requires an action.
{rule_type}
ACTIONS
  Store({{"shelf": <int>, "sku": "<str>", "units": <int>, "lot": "<str>"}})
      Put the arriving pallet on the lowest-numbered shelf that is currently empty. Copy sku, units
      and lot from the event.
  Ship({{"shelf": <int>, "sku": "<str>"}})
      Ship from the lowest-numbered shelf that holds the ordered SKU. Shipping empties the shelf.
  Wait({{}})
      The event needs no action.

PROCEDURE
  1. inbound_pallet: Store on the lowest-numbered empty shelf. Shelves freed by shipping are reused
     before higher-numbered shelves that were never used.
  2. outbound_order: Ship from the lowest-numbered shelf holding that SKU; if it is not in stock, Wait({{}}).
  3. telemetry: Wait({{}}).
{rule_step}
An action that is physically impossible (for example storing on an occupied shelf) is rejected
with a reason, and you answer the same event again.
"""

RULE_TYPE = "  facility_notice - a shelf restriction from facilities. Fields used: shelf, do_not_store.\n"
RULE_STEP = ("  4. facility_notice: Wait({}). If it says do_not_store=true, never Store on that shelf afterwards;\n"
             "     use the lowest-numbered empty shelf that is not restricted.\n")

PIN_SPEC = """
PINNED NOTES
If an event establishes a rule or restriction that could affect later actions, attach a note to
your action with --pin '<note>'. Every pinned note is shown to you again above each later event,
under the heading PINNED NOTES.
"""


def procedure(cond):
    rule = cond == "RULE"
    text = SPEC.format(rule_type=RULE_TYPE if rule else "", rule_step=RULE_STEP if rule else "")
    return text + (PIN_SPEC if cond == "PIN" else "")


# ---------------------------------------------------------------- CLI
def _path(root, ep):
    return root / f".{ep}.state.json"


def _load(root, ep):
    return json.loads(_path(root, ep).read_text())


def _save(root, ep, st):
    _path(root, ep).write_text(json.dumps(st))


def parse_action(text):
    m = re.search(r"\b(Store|Ship|Wait)\s*\(\s*(\{.*\})\s*\)\s*$", text.strip(), re.S)
    if not m:
        return None
    try:
        args = json.loads(m.group(2))
    except json.JSONDecodeError:
        return None
    return {"name": m.group(1), "args": args} if isinstance(args, dict) else None


def world(st):
    """True shelves after replaying the accepted actions."""
    shelves = [None] * N_SHELVES
    for rec in st["log"]:
        if rec["accepted"]:
            apply(shelves, rec["parsed"])
    return shelves


def observation(st):
    t = st["t"]
    sc = generate(st["seed"])
    text = render(sc["events"][t], t, st["seed"])
    if st["cond"] == "REM" and t > NOTICE:
        text = f"STANDING NOTICE | shelf={sc['quarantined']} | status=quarantined | do_not_store=true\n" + text
    if st["cond"] == "PIN" and st["pins"]:
        text = "PINNED NOTES:\n" + "\n".join(f"- {p}" for p in st["pins"]) + "\n" + text
    return text


def cmd_new(root, ep, seed, cond):
    root.mkdir(parents=True, exist_ok=True)
    _save(root, ep, {"seed": seed, "cond": cond, "t": 0, "attempt": 0, "pins": [], "log": []})


def cmd_start(root, ep):
    st = _load(root, ep)
    if st["t"] != 0 or st["log"]:
        return "This episode has already started; answer the latest event with the act command."
    return procedure(st["cond"]) + "\nFIRST EVENT\n" + observation(st)


def cmd_act(root, ep, action_text, pins=()):
    st = _load(root, ep)
    if st["t"] >= HORIZON:
        return "EPISODE COMPLETE"
    sc = generate(st["seed"])
    shelves, t = world(st), st["t"]
    ev = sc["events"][t]
    exp = expected(shelves, ev, sc["quarantined"] if t > NOTICE else None)
    act = parse_action(action_text)
    reason = check(shelves, ev, act)
    pins = [p for p in pins if p.strip()]
    shown = pins if st["cond"] == "PIN" else []          # --pin has no effect outside PIN
    st["log"].append({"t": t, "attempt": st["attempt"], "kind": ev["kind"], "raw": action_text, "parsed": act,
                      "accepted": reason is None, "reason": reason, "expected": exp, "correct": act == exp,
                      "dependent": ev["kind"] == "inbound" and t > NOTICE and _lowest_empty(shelves) == sc["quarantined"],
                      "pins_added": shown, "pins_ignored": pins if not shown else []})
    st["pins"] += shown
    if reason is not None and st["attempt"] + 1 < MAX_ATTEMPTS:
        st["attempt"] += 1
        _save(root, ep, st)
        return f"ACTION REJECTED | {reason} | the action had no effect; answer the same event again\n" + observation(st)
    head = "ACCEPTED" if reason is None else f"ACTION REJECTED | {reason} | the event was skipped"
    st["t"], st["attempt"] = t + 1, 0
    _save(root, ep, st)
    if st["t"] >= HORIZON:
        return head + "\nEPISODE COMPLETE"
    return head + "\nNEXT EVENT\n" + observation(st)


def mentions(note, shelf):
    return bool(re.search(rf"(?<!\d){shelf}(?!\d)", note)) and bool(re.search(r"quarantin|do.?not.?store|restrict", note, re.I))


def score_state(st):
    sc = generate(st["seed"])
    s = sc["quarantined"]
    first = [r for r in st["log"] if r["attempt"] == 0]
    dep = [r for r in first if r["dependent"]]
    d = dep[0] if dep else None
    if d is None:
        decision = "none"
    elif d["correct"]:
        decision = "honor"
    elif d["parsed"] and d["parsed"]["name"] == "Store" and d["parsed"]["args"].get("shelf") == s:
        decision = "violate"
    else:
        decision = "other"
    before = [r for r in first if d is None or r["t"] < d["t"]]
    return {"cond": st["cond"], "seed": st["seed"], "quarantined": s, "steps": len(first),
            "complete": st["t"] >= HORIZON, "decision": decision, "decision_step": d["t"] if d else None,
            "decision_action": d["raw"] if d else None,
            "on_track": all(r["correct"] for r in before),
            "step_acc": round(sum(r["correct"] for r in first) / max(len(first), 1), 3),
            "rejections": sum(not r["accepted"] for r in st["log"]),
            "violations": sum(1 for r in st["log"] if r["accepted"] and r["t"] > NOTICE and r["parsed"]["name"] == "Store"
                              and r["parsed"]["args"]["shelf"] == s),
            "pinned_at_notice": any(mentions(p, s) for r in st["log"] if r["t"] == NOTICE for p in r["pins_added"]),
            "pinned_ever": any(mentions(p, s) for p in st["pins"]), "n_pins": len(st["pins"])}


def cmd_score(root):
    return [{"ep": f.name[1:-len(".state.json")], **score_state(json.loads(f.read_text()))}
            for f in sorted(root.glob(".*.state.json"))]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    sub = p.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("new"); n.add_argument("ep"); n.add_argument("--seed", type=int, required=True)
    n.add_argument("--cond", choices=CONDS, required=True)
    s = sub.add_parser("start"); s.add_argument("ep")
    a = sub.add_parser("act"); a.add_argument("ep"); a.add_argument("action")
    a.add_argument("--pin", action="append", default=[])
    sub.add_parser("score")
    args = p.parse_args(argv)
    if args.cmd == "new":
        cmd_new(args.root, args.ep, args.seed, args.cond)
    elif args.cmd == "start":
        print(cmd_start(args.root, args.ep))
    elif args.cmd == "act":
        print(cmd_act(args.root, args.ep, args.action, args.pin))
    else:
        for r in cmd_score(args.root):
            print(json.dumps(r))


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    main()

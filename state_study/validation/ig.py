"""Integration-gate probe: a two-call closed loop written from scratch (no third-party code).

Call A: the latest observation is a correction notice that empties shelf s (a put-away recorded
earlier never completed). Call B: the next inbound pallet; the correct shelf is s.
Representation H keeps a ReAct transcript (observations + the agent's own reasoning lines); S shows
only the spec, a JSON state and the latest observation, and asks for a state patch plus an action.
Spec variant 'covered' lists correction_notice as an event type; 'uncovered' does not. Variant
'integrate' (H only) adds one generic sentence asking the agent to update its occupancy on every event.
"""
from __future__ import annotations
import json, random, re

SKUS = [f"SKU-{c}" for c in "ABCDEFGHJKLMN"]


def spec(covered: bool, integrate: bool = False, filt: bool = False) -> str:
    s = ("WAREHOUSE PROCEDURE\nShelves are numbered 0 to 99; each holds one pallet or nothing. You receive one event "
         "per step and reply with exactly one action.\n\n")
    if filt:
        s += ("EVENT FORMAT\nEvery event is a pipe-separated record that begins with its type. Most fields are "
              "operational metadata that do not affect your action. Read the type first, then use only the "
              "fields the procedure below tells you to use.\n\n")
    s += ("EVENT TYPES\n"
         "  inbound_pallet - a pallet has arrived and must be put away. Load-bearing fields: sku, units, lot.\n"
         "  outbound_order - an order must be shipped from stock. Load-bearing field: sku.\n"
         "  telemetry - an equipment reading. Telemetry never requires an action.\n")
    if covered:
        s += ("  correction_notice - an audit found that an earlier record was wrong. Load-bearing fields: shelf, "
              "shelf_is_now. Update your occupancy accordingly; it requires no action.\n")
    s += ("\nACTIONS\n  Store({\"shelf\": <int>, \"sku\": \"<str>\", \"units\": <int>, \"lot\": \"<str>\"}) - put an "
          "inbound pallet on the lowest-numbered shelf that is currently empty.\n"
          "  Ship({\"shelf\": <int>, \"sku\": \"<str>\"}) - ship from the lowest-numbered shelf holding that SKU; "
          "shipping empties the shelf.\n  Wait({}) - the event requires no action.\n\n"
          "PROCEDURE\n  1. On inbound_pallet: Store on the lowest-numbered empty shelf.\n"
          "  2. On outbound_order: Ship from the shelf holding that SKU; if it is not in stock, Wait({}).\n"
          "  3. On telemetry: Wait({}).\n")
    if integrate:
        s += ("\nBefore choosing an action, update your record of shelf occupancy using every event, including "
              "events that need no action.\n")
    return s


def generate(seed: int, n_steps: int = 36):
    r = random.Random(f"ig-{seed}")
    shelves: dict[int, tuple | None] = {i: None for i in range(100)}
    steps = []
    def lowest():
        return min(i for i, v in shelves.items() if v is None)
    def occ_text():
        occ = [f"{i}:{v[0]}" for i, v in sorted(shelves.items()) if v]
        return "shelves " + ", ".join(occ) if occ else "all shelves empty"
    voided = None
    for t in range(n_steps):
        u = r.random()
        if t == n_steps - 10:   # the put-away that will be voided: lowest empty must be a low shelf
            kind = "inbound"
        elif u < 0.45: kind = "inbound"
        elif u < 0.70 and any(shelves.values()): kind = "outbound"
        else: kind = "telemetry"
        if kind == "inbound":
            sku, units, lot = r.choice(SKUS), r.randint(1, 20), f"L-{r.randint(1000, 9999)}"
            s = lowest()
            obs = (f"[step {t}] EVENT inbound_pallet | pallet_id=PAL-{r.randint(1000, 9999)} | sku={sku} | units={units} "
                   f"| lot={lot} | carrier={r.choice(['NORDLINE', 'OAKLINE', 'MERIDIAN'])} | dock={r.randint(1, 9)}")
            act = f'Store({{"shelf": {s}, "sku": "{sku}", "units": {units}, "lot": "{lot}"}})'
            reason = (f"Reasoning: step {t}, inbound_pallet with sku={sku}. Current occupancy: {occ_text()}. "
                      f"The lowest-numbered empty shelf is {s}. Action: {act}")
            shelves[s] = (sku, units, lot)
            if t == n_steps - 10: voided = (t, s)
        elif kind == "outbound":
            held = [(i, v) for i, v in sorted(shelves.items()) if v]
            if voided is not None:  # after the voided put-away, never empty a shelf at or below it
                low_skus = {w[0] for j, w in shelves.items() if w and j <= voided[1]}
                held = [(i, v) for i, v in held if i > voided[1] and v[0] not in low_skus]
                if not held:
                    held = None
            if not held:
                obs = (f"[step {t}] EVENT telemetry | source=hvac_zone_b | value={r.uniform(10, 70):.1f} | status=within_band")
                act = "Wait({})"
                reason = f"Reasoning: step {t}, telemetry. Telemetry never requires an action. Action: {act}"
                steps.append({"obs": obs, "reason": reason, "act": act})
                continue
            i, v = r.choice(held)
            obs = (f"[step {t}] EVENT outbound_order | order_id=ORD-{r.randint(1000, 9999)} | sku={v[0]} | "
                   f"customer={r.choice(['ACME', 'BRIGHTFOLD', 'KESTREL'])}")
            i = min(j for j, w in shelves.items() if w and w[0] == v[0])
            act = f'Ship({{"shelf": {i}, "sku": "{v[0]}"}})'
            reason = (f"Reasoning: step {t}, outbound_order for {v[0]}. Current occupancy: {occ_text()}. "
                      f"{v[0]} is on shelf {i}. Action: {act}")
            shelves[i] = None
        else:
            obs = (f"[step {t}] EVENT telemetry | source={r.choice(['hvac_zone_b', 'humidity_north', 'dock_door_3'])} | "
                   f"value={r.uniform(10, 70):.1f} | status=within_band")
            act = "Wait({})"
            reason = f"Reasoning: step {t}, telemetry. Telemetry never requires an action. Action: {act}"
        steps.append({"obs": obs, "reason": reason, "act": act})
    # keep the voided shelf the lowest empty shelf after the correction: no outbound may empty a lower shelf
    t_c = n_steps
    vt, vs = voided
    corr = (f"[step {t_c}] EVENT correction_notice | notice_id=COR-{r.randint(1000, 9999)} | corrects_step={vt} | "
            f"shelf={vs} | finding=putaway_never_completed | shelf_is_now=empty | stock_removed=true")
    state_before = {i: v for i, v in shelves.items() if v}
    shelves[vs] = None
    sku, units, lot = r.choice(SKUS), r.randint(1, 20), f"L-{r.randint(1000, 9999)}"
    dec = (f"[step {t_c + 1}] EVENT inbound_pallet | pallet_id=PAL-{r.randint(1000, 9999)} | sku={sku} | "
           f"units={units} | lot={lot} | carrier=NORDLINE | dock=4")
    answer = min(i for i, v in shelves.items() if v is None)
    deaf = min(i for i in range(100) if i not in state_before)
    return {"steps": steps, "corr": corr, "dec": dec, "answer": answer, "deaf": deaf, "vs": vs,
            "state_before": state_before}


def _state_json(st: dict) -> str:
    return json.dumps({"shelf_contents": {str(i): {"sku": v[0], "units": v[1], "lot": v[2]} for i, v in sorted(st.items())}},
                      indent=1)


def prompt_A(sc, rep, covered, integrate=False, filt=False):
    sp = spec(covered, integrate, filt)
    if rep == "H":
        hist = "\n".join(f"Observation: {s['obs']}\nReasoning & Action: {s['reason']}" for s in sc["steps"])
        return (f"Instructions:\n{sp}\nHistory:\n{hist}\n\nLatest Observation: {sc['corr']}\n"
                "Generate your next reasoning and action (format 'Action: <cmd>').")
    return (f"Instructions:\n{sp}\nYou keep an explicit state instead of a transcript. After every event, output a "
            "state patch (keys you change; null empties a shelf) and one action.\n\nCurrent state:\n"
            f"{_state_json(sc['state_before'])}\n\nLatest Observation: {sc['corr']}\n"
            'Reply with one short line of reasoning, then a JSON object {"state_patch": {...}, "action": "<cmd>"}.')


def apply_patch(st: dict, text: str) -> dict:
    st = dict(st)
    m = re.search(r"\{.*\}", text, re.S)
    try:
        obj = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        obj = {}
    patch = (obj.get("state_patch") or {}).get("shelf_contents", {}) if isinstance(obj, dict) else {}
    for k, v in (patch or {}).items():
        try:
            i = int(k)
        except ValueError:
            continue
        if v in (None, "", "empty", "EMPTY", {}):
            st.pop(i, None)
        elif isinstance(v, dict):
            st[i] = (v.get("sku"), v.get("units"), v.get("lot"))
    return st


def prompt_B(sc, rep, covered, a_text, integrate=False, filt=False):
    sp = spec(covered, integrate, filt)
    if rep == "H":
        hist = "\n".join(f"Observation: {s['obs']}\nReasoning & Action: {s['reason']}" for s in sc["steps"])
        hist += f"\nObservation: {sc['corr']}\nReasoning & Action: {a_text.strip()}"
        return (f"Instructions:\n{sp}\nHistory:\n{hist}\n\nLatest Observation: {sc['dec']}\n"
                "Generate your next reasoning and action (format 'Action: <cmd>').")
    st = apply_patch(sc["state_before"], a_text)
    return (f"Instructions:\n{sp}\nYou keep an explicit state instead of a transcript. After every event, output a "
            "state patch (keys you change; null empties a shelf) and one action.\n\nCurrent state:\n"
            f"{_state_json(st)}\n\nLatest Observation: {sc['dec']}\n"
            'Reply with one short line of reasoning, then a JSON object {"state_patch": {...}, "action": "<cmd>"}.')


def store_shelf(text: str):
    # the state arm writes the action inside a JSON string, so its quotes arrive escaped
    m = re.findall(r'Store\(\{\s*\\?"shelf\\?"\s*:\s*(\d+)', text)
    return int(m[-1]) if m else None

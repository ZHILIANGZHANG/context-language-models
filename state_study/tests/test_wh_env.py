"""Rule-reader tests for the closed-loop warehouse environment (state_study/validation/wh_env.py).

The readers below see only the text the CLI prints, as a subagent would. An attentive reader that
honors the facility notice gets every step right in every condition; a deaf reader that ignores
facility notices stores on the quarantined shelf at the decision step. Readers that keep no memory
of the notice also succeed when the text re-delivers it (REM line, pinned note).
"""
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validation"))

import wh_env as W  # noqa: E402


def fields(text):
    """Event type and key=value fields of the event in a CLI output (the line starting with [step t])."""
    m = re.search(r"^\[step (\d+)\] EVENT (\w+) \| (.*)$", text, re.M)
    kv = dict(f.split("=", 1) for f in m.group(3).split(" | "))
    return int(m.group(1)), m.group(2), kv


def play(root, ep, seed, cond, policy):
    """Play an episode through the CLI functions. policy(text, mem) -> (action, pins); mem is the
    reader's own memory: shelves it filled, plus whatever the policy stores."""
    W.cmd_new(root, ep, seed, cond)
    text, mem = W.cmd_start(root, ep), {"shelves": {}, "q": None}
    while "EPISODE COMPLETE" not in text:
        action, pins = policy(text, mem)
        text = W.cmd_act(root, ep, action, pins)
        if not text.startswith("ACTION REJECTED"):
            mem["last"] and mem["last"]()
    return W.score_state(W._load(root, ep))


def reader(source):
    """source: where the reader learns the quarantine. 'event' remembers the notice when it arrives,
    'deaf' never does, 'rem' reads only the STANDING NOTICE line, 'pin' pins the notice and reads only
    the PINNED NOTES block."""
    def policy(text, mem):
        t, kind, kv = fields(text)
        q = None
        if source == "event":
            if kind == "facility_notice" and kv["do_not_store"] == "true":
                mem["q"] = int(kv["shelf"])
            q = mem["q"]
        elif source == "rem" and (m := re.search(r"^STANDING NOTICE \| shelf=(\d+)", text, re.M)):
            q = int(m.group(1))
        elif source == "pin" and (m := re.search(r"^- shelf (\d+) quarantined: do not store", text, re.M)):
            q = int(m.group(1))
        pins = [f"shelf {kv['shelf']} quarantined: do not store"] if source == "pin" and kind == "facility_notice" else []
        sh = mem["shelves"]
        mem["last"] = None
        if kind == "inbound_pallet":
            s = next(i for i in range(W.N_SHELVES) if i not in sh and i != q)
            mem["last"] = lambda: sh.__setitem__(s, kv["sku"])
            return (f'Store({{"shelf": {s}, "sku": "{kv["sku"]}", "units": {kv["units"]}, "lot": "{kv["lot"]}"}})', pins)
        if kind == "outbound_order":
            held = sorted(i for i, v in sh.items() if v == kv["sku"])
            if held:
                mem["last"] = lambda: sh.pop(held[0])
                return f'Ship({{"shelf": {held[0]}, "sku": "{kv["sku"]}"}})', pins
        return "Wait({})", pins
    return policy


@pytest.mark.parametrize("seed", range(50))
def test_script_invariants(seed):
    sc = W.generate(seed)
    ev, s = sc["events"], sc["quarantined"]
    assert ev[W.NOTICE]["kind"] == "notice" and ev[W.NOTICE]["shelf"] == s
    assert ev[W.DECISION]["kind"] == "inbound"
    shelves, dependent = [None] * W.N_SHELVES, []
    for t, e in enumerate(ev):
        a = W.expected(shelves, e, s if t > W.NOTICE else None)
        if e["kind"] == "inbound" and t > W.NOTICE and W._lowest_empty(shelves) == s:
            dependent.append(t)
        if t < W.DECISION:
            assert a["args"].get("shelf") != s or a["name"] == "Wait", (seed, t)  # S untouched before the decision
        assert W.check(shelves, e, a) is None
        W.apply(shelves, a)
    assert dependent[0] == W.DECISION


@pytest.mark.parametrize("cond", W.CONDS)
def test_attentive_reader_gets_everything_right(tmp_path, cond):
    for seed in range(10):
        r = play(tmp_path, f"e{seed}", seed, cond, reader("event"))
        assert r["complete"] and r["decision"] == "honor" and r["decision_step"] == W.DECISION, (cond, seed, r)
        assert r["step_acc"] == 1.0 and r["rejections"] == 0 and r["violations"] == 0


def test_deaf_reader_violates_at_the_decision(tmp_path):
    for seed in range(10):
        r = play(tmp_path, f"d{seed}", seed, "BASE", reader("deaf"))
        assert r["decision"] == "violate" and r["on_track"] and r["violations"] == 1, (seed, r)


def test_re_delivered_notice_is_enough_without_memory(tmp_path):
    for seed in range(10):
        assert play(tmp_path, f"r{seed}", seed, "REM", reader("rem"))["decision"] == "honor"
        r = play(tmp_path, f"p{seed}", seed, "PIN", reader("pin"))
        assert r["decision"] == "honor" and r["pinned_at_notice"] and r["n_pins"] == 1, (seed, r)
        # pins are only shown in PIN; elsewhere the same reader is deaf
        assert play(tmp_path, f"q{seed}", seed, "BASE", reader("pin"))["decision"] == "violate"


def test_procedure_text_per_condition():
    base, rule, pin, rem = (W.procedure(c) for c in W.CONDS)
    assert base == rem and "facility_notice" not in base and "--pin" not in base
    assert "facility_notice - a shelf restriction" in rule and "never Store on that shelf" in rule
    assert pin.startswith(base) and "--pin" in pin
    assert 'Store({"shelf": <int>' in base and "Wait({})" in base


def test_rejection_retries_the_same_event(tmp_path):
    seed = 0
    ev = W.generate(seed)["events"]
    t0 = next(t for t, e in enumerate(ev) if e["kind"] == "inbound")
    W.cmd_new(tmp_path, "x", seed, "BASE")
    W.cmd_start(tmp_path, "x")
    for _ in range(t0):                                  # nothing is in stock before the first inbound
        W.cmd_act(tmp_path, "x", "Wait({})")
    st = W._load(tmp_path, "x")
    assert st["t"] == t0
    e = ev[t0]
    good = f'Store({{"shelf": 0, "sku": "{e["sku"]}", "units": {e["units"]}, "lot": "{e["lot"]}"}})'
    assert W.cmd_act(tmp_path, "x", good).startswith("ACCEPTED")
    t1 = next(t for t, x in enumerate(ev) if x["kind"] == "inbound" and t > t0)
    while W._load(tmp_path, "x")["t"] < t1:
        W.cmd_act(tmp_path, "x", "Wait({})")
    e = ev[t1]
    clash = f'Store({{"shelf": 0, "sku": "{e["sku"]}", "units": {e["units"]}, "lot": "{e["lot"]}"}})'
    out = W.cmd_act(tmp_path, "x", clash)
    assert out.startswith("ACTION REJECTED | shelf 0 is occupied") and f"[step {t1}]" in out
    out = W.cmd_act(tmp_path, "x", "Store(nonsense)")
    assert out.startswith("ACTION REJECTED | the action could not be parsed") and f"[step {t1}]" in out
    out = W.cmd_act(tmp_path, "x", clash)                # third failed attempt: the event is skipped
    assert "the event was skipped" in out and f"[step {t1 + 1}]" in out
    st = W._load(tmp_path, "x")
    assert [r["attempt"] for r in st["log"] if r["t"] == t1] == [0, 1, 2]
    assert W.world(st)[0] == ev[t0]["sku"]


def test_parse_action():
    assert W.parse_action('Store({"shelf": 3, "sku": "SKU-A", "units": 2, "lot": "L-1"})')["args"]["shelf"] == 3
    assert W.parse_action("  Wait({})  ") == {"name": "Wait", "args": {}}
    assert W.parse_action('I will ship. Ship({"shelf": 1, "sku": "SKU-B"})')["name"] == "Ship"
    assert W.parse_action("Store(shelf=3)") is None and W.parse_action("Wait([])") is None

"""Checks that the warehouse probe is deterministic, dependent, and solvable from its text."""

from __future__ import annotations

import json
import re

import pytest

from state_study.probes import CONDITIONS, KINDS, applicable, generate, render

SEEDS = range(12)
GAPS = (5, 20, 40, 80)


def _replay(sc):
    """Independent re-simulation from the structured events; returns the correct shelf."""
    shelves, quarantined = {}, set()
    for e in sc.events[:-1]:
        if e.kind == "arrival":
            free = next(i for i in range(1, sc.n_shelves + 1)
                        if shelves.get(i) is None and i not in quarantined)
            assert e.shelf == free, f"oracle stored at S{e.shelf}, rule says S{free}"
            shelves[e.shelf] = e.item
        elif e.kind in ("dispatch", "correction", "scan"):
            shelves[e.shelf] = None
        elif e.kind == "quarantine":
            quarantined.add(e.shelf)
    return next(i for i in range(1, sc.n_shelves + 1)
                if shelves.get(i) is None and i not in quarantined)


def _solve_from_text(prompt: str) -> int:
    """A rule-following reader that sees only the rendered prompt (no structured fields)."""
    m = re.search(r"CURRENT STATE:\n(\{.*?\n\})", prompt, re.S)
    if m and "HISTORY LOG" not in prompt:
        st = json.loads(m.group(1))
        q = set(st.get("quarantined", []))
        return next(int(k[1:]) for k, v in st["shelves"].items() if v == "EMPTY" and k not in q)
    occupied, quarantined, void_steps = set(), set(), set()
    body = prompt.split("=== CONTEXT ===")[1].split("=== END CONTEXT ===")[0]
    for line in body.splitlines():
        if (t := re.search(r"- step (\d+) .* is VOID", line)):
            void_steps.add(int(t.group(1)))
    step = None
    for line in body.splitlines():
        if (s := re.match(r"\[step (\d+)\]", line)):
            step = int(s.group(1))
        if (a := re.search(r"ACTION: STORE \S+ -> S(\d+)", line)) and step not in void_steps:
            occupied.add(int(a.group(1)))
        if re.search(r"\[VOID: superseded", line):
            pass  # the correction / scan line itself carries the state change
        if (d := re.search(r"S(\d+) is now EMPTY", line)):
            occupied.discard(int(d.group(1)))
        if (q := re.search(r"shelf S(\d+) QUARANTINED", line)):
            quarantined.add(int(q.group(1)))
        if (c := re.search(r"never completed;.*S(\d+) is EMPTY", line)):
            occupied.discard(int(c.group(1)))
        if line.startswith("[step") and "Inventory scan" in line:
            for shelf, status in re.findall(r"S(\d+): (EMPTY|OCCUPIED)", line):
                (occupied.discard if status == "EMPTY" else occupied.add)(int(shelf))
    return next(i for i in range(1, 200) if i not in occupied and i not in quarantined)


@pytest.mark.parametrize("kind", KINDS)
def test_deterministic(kind):
    a, b = generate(kind, seed=3), generate(kind, seed=3)
    assert [e.text for e in a.events] == [e.text for e in b.events]
    assert (a.answer, a.deaf_answer) == (b.answer, b.deaf_answer)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("gap", GAPS)
def test_dependent_and_consistent(kind, gap):
    for seed in SEEDS:
        sc = generate(kind, seed=seed, gap=gap)
        assert sc.answer != sc.deaf_answer
        assert _replay(sc) == sc.answer
        key_idx = next(i for i, e in enumerate(sc.events) if e.step == sc.key_step)
        decision_idx = len(sc.events) - 1
        if kind in ("explicit", "implicit"):
            # The decision is the first arrival after the key event, exactly `gap` events later.
            assert decision_idx - key_idx - 1 == gap
            assert not any(e.kind == "arrival" for e in sc.events[key_idx + 1:decision_idx])
            voided = [e for e in sc.events if e.superseded_by == sc.key_step]
            assert len(voided) == 1 and voided[0].shelf == sc.answer
        else:
            assert sc.deaf_answer in sc.quarantined
            assert sc.shelves[sc.deaf_answer] is None


@pytest.mark.parametrize("kind", KINDS)
def test_text_is_sufficient(kind):
    """Every applicable condition except state_noq contains enough text to answer correctly."""
    for seed in SEEDS:
        sc = generate(kind, seed=seed, gap=40)
        for cond in CONDITIONS:
            if not applicable(sc, cond):
                continue
            got = _solve_from_text(render(sc, cond))
            if cond == "state_noq":
                assert got == sc.deaf_answer  # the schema cannot represent the quarantine
            else:
                assert got == sc.answer, (kind, seed, cond)


def test_condition_shapes():
    sc = generate("explicit", seed=1, gap=20)
    voided = next(e for e in sc.events if e.superseded_by is not None)
    raw, tail = render(sc, "raw"), render(sc, "tomb_tail")
    assert voided.action in raw
    assert voided.action not in render(sc, "delete")
    assert "[VOID: superseded" in render(sc, "tomb_inline")
    assert tail.startswith(raw.split("=== END CONTEXT ===")[0])  # tail tombstone is append-only
    state = render(sc, "state")
    assert "[step" not in state.split("=== END CONTEXT ===")[0]
    assert '"quarantined"' not in render(generate("delayed", seed=1), "state_noq")
    assert '"quarantined"' in render(generate("delayed", seed=1), "state")
    for cond in CONDITIONS:
        assert f'"shelf": "S{sc.answer}"' not in render(sc, cond)

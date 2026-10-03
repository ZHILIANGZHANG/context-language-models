"""Context conditions for the warehouse probe (proposal I3).

Every condition renders the *same* scenario prefix; only the representation changes.

==========  ==============================================================  ===================
condition   what the model sees                                             prefix-cache safe?
==========  ==============================================================  ===================
raw         full chronological log                                          yes (append-only)
tomb_tail   raw log + a tombstone list appended at the end                  yes (append-only)
tomb_inline raw log with superseded entries marked VOID in place            no  (edits the past)
delete      raw log with superseded entries removed                         no  (edits the past)
state       current shelf state as JSON + the decision event, no history    n/a (rewritten)
state_noq   like ``state`` but the schema has no quarantine field           n/a (rewritten)
state_log   raw log + current state JSON appended at the end                no  (tail rewritten)
==========  ==============================================================  ===================

``tomb_*`` and ``delete`` only differ from ``raw`` for scenarios with superseded entries
(``explicit`` / ``implicit``); ``state_noq`` only differs from ``state`` for ``delayed``.
"""

from __future__ import annotations

import json

from .warehouse import RULE, Event, Scenario

CONDITIONS = ("raw", "tomb_tail", "tomb_inline", "delete", "state", "state_noq", "state_log")


def _line(e: Event, *, void_mark: bool = False) -> str:
    s = f"[step {e.step}] {e.text}"
    if e.action:
        s += f"\n    ACTION: {e.action}\n    RESULT: {e.result}"
    if void_mark and e.superseded_by is not None:
        s += f"\n    [VOID: superseded by step {e.superseded_by}]"
    return s


def _log(sc: Scenario, *, void_mark: bool = False, drop_superseded: bool = False) -> str:
    history = sc.events[:-1]
    lines = [_line(e, void_mark=void_mark) for e in history
             if not (drop_superseded and e.superseded_by is not None)]
    return "\n".join(lines)


def _state_json(sc: Scenario, *, with_quarantine: bool = True) -> str:
    state: dict = {"shelves": {f"S{i}": (v if v is not None else "EMPTY")
                               for i, v in sorted(sc.shelves.items())}}
    if with_quarantine:
        state["quarantined"] = [f"S{i}" for i in sc.quarantined]
    return json.dumps(state, indent=1)


def _tombstones(sc: Scenario) -> str:
    rows = [f"- step {e.step} ({e.action}) is VOID, superseded by step {e.superseded_by}."
            for e in sc.events if e.superseded_by is not None]
    return "SUPERSEDED ENTRIES (do not rely on them):\n" + "\n".join(rows) if rows else ""


def render(sc: Scenario, condition: str) -> str:
    """Return the full prompt for ``condition``; the answer is never included."""

    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")
    if condition == "raw":
        body = "HISTORY LOG:\n" + _log(sc)
    elif condition == "tomb_tail":
        tail = _tombstones(sc)
        body = "HISTORY LOG:\n" + _log(sc) + ("\n\n" + tail if tail else "")
    elif condition == "tomb_inline":
        body = "HISTORY LOG:\n" + _log(sc, void_mark=True)
    elif condition == "delete":
        body = "HISTORY LOG:\n" + _log(sc, drop_superseded=True)
    elif condition == "state":
        body = "CURRENT STATE:\n" + _state_json(sc)
    elif condition == "state_noq":
        body = "CURRENT STATE:\n" + _state_json(sc, with_quarantine=False)
    else:  # state_log
        body = "HISTORY LOG:\n" + _log(sc) + "\n\nCURRENT STATE:\n" + _state_json(sc)

    decision = sc.events[-1]
    return (
        f"{RULE}\n\nYou are the warehouse agent. Your working context follows.\n\n"
        f"=== CONTEXT ===\n{body}\n=== END CONTEXT ===\n\n"
        f"[step {decision.step}] {decision.text}\n"
        f"Which shelf do you store item {sc.decision_item} on?\n"
        'Reply with JSON only, exactly of the form {"shelf": "S<number>"}.'
    )


def applicable(sc: Scenario, condition: str) -> bool:
    """Whether ``condition`` is a distinct treatment for this scenario kind."""

    if condition in ("tomb_tail", "tomb_inline", "delete"):
        return sc.kind in ("explicit", "implicit")
    if condition == "state_noq":
        return sc.kind == "delayed"
    return True

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
state_stale the state an agent holds if it missed the key event (I4 base)   n/a (rewritten)
recall_cand state_stale + facts recalled from the log for the intended shelf n/a
recall_alt  recall_cand + facts for every lower shelf the state calls full  n/a
self_schema raw log; the model first writes a state in a structure it        yes (append-only)
            designs itself, then decides (I1 proxy)
==========  ==============================================================  ===================

The ``recall_*`` conditions implement proposal I4 (bind at act time): propose a shelf from the
stale state, retrieve every log event about that shelf (``recall_cand``) or about it and all
lower-numbered shelves the state calls occupied (``recall_alt``), distil them into one status line
per shelf, and decide again. In this probe the proposal is the stale-state answer
(``deaf_answer``); in the full method it is the model's own first-pass action.

``tomb_*`` and ``delete`` only differ from ``raw`` for scenarios with superseded entries
(``explicit`` / ``implicit``); ``state_noq`` only differs from ``state`` for ``delayed``.
"""

from __future__ import annotations

import json

from .warehouse import RULE, Event, Scenario

CONDITIONS = ("raw", "tomb_tail", "tomb_inline", "delete", "state", "state_noq", "state_log",
              "state_stale", "recall_cand", "recall_alt", "self_schema")


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


def _stale_shelves(sc: Scenario) -> tuple[dict[int, str | None], list[int]]:
    """Shelf contents and quarantine list as an agent that missed the key event holds them."""
    shelves, quarantined = dict(sc.shelves), list(sc.quarantined)
    if sc.kind in ("explicit", "implicit"):
        voided = next(e for e in sc.events if e.superseded_by == sc.key_step)
        shelves[voided.shelf] = voided.item
    else:  # delayed: the stale schema has no quarantine field
        quarantined = []
    return shelves, quarantined


def _stale_state_json(sc: Scenario) -> str:
    shelves, _ = _stale_shelves(sc)
    return json.dumps({"shelves": {f"S{i}": (v if v is not None else "EMPTY")
                                   for i, v in sorted(shelves.items())}}, indent=1)


def recall(sc: Scenario, shelves: list[int]) -> str:
    """Distil every log event about ``shelves`` into one status line per shelf."""
    lines = []
    for k in shelves:
        status, why, quarantine = "EMPTY", "never used", None
        for e in sc.events[:-1]:
            if e.shelf != k:
                continue
            if e.kind == "arrival" and e.superseded_by is None:
                status, why = "OCCUPIED", f"{e.item} stored at step {e.step}"
            elif e.kind == "arrival":
                status, why = "OCCUPIED", f"{e.item} recorded at step {e.step}"
            elif e.kind == "dispatch":
                status, why = "EMPTY", f"{e.item} dispatched at step {e.step}"
            elif e.kind == "correction":
                status, why = "EMPTY", f"audit correction at step {e.step} voided the earlier storage"
            elif e.kind == "scan":
                status, why = "EMPTY", f"inventory scan at step {e.step}"
            elif e.kind == "quarantine":
                quarantine = f"quarantine notice at step {e.step}, do_not_store, no release since"
        q = f"quarantined=YES ({quarantine})" if quarantine else "quarantined=NO"
        lines.append(f"- S{k}: status={status} ({why}); {q}")
    return "\n".join(lines)


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
    elif condition == "state_log":
        body = "HISTORY LOG:\n" + _log(sc) + "\n\nCURRENT STATE:\n" + _state_json(sc)
    elif condition == "state_stale":
        body = "CURRENT STATE:\n" + _stale_state_json(sc)
    elif condition in ("recall_cand", "recall_alt"):
        cand = sc.deaf_answer
        query = [cand]
        if condition == "recall_alt":
            stale, _ = _stale_shelves(sc)
            query = [i for i in range(1, cand) if stale[i] is not None] + [cand]
        body = ("CURRENT STATE:\n" + _stale_state_json(sc) +
                f"\n\nYou intended to store the item on S{cand}. Before acting, these facts were "
                f"recalled from the full event log:\n" + recall(sc, query))
    else:  # self_schema
        body = "HISTORY LOG:\n" + _log(sc)

    decision = sc.events[-1]
    if condition == "self_schema":
        ask = ("First, design a compact state representation for this warehouse -- choose whatever "
               "fields you think matter -- and fill it in from the log inside <state>...</state>. "
               "Then decide. End your reply with one line of JSON exactly of the form "
               '{"shelf": "S<number>"}.')
    else:
        ask = 'Reply with JSON only, exactly of the form {"shelf": "S<number>"}.'
    return (
        f"{RULE}\n\nYou are the warehouse agent. Your working context follows.\n\n"
        f"=== CONTEXT ===\n{body}\n=== END CONTEXT ===\n\n"
        f"[step {decision.step}] {decision.text}\n"
        f"Which shelf do you store item {sc.decision_item} on?\n{ask}"
    )


def applicable(sc: Scenario, condition: str) -> bool:
    """Whether ``condition`` is a distinct treatment for this scenario kind."""

    if condition in ("tomb_tail", "tomb_inline", "delete"):
        return sc.kind in ("explicit", "implicit")
    if condition == "state_noq":
        return sc.kind == "delayed"
    return True

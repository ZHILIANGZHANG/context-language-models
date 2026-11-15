"""T1: multi-turn warehouse environment (docs/experiment_plan.md §3, docs/data_and_training.md §1.1).

The event stream reuses the world simulator of the warehouse probe (``state_study.probes.warehouse``).
An episode has several decision points; each one depends on a key event earlier in the stream
(an audit correction, an inventory scan, or a quarantine notice that becomes relevant later).

The agent pulls pages with ``next_events``; when a page ends with a decision it must call
``store(shelf)``. The environment then applies the rule-correct storage regardless of the answer,
so the world (and therefore every page and every "needed" set) is fixed by the seed. That is what
lets the oracle and the minimal sufficient sets be computed exactly.

``rule_read`` is the rule reader: it sees only page text. ``need[d]`` is a minimal set of pages with
which the rule reader still gets decision ``d`` right.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

from state_study.probes.warehouse import RULE, Event, _World

SYSTEM = ("You are a warehouse agent with a limited working memory. " + RULE + " "
          "Read the event log page by page with next_events. When a page ends with DECISION REQUIRED, "
          "decide the shelf and call store. Your memory is paged: every page has a header line "
          "[[PAGE n ...]]. Use kv_out to move pages you do not need out of memory, kv_in to bring them "
          "back, and note to keep a short summary. The [KV] line shows what is resident.")

TASK = "Process the warehouse log until it ends. Call next_events to get the first page."


@dataclass
class T1Config:
    n_decisions: int = 4
    warmup: int = 20
    gap: tuple[int, int] = (5, 25)
    events_per_page: tuple[int, int] = (4, 10)
    kinds: tuple[str, ...] = ("explicit", "implicit", "delayed")
    p_arrival: float = 0.3
    p_dispatch: float = 0.15
    n_shelves: int = 60


@dataclass
class Decision:
    page: int  # index of the page that asks for the decision
    item: str
    answer: int
    kind: str
    key_step: int


@dataclass
class Episode:
    seed: int
    pages: list[str]
    decisions: list[Decision]
    need: dict[int, set[int]] = field(default_factory=dict)  # decision page -> minimal page set

    def decision_at(self, page: int) -> Decision | None:
        return next((d for d in self.decisions if d.page == page), None)

    def next_use(self, page: int, after: int) -> int | None:
        """Index of the first decision page > ``after`` whose need set contains ``page``."""
        uses = [d.page for d in self.decisions if d.page > after and page in self.need[d.page]]
        return min(uses) if uses else None


# -- rendering -------------------------------------------------------------------------------
def _line(e: Event) -> str:
    s = f"[step {e.step}] {e.text}"
    if e.kind == "decision":
        return s + f"\n    DECISION REQUIRED: choose the shelf for item {e.item} and call store."
    if e.action:
        s += f"\n    ACTION: {e.action}\n    RESULT: {e.result}"
    return s


def _resolved(e: Event) -> str:
    return f"[step {e.step}] RESOLVED decision for {e.item}\n    ACTION: {e.action}\n    RESULT: {e.result}"


# -- rule reader -----------------------------------------------------------------------------
def read_state(texts: list[str]) -> tuple[set[int], set[int]]:
    """(occupied, quarantined) shelves, replaying the page text in order with the warehouse rules."""
    occupied, quarantined = set(), set()
    for text in texts:
        for line in text.splitlines():
            if (a := re.search(r"ACTION: STORE \S+ -> S(\d+)", line)):
                occupied.add(int(a.group(1)))
            if (d := re.search(r"S(\d+) is now EMPTY", line)):
                occupied.discard(int(d.group(1)))
            if (q := re.search(r"shelf S(\d+) QUARANTINED", line)):
                quarantined.add(int(q.group(1)))
            if (c := re.search(r"never completed;.*S(\d+) is EMPTY", line)):
                occupied.discard(int(c.group(1)))
            if "Inventory scan" in line:
                for shelf, status in re.findall(r"S(\d+): (EMPTY|OCCUPIED)", line):
                    (occupied.discard if status == "EMPTY" else occupied.add)(int(shelf))
            if (n := re.search(r"NOTE-STATE occupied=\[([\d,]*)\] quarantined=\[([\d,]*)\]", line)):
                occupied = {int(x) for x in n.group(1).split(",") if x}
                quarantined = {int(x) for x in n.group(2).split(",") if x}
    return occupied, quarantined


def rule_read(texts: list[str]) -> int:
    """Lowest shelf that is EMPTY and not quarantined, from page text only."""
    occupied, quarantined = read_state(texts)
    return next(i for i in range(1, 1000) if i not in occupied and i not in quarantined)


def state_note(texts: list[str]) -> str:
    """A note that captures the full state the rule reader needs, in a form rule_read accepts."""
    occupied, quarantined = read_state(texts)
    occ = ",".join(str(i) for i in sorted(occupied))
    qua = ",".join(str(i) for i in sorted(quarantined))
    return f"NOTE-STATE occupied=[{occ}] quarantined=[{qua}]"


def derivation(texts: list[str]) -> str:
    """Explicit, checkable reasoning for a decision: the status of every shelf up to the answer."""
    occupied, quarantined = read_state(texts)
    a = rule_read(texts)
    parts = []
    for i in range(1, a):
        parts.append(f"S{i} " + ("quarantined" if i in quarantined else "occupied"))
    parts.append(f"S{a} empty and not quarantined")
    return "Shelf check from the visible records: " + ", ".join(parts) + f". Lowest free shelf: S{a}."


def page_role(text: str, current_answer: int | None) -> str:
    """Why a page can go, judged from its own text (used for eviction prose)."""
    lines = [ln for ln in text.splitlines() if ln.startswith("[step")]
    if lines and all("Telemetry:" in ln for ln in lines):
        return "telemetry only"
    if "QUARANTINED" in text:
        return "quarantine notice"
    shelves = [int(x) for x in re.findall(r"S(\d+)", text)]
    if current_answer is not None and shelves and min(shelves) > current_answer:
        return f"only shelves above S{current_answer}"
    return "not needed for the current shelf state"


def minimal_sufficient(texts: list[str], answer: int, must: set[int]) -> set[int]:
    """Greedy minimal set of page indices with which rule_read still returns ``answer``.
    ``must`` (the decision page) is never removed. Larger pages are tried first."""
    keep = set(range(len(texts)))
    if rule_read(texts) != answer:
        raise AssertionError("rule reader fails on the full history; the episode is broken")
    changed = True
    while changed:  # repeat until no single page can be removed (the rule is not monotone:
        changed = False  # a dispatch line becomes redundant once its store line is gone)
        for p in sorted(keep - must, key=lambda i: -len(texts[i])):
            trial = keep - {p}
            if rule_read([texts[i] for i in sorted(trial)]) == answer:
                keep, changed = trial, True
    return keep


# -- generation ------------------------------------------------------------------------------
def generate(seed: int, cfg: T1Config | None = None) -> Episode:
    cfg = cfg or T1Config()
    rng = random.Random(f"t1-{seed}")
    w = _World(cfg.n_shelves, rng)
    w.filler(cfg.warmup, arrivals=True, dispatch_above=0, p_arrival=0.35, p_dispatch=0.12)
    decision_events: list[tuple[Event, str, int]] = []
    for _ in range(cfg.n_decisions):
        while w.lowest_free() <= 8:
            w.arrival()
            if rng.random() < 0.5:
                w.telemetry()
        kind = rng.choice(cfg.kinds)
        cands = [s for s in range(3, 9) if w.shelves[s] is not None and s not in w.quarantined]
        if not cands:
            break
        target = rng.choice(cands)
        gap = rng.randint(*cfg.gap)
        if kind in ("explicit", "implicit"):
            w.dispatch(target)
            stored = w.arrival()
            if stored.shelf != target:
                continue
            w.filler(5, arrivals=True, dispatch_above=target, p_arrival=cfg.p_arrival, p_dispatch=cfg.p_dispatch)
            key = w.correction(stored) if kind == "explicit" else w.scan(stored)
            w.filler(gap, arrivals=False, dispatch_above=target, p_arrival=cfg.p_arrival, p_dispatch=cfg.p_dispatch)
            deaf = w.lowest_free(treat_occupied=(target,))
        else:
            key = w.quarantine(target)
            w.filler(gap, arrivals=True, dispatch_above=target, keep=(target,),
                     p_arrival=cfg.p_arrival, p_dispatch=cfg.p_dispatch)
            w.dispatch(target)
            deaf = w.lowest_free(ignore_quarantine=True)
        answer = w.lowest_free()
        if answer == deaf:
            continue
        item = w._new_item()
        ev = Event(w.step, "decision",
                   f"Inbound: item {item} arrived at dock A1 (supplier=Helix Pharma, carrier=DHL, "
                   f"weight={rng.randint(2, 90)}kg).", action=f"STORE {item} -> S{answer}",
                   result=f"OK: {item} stored on S{answer}.", shelf=answer, item=item)
        w.events.append(ev)
        w.shelves[answer] = item
        decision_events.append((ev, kind, key.step))
        w.filler(rng.randint(1, 4), arrivals=False, dispatch_above=8)

    # paginate: a decision always ends a page; its resolution opens the next page
    pages: list[str] = []
    decisions: list[Decision] = []
    cur: list[str] = []
    limit = rng.randint(*cfg.events_per_page)
    pending: Event | None = None
    dmeta = {id(e): (k, s) for e, k, s in decision_events}
    for e in w.events:
        if pending is not None:
            cur.append(_resolved(pending))
            pending = None
        cur.append(_line(e))
        if e.kind == "decision":
            pages.append("\n".join(cur))
            kind, key_step = dmeta[id(e)]
            decisions.append(Decision(len(pages) - 1, e.item, e.shelf, kind, key_step))
            cur, pending, limit = [], e, rng.randint(*cfg.events_per_page)
        elif len(cur) >= limit:
            pages.append("\n".join(cur))
            cur, limit = [], rng.randint(*cfg.events_per_page)
    if pending is not None:
        cur.append(_resolved(pending))
    if cur:
        pages.append("\n".join(cur))

    ep = Episode(seed, pages, decisions)
    for d in decisions:
        ep.need[d.page] = minimal_sufficient(pages[: d.page + 1], d.answer, {d.page})
    return ep


# -- environment -----------------------------------------------------------------------------
def _fn(name, desc, props, required):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required}}}


TOOLS = [
    _fn("next_events", "Return the next page of the event log.", {}, []),
    _fn("store", "Store the item of the pending decision on a shelf.",
        {"shelf": {"type": "integer", "description": "Shelf number, e.g. 7 for S7."}}, ["shelf"]),
]


class WarehouseEnv:
    """Environment protocol used by ``self_paging.agent.episode``:

    ``system``/``task``/``tools`` describe the episode; ``step(name, args)`` returns
    ``(text, meta, done)`` where ``meta["env_page"]`` marks a log page; ``score()`` is the reward.
    """

    def __init__(self, seed: int, cfg: T1Config | None = None) -> None:
        self.ep = generate(seed, cfg)
        self.system, self.task, self.tools = SYSTEM, TASK, TOOLS
        self.cursor = 0  # next page to deliver
        self.pending: Decision | None = None
        self.answers: dict[int, int | None] = {d.page: None for d in self.ep.decisions}

    @property
    def done(self) -> bool:
        return self.cursor >= len(self.ep.pages) and self.pending is None

    def step(self, name: str, args: dict) -> tuple[str, dict, bool]:
        if name == "next_events":
            if self.pending is not None:
                return (f"ERROR: a decision is pending for item {self.pending.item}; call store first.",
                        {}, False)
            if self.cursor >= len(self.ep.pages):
                return "END OF LOG.", {}, True
            idx = self.cursor
            self.cursor += 1
            self.pending = self.ep.decision_at(idx)
            return self.ep.pages[idx], {"env_page": idx}, False
        if name == "store":
            if self.pending is None:
                return "ERROR: no decision is pending.", {}, False
            try:
                shelf = int(args.get("shelf"))
            except (TypeError, ValueError):
                return "ERROR: shelf must be an integer.", {}, False
            self.answers[self.pending.page] = shelf
            self.pending = None
            return "Recorded.", {}, self.done
        return f"ERROR: unknown tool {name}.", {}, False

    def score(self) -> float:
        if not self.answers:
            return 0.0
        right = sum(self.answers[d.page] == d.answer for d in self.ep.decisions)
        return right / len(self.ep.decisions)

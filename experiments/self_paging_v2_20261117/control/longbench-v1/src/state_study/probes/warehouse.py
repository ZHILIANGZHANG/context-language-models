"""Warehouse probe: a deterministic environment for decisions that depend on one earlier fact.

The agent follows one rule -- store each arriving item on the lowest-numbered shelf that is
empty and not quarantined -- so the correct action at any step is a pure function of the true
world state. A scenario is a prefix of events produced by an oracle agent (every earlier action
is correct) that ends at a *decision step*: an arrival whose correct shelf differs from the shelf
an agent would pick if it ignored one key event.

Three kinds of key event:

``explicit``  An audit correction says an earlier storage "never completed"; the shelf is empty.
              The earlier storage entry is superseded.
``implicit``  The same retroactive invalidation, but delivered as an inventory scan that lists
              the shelf as EMPTY without referring to the earlier entry.
``delayed``   A quarantine notice for an occupied shelf; ``gap`` events later that shelf is emptied
              and becomes the lowest free one, so the agent must skip it.

The environment is reimplemented from the warehouse task described in SKILL.state (arXiv
2608.26263, Sec. 4.1); no code from other replications is used.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

RULE = (
    "Warehouse rule: when an item arrives, store it on the lowest-numbered shelf that is "
    "currently EMPTY and NOT quarantined. Quarantine notices stay in force until a release "
    "notice. Audit corrections and inventory scans override earlier records."
)

KINDS = ("explicit", "implicit", "delayed")

_SUPPLIERS = ["Arcadia Foods", "Borealis Parts", "Cobalt Medical", "Delta Textiles",
              "Equinox Labs", "Fjord Marine", "Granite Tools", "Helix Pharma"]
_CARRIERS = ["DHL", "UPS", "Maersk", "FedEx", "Kuehne+Nagel"]
_ZONES = ["A", "B", "C"]


@dataclass
class Event:
    """One log entry. ``action``/``result`` are set for arrivals the oracle agent handled."""

    step: int
    kind: str  # arrival | dispatch | quarantine | correction | scan | telemetry | decision
    text: str
    action: str | None = None
    result: str | None = None
    shelf: int | None = None
    item: str | None = None
    superseded_by: int | None = None  # step of the event that voids this entry


@dataclass
class Scenario:
    kind: str
    seed: int
    gap: int
    n_shelves: int
    events: list[Event]  # ends with the decision arrival (no action)
    decision_item: str
    answer: int  # correct shelf
    deaf_answer: int  # shelf chosen if the key event is ignored
    key_step: int
    shelves: dict[int, str | None]  # true contents at the decision step
    quarantined: list[int] = field(default_factory=list)


class _World:
    def __init__(self, n_shelves: int, rng: random.Random):
        self.n = n_shelves
        self.rng = rng
        self.shelves: dict[int, str | None] = {i: None for i in range(1, n_shelves + 1)}
        self.quarantined: set[int] = set()
        self.events: list[Event] = []
        self.item_counter = 0

    # ---- helpers -------------------------------------------------------------------------
    @property
    def step(self) -> int:
        return len(self.events) + 1

    def lowest_free(self, *, ignore_quarantine: bool = False, treat_occupied: tuple = ()) -> int:
        for i in range(1, self.n + 1):
            if self.shelves[i] is not None or i in treat_occupied:
                continue
            if not ignore_quarantine and i in self.quarantined:
                continue
            return i
        raise RuntimeError("warehouse full; increase n_shelves")

    def _new_item(self) -> str:
        self.item_counter += 1
        return f"I-{self.rng.randint(100, 999)}{self.item_counter:03d}"

    # ---- event writers -------------------------------------------------------------------
    def arrival(self) -> Event:
        item = self._new_item()
        r = self.rng
        text = (f"Inbound: item {item} arrived at dock {r.choice(_ZONES)}{r.randint(1, 4)} "
                f"(supplier={r.choice(_SUPPLIERS)}, carrier={r.choice(_CARRIERS)}, "
                f"weight={r.randint(2, 90)}kg, pallets={r.randint(1, 3)}, "
                f"po=PO-{r.randint(10000, 99999)}, temp_class={r.choice(['ambient', 'chilled'])}).")
        shelf = self.lowest_free()
        self.shelves[shelf] = item
        ev = Event(self.step, "arrival", text, action=f"STORE {item} -> S{shelf}",
                   result=f"OK: {item} stored on S{shelf}.", shelf=shelf, item=item)
        self.events.append(ev)
        return ev

    def dispatch(self, shelf: int) -> Event:
        item = self.shelves[shelf]
        assert item is not None
        r = self.rng
        text = (f"Outbound: order SO-{r.randint(10000, 99999)} picked item {item} from S{shelf} "
                f"(customer=C-{r.randint(100, 999)}, carrier={r.choice(_CARRIERS)}); S{shelf} is now EMPTY.")
        self.shelves[shelf] = None
        ev = Event(self.step, "dispatch", text, shelf=shelf, item=item)
        self.events.append(ev)
        return ev

    def telemetry(self) -> Event:
        r = self.rng
        text = (f"Telemetry: zone={r.choice(_ZONES)} temp={r.uniform(14, 24):.1f}C "
                f"humidity={r.randint(30, 60)}% forklift_F{r.randint(1, 6)}_battery={r.randint(20, 99)}% "
                f"dock_queue={r.randint(0, 7)} conveyor_speed={r.uniform(0.8, 1.6):.2f}m/s "
                f"badge_scans={r.randint(5, 80)} hvac_mode={r.choice(['eco', 'normal', 'boost'])}.")
        ev = Event(self.step, "telemetry", text)
        self.events.append(ev)
        return ev

    def quarantine(self, shelf: int) -> Event:
        r = self.rng
        text = (f"Facilities notice FN-{r.randint(1000, 9999)}: shelf S{shelf} QUARANTINED "
                f"(reason=sprinkler inspection, inspector=E-{r.randint(10, 99)}, ticket=T-{r.randint(1000, 9999)}, "
                f"do_not_store=true, existing stock may remain until picked, valid until a release notice).")
        self.quarantined.add(shelf)
        ev = Event(self.step, "quarantine", text, shelf=shelf)
        self.events.append(ev)
        return ev

    def correction(self, voided: Event) -> Event:
        r = self.rng
        text = (f"Audit correction AC-{r.randint(1000, 9999)}: the storage recorded at step {voided.step} "
                f"({voided.item} -> S{voided.shelf}) never completed; the pallet was returned to the vendor. "
                f"S{voided.shelf} is EMPTY.")
        self.shelves[voided.shelf] = None
        ev = Event(self.step, "correction", text, shelf=voided.shelf, item=voided.item)
        self.events.append(ev)
        voided.superseded_by = ev.step
        return ev

    def scan(self, voided: Event, extra: int = 4) -> Event:
        """Implicit invalidation: a scan reporting the shelf EMPTY, with no reference to the old entry."""
        shelf = voided.shelf
        self.shelves[shelf] = None
        others = [i for i in range(1, self.n + 1) if i != shelf]
        listed = sorted(self.rng.sample(others, k=min(extra, len(others))) + [shelf])
        parts = [f"S{i}: {'EMPTY' if self.shelves[i] is None else 'OCCUPIED'}" for i in listed]
        text = f"Inventory scan (zone sweep {self.rng.randint(100, 999)}): " + "; ".join(parts) + "."
        ev = Event(self.step, "scan", text, shelf=shelf)
        self.events.append(ev)
        voided.superseded_by = ev.step
        return ev

    # ---- background ----------------------------------------------------------------------
    def filler(self, n: int, *, arrivals: bool, dispatch_above: int, keep: tuple = (),
               p_arrival: float = 0.3, p_dispatch: float = 0.15) -> None:
        """Telemetry-heavy filler. Dispatches only touch shelves above ``dispatch_above``."""
        for _ in range(n):
            u = self.rng.random()
            occupied_high = [i for i, v in self.shelves.items()
                             if v is not None and i > dispatch_above and i not in keep]
            if arrivals and u < p_arrival:
                self.arrival()
            elif u < p_arrival + p_dispatch and occupied_high:
                self.dispatch(self.rng.choice(occupied_high))
            else:
                self.telemetry()


def generate(kind: str, seed: int, gap: int = 40, warmup: int = 60,
             n_shelves: int | None = None) -> Scenario:
    """Build one scenario whose decision step depends on the key event ``gap`` events earlier.

    ``n_shelves`` defaults to enough capacity for the scenario length (arrivals outpace
    dispatches during warm-up)."""

    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    if n_shelves is None:
        n_shelves = 20 + (warmup + gap) // 2
    rng = random.Random(f"{kind}-{seed}")
    w = _World(n_shelves, rng)

    # Warm-up: churn with arrivals and dispatches until shelves 1..8 are all occupied.
    w.filler(warmup, arrivals=True, dispatch_above=0, p_arrival=0.35, p_dispatch=0.12)
    while w.lowest_free() <= 8:
        w.arrival()
        if rng.random() < 0.5:
            w.telemetry()
    target = rng.randint(3, 6)

    if kind in ("explicit", "implicit"):
        w.dispatch(target)  # the target becomes the lowest free shelf ...
        stored = w.arrival()  # ... so the next item is stored there (the entry to be voided)
        assert stored.shelf == target
        w.filler(10, arrivals=True, dispatch_above=target)
        key = w.correction(stored) if kind == "explicit" else w.scan(stored)
        # No arrivals inside the gap: the first arrival after the key event is the decision.
        w.filler(gap, arrivals=False, dispatch_above=target)
        answer = w.lowest_free()
        deaf = w.lowest_free(treat_occupied=(target,))
    else:  # delayed relevance
        key = w.quarantine(target)  # target is occupied, so the notice is not yet relevant
        w.filler(gap, arrivals=True, dispatch_above=target, keep=(target,))
        w.dispatch(target)  # now the quarantined shelf is the lowest empty one
        answer = w.lowest_free()
        deaf = w.lowest_free(ignore_quarantine=True)

    assert answer != deaf, "scenario is not dependent on the key event"
    item = w._new_item()
    decision = Event(w.step, "decision",
                     f"Inbound: item {item} arrived at dock A1 (supplier={rng.choice(_SUPPLIERS)}, "
                     f"carrier={rng.choice(_CARRIERS)}, weight={rng.randint(2, 90)}kg).",
                     item=item)
    w.events.append(decision)
    return Scenario(kind=kind, seed=seed, gap=gap, n_shelves=n_shelves, events=w.events,
                    decision_item=item, answer=answer, deaf_answer=deaf, key_step=key.step,
                    shelves=dict(w.shelves), quarantined=sorted(w.quarantined))

"""Page table for self-paging: which pages of the append-only KV log are resident.

The KV log is append-only: every token's KV is computed once, when it is written. The model
changes only the *resident set* R (the pages its attention can read). Resident-set versions are
numbered; every token records the version it was processed under, which is all a trainer needs
to rebuild the exact attention mask in a single forward pass (see ``self_paging.hf.masks``).
"""

from __future__ import annotations

from dataclasses import dataclass

KINDS = ("system", "task", "tool", "assistant", "think", "note", "gauge")
WHERE = ("gpu", "host", "dropped")


class PagingError(ValueError):
    """An illegal paging action (unknown page, pinned page, wrong location, over budget)."""


@dataclass
class Page:
    pid: int
    kind: str
    start: int  # global log position of the page's first token
    length: int = 0
    pinned: bool = False
    where: str = "gpu"

    @property
    def end(self) -> int:
        return self.start + self.length


class PageTable:
    """Pages in log order, plus the history of resident-set versions.

    ``snapshots[v]`` is the resident set of version ``v``. A page opened during version ``v`` is
    added to ``snapshots[v]`` in place: tokens written earlier in that version cannot see it anyway
    (causality), so no new version is needed. ``page_out``/``page_in``/``drop`` start a new version.
    """

    def __init__(self) -> None:
        self.pages: dict[int, Page] = {}
        self.snapshots: list[set[int]] = [set()]
        self.current: int | None = None  # the page being written; always resident

    # -- queries ---------------------------------------------------------------------------
    @property
    def version(self) -> int:
        return len(self.snapshots) - 1

    def resident(self) -> set[int]:
        return set(self.snapshots[-1])

    def resident_tokens(self) -> int:
        return sum(self.pages[p].length for p in self.snapshots[-1])

    def in_order(self, pids) -> list[int]:
        return sorted(pids, key=lambda p: self.pages[p].start)

    # -- writing ---------------------------------------------------------------------------
    def open(self, kind: str, start: int, pinned: bool = False) -> int:
        if kind not in KINDS:
            raise PagingError(f"unknown page kind {kind!r}")
        pid = len(self.pages)
        self.pages[pid] = Page(pid, kind, start, pinned=pinned)
        self.snapshots[-1].add(pid)
        self.current = pid
        return pid

    def grow(self, pid: int, n: int) -> None:
        if pid != self.current:
            raise PagingError(f"page {pid} is not the page being written ({self.current})")
        self.pages[pid].length += n

    # -- actions ---------------------------------------------------------------------------
    def _check(self, pids, where: str) -> list[int]:
        out = []
        for p in pids:
            if p not in self.pages:
                raise PagingError(f"page {p} does not exist")
            page = self.pages[p]
            if page.where != where:
                raise PagingError(f"page {p} is {page.where}, expected {where}")
            if where == "gpu" and (page.pinned or p == self.current):
                raise PagingError(f"page {p} is pinned or being written")
            out.append(p)
        return out

    def page_out(self, pids) -> list[int]:
        pids = self._check(pids, "gpu")
        self.snapshots.append(self.resident() - set(pids))
        for p in pids:
            self.pages[p].where = "host"
        return pids

    def page_in(self, pids, budget: int | None = None) -> list[int]:
        pids = self._check(pids, "host")
        new = self.resident() | set(pids)
        if budget is not None and sum(self.pages[p].length for p in new) > budget:
            raise PagingError("page_in would exceed the budget")
        self.snapshots.append(new)
        for p in pids:
            self.pages[p].where = "gpu"
        return pids

    def drop(self, pids) -> list[int]:
        """Irreversible removal (only offered in the no-host-tier ablation)."""
        pids = [p for p in pids if p in self.pages]
        resident = [p for p in pids if self.pages[p].where == "gpu"]
        self._check(resident, "gpu")
        self.snapshots.append(self.resident() - set(pids))
        for p in pids:
            self.pages[p].where = "dropped"
        return pids

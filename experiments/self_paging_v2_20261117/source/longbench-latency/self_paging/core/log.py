"""Token log: the bookkeeping half of self-paging, without a model.

``TokenLog`` records every token with its page and resident-set version. That is all SFT data
generation needs (the oracle does not run a model) and all a trainer needs to rebuild the exact
attention mask. ``self_paging.hf.logstore.PagedKV`` subclasses it and adds the KV cache.
"""

from __future__ import annotations

import torch

from .pages import PageTable, PagingError


class TokenLog:
    def __init__(self, budget: int | None = None, strict: bool = False) -> None:
        self.budget = budget
        self.strict = strict  # raise when a feed would exceed the budget (tests); episodes truncate instead
        self.table = PageTable()
        self.log_len = 0
        self.tokens: list[int] = []
        self.page_of: list[int] = []
        self.seg_of: list[int] = []
        self.peak_resident = 0

    # -- pages -----------------------------------------------------------------------------
    @property
    def next_pid(self) -> int:
        return len(self.table.pages)

    def open(self, kind: str, pinned: bool = False) -> int:
        return self.table.open(kind, self.log_len, pinned=pinned)

    def page_out(self, pids, physical: bool = False) -> list[int]:
        return self.table.page_out(pids)

    def page_in(self, pids) -> list[int]:
        return self.table.page_in(pids, budget=self.budget)

    def drop(self, pids) -> list[int]:
        return self.table.drop(pids)

    def resident_tokens(self) -> int:
        return self.table.resident_tokens()

    # -- writing ---------------------------------------------------------------------------
    def _record(self, pid: int, ids: list[int]) -> None:
        if not ids:
            raise PagingError("nothing to feed")
        if self.strict and self.budget is not None and self.resident_tokens() + len(ids) > self.budget:
            raise PagingError("resident set would exceed the budget")
        n = len(ids)
        self.table.grow(pid, n)
        self.tokens += list(ids)
        self.page_of += [pid] * n
        self.seg_of += [self.table.version] * n
        self.log_len += n
        self.peak_resident = max(self.peak_resident, self.resident_tokens())

    def feed(self, pid: int, ids: list[int]):
        self._record(pid, ids)
        return None

    # -- export ----------------------------------------------------------------------------
    def record(self) -> dict:
        """JSON-serialisable record (lists only)."""
        return {
            "tokens": list(self.tokens),
            "page_of": list(self.page_of),
            "seg_of": list(self.seg_of),
            "snapshots": [sorted(s) for s in self.table.snapshots],
            "pages": [{"pid": p.pid, "kind": p.kind, "start": p.start, "length": p.length,
                       "pinned": p.pinned} for p in self.table.pages.values()],
        }

    def trajectory(self) -> dict:
        from ..hf.masks import resident_matrix  # local import keeps core free of hf at import time

        return {
            "tokens": torch.tensor(self.tokens),
            "page_of": torch.tensor(self.page_of),
            "seg_of": torch.tensor(self.seg_of),
            "resident": resident_matrix(self.table.snapshots, len(self.table.pages)),
        }

"""Self-paging KV store on top of a HuggingFace decoder (dense, every layer full attention).

Usage::

    kv = PagedKV(model)
    sys = kv.open("system", pinned=True); kv.feed(sys, system_ids)
    obs = kv.open("tool");                kv.feed(obs, tool_ids)
    kv.page_out([obs])                    # model issued kv_out
    ans = kv.open("assistant");           logits = kv.feed(ans, [tok])   # decode token by token

How a forward works (docs/algorithm_design.md §3.2):
  1. gather the KV of resident pages (plus the page being written) in log order into a fresh
     ``DynamicCache``; its length is L_res;
  2. run the new tokens with ``position_ids`` = their *global* log positions, and an explicit
     additive mask: every gathered token visible, causal among the new tokens;
  3. cut the new tokens' KV off the end of the cache and append it to the page.

Keys in the cache are already rotated (Qwen3 applies RoPE before ``cache.update``), so a key's
position travels with it; gaps in positions are harmless. The result equals one forward over the
whole log with ``masks.visibility`` (tested in ``tests/test_equivalence.py``).

``physical=True`` on ``page_out`` really moves the page's tensors to host memory (pinned when the
model is on CUDA); ``page_in`` moves them back. Logical paging (the default) only changes the
resident set, which is all RL rollouts need.
"""

from __future__ import annotations

import torch
from transformers import DynamicCache

from ..core.pages import PageTable, PagingError
from .masks import resident_matrix


class PagedKV:
    def __init__(self, model, budget: int | None = None) -> None:
        self.model = model
        self.n_layers = model.config.num_hidden_layers
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.budget = budget
        self.table = PageTable()
        self.kv: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}
        self.log_len = 0
        self.tokens: list[int] = []
        self.page_of: list[int] = []
        self.seg_of: list[int] = []

    # -- pages -----------------------------------------------------------------------------
    def open(self, kind: str, pinned: bool = False) -> int:
        pid = self.table.open(kind, self.log_len, pinned=pinned)
        self.kv[pid] = []
        return pid

    def page_out(self, pids, physical: bool = False) -> None:
        pids = self.table.page_out(pids)
        if physical:
            pin = self.device.type == "cuda"
            for p in pids:
                self.kv[p] = [(k.to("cpu", non_blocking=pin).pin_memory() if pin else k.to("cpu"),
                               v.to("cpu", non_blocking=pin).pin_memory() if pin else v.to("cpu"))
                              for k, v in self.kv[p]]

    def page_in(self, pids) -> None:
        pids = self.table.page_in(pids, budget=self.budget)
        for p in pids:
            self.kv[p] = [(k.to(self.device, non_blocking=True), v.to(self.device, non_blocking=True))
                          for k, v in self.kv[p]]

    def drop(self, pids) -> None:
        for p in self.table.drop(pids):
            self.kv[p] = []

    # -- forward ---------------------------------------------------------------------------
    def _visible_pages(self, pid: int) -> list[int]:
        return self.table.in_order(self.table.resident() | {pid})

    def _gather(self, pid: int) -> tuple[DynamicCache, int]:
        cache = DynamicCache(config=self.model.config)
        pages = [p for p in self._visible_pages(pid) if self.kv[p]]
        if not pages:
            return cache, 0
        for layer in range(self.n_layers):
            k = torch.cat([self.kv[p][layer][0] for p in pages], dim=-2)
            v = torch.cat([self.kv[p][layer][1] for p in pages], dim=-2)
            cache.update(k, v, layer)
        return cache, cache.get_seq_length()

    @torch.no_grad()
    def feed(self, pid: int, ids: list[int]) -> torch.Tensor:
        """Append ``ids`` to page ``pid`` (the page being written). Returns logits [n, vocab]."""
        if not ids:
            raise PagingError("nothing to feed")
        if self.budget is not None and self.table.resident_tokens() + len(ids) > self.budget:
            raise PagingError("resident set would exceed the budget")
        n = len(ids)
        cache, l_res = self._gather(pid)
        mask = torch.zeros(1, 1, n, l_res + n, dtype=self.dtype, device=self.device)
        mask[0, 0, :, l_res:] = torch.triu(
            torch.full((n, n), torch.finfo(self.dtype).min, dtype=self.dtype, device=self.device), 1)
        pos = torch.arange(self.log_len, self.log_len + n, device=self.device)[None]
        out = self.model(input_ids=torch.tensor([ids], device=self.device), position_ids=pos,
                         attention_mask=mask, past_key_values=cache, use_cache=True)
        for layer in range(self.n_layers):
            lay = cache.layers[layer]
            k_new, v_new = lay.keys[:, :, l_res:], lay.values[:, :, l_res:]
            if len(self.kv[pid]) < self.n_layers:
                self.kv[pid].append((k_new.clone(), v_new.clone()))
            else:
                k_old, v_old = self.kv[pid][layer]
                self.kv[pid][layer] = (torch.cat([k_old, k_new], -2), torch.cat([v_old, v_new], -2))
        self.table.grow(pid, n)
        self.tokens += list(ids)
        self.page_of += [pid] * n
        self.seg_of += [self.table.version] * n
        self.log_len += n
        return out.logits[0]

    # -- export ----------------------------------------------------------------------------
    def trajectory(self) -> dict:
        """Everything a trainer needs to rebuild the mask: token ids, page and version per token,
        and the resident matrix."""
        return {
            "tokens": torch.tensor(self.tokens),
            "page_of": torch.tensor(self.page_of),
            "seg_of": torch.tensor(self.seg_of),
            "resident": resident_matrix(self.table.snapshots, len(self.table.pages)),
        }

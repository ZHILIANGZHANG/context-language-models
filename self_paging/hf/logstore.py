"""Self-paging KV store on top of a HuggingFace decoder (dense, every layer full attention).

Usage::

    kv = PagedKV(model)
    sys = kv.open("system", pinned=True); kv.feed(sys, system_ids)
    obs = kv.open("tool");                kv.feed(obs, tool_ids)
    kv.page_out([obs])                    # the model issued kv_out
    ans = kv.open("assistant");           logits = kv.feed(ans, [tok])   # decode token by token

How a forward works (docs/algorithm_design.md §3.2):
  1. if the resident set changed since the last forward, gather the KV of the resident pages (plus
     the page being written) in log order into a fresh ``DynamicCache``; otherwise keep appending to
     the live cache;
  2. run the new tokens with ``position_ids`` = their *global* log positions, and an explicit
     additive mask: every cached token visible, causal among the new tokens;
  3. copy the new tokens' KV off the end of the cache into the page store.

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

from ..core.log import TokenLog


class PagedKV(TokenLog):
    def __init__(self, model, budget: int | None = None, strict: bool = True) -> None:
        super().__init__(budget=budget, strict=strict)
        self.model = model
        self.n_layers = model.config.num_hidden_layers
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.kv: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}
        self._live: DynamicCache | None = None  # valid while the resident set is unchanged

    # -- pages -----------------------------------------------------------------------------
    def open(self, kind: str, pinned: bool = False) -> int:
        pid = super().open(kind, pinned=pinned)
        self.kv[pid] = []
        return pid

    def page_out(self, pids, physical: bool = False) -> list[int]:
        pids = super().page_out(pids)
        self._live = None
        if physical:
            pin = self.device.type == "cuda"
            for p in pids:
                self.kv[p] = [(self._to_host(k, pin), self._to_host(v, pin)) for k, v in self.kv[p]]
        return pids

    @staticmethod
    def _to_host(t: torch.Tensor, pin: bool) -> torch.Tensor:
        return t.to("cpu", non_blocking=pin).pin_memory() if pin else t.to("cpu")

    def page_in(self, pids) -> list[int]:
        pids = super().page_in(pids)
        self._live = None
        for p in pids:
            self.kv[p] = [(k.to(self.device, non_blocking=True), v.to(self.device, non_blocking=True))
                          for k, v in self.kv[p]]
        return pids

    def drop(self, pids) -> list[int]:
        pids = super().drop(pids)
        self._live = None
        for p in pids:
            self.kv[p] = []
        return pids

    # -- forward ---------------------------------------------------------------------------
    def _gather(self, pid: int) -> DynamicCache:
        cache = DynamicCache(config=self.model.config)
        pages = [p for p in self.table.in_order(self.table.resident() | {pid}) if self.kv[p]]
        if pages:
            for layer in range(self.n_layers):
                k = torch.cat([self.kv[p][layer][0] for p in pages], dim=-2)
                v = torch.cat([self.kv[p][layer][1] for p in pages], dim=-2)
                cache.update(k, v, layer)
        return cache

    @torch.no_grad()
    def feed(self, pid: int, ids: list[int]) -> torch.Tensor:
        """Append ``ids`` to page ``pid`` (the page being written). Returns logits [n, vocab]."""
        n = len(ids)
        cache = self._live if self._live is not None else self._gather(pid)
        l_res = cache.get_seq_length()
        mask = torch.zeros(1, 1, n, l_res + n, dtype=self.dtype, device=self.device)
        mask[0, 0, :, l_res:] = torch.triu(
            torch.full((n, n), torch.finfo(self.dtype).min, dtype=self.dtype, device=self.device), 1)
        pos = torch.arange(self.log_len, self.log_len + n, device=self.device)[None]
        out = self.model(input_ids=torch.tensor([ids], device=self.device), position_ids=pos,
                         attention_mask=mask, past_key_values=cache, use_cache=True)
        for layer in range(self.n_layers):
            lay = cache.layers[layer]
            k_new, v_new = lay.keys[:, :, l_res:].clone(), lay.values[:, :, l_res:].clone()
            if len(self.kv[pid]) < self.n_layers:
                self.kv[pid].append((k_new, v_new))
            else:
                k_old, v_old = self.kv[pid][layer]
                self.kv[pid][layer] = (torch.cat([k_old, k_new], -2), torch.cat([v_old, v_new], -2))
        self._live = cache
        self._record(pid, ids)
        return out.logits[0]

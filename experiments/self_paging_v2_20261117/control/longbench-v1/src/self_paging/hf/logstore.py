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

Token-level retention (docs/baseline_protocol.md §7, D10): ``set_keep`` installs a boolean
``keep[layer, kv_head, position]``. A token is visible iff its page is resident *and* it is kept for
that layer and KV head, so system baselines that prune tokens (StreamingLLM, H2O, TOVA, SnapKV) or
select token chunks (Quest) run on the same engine. Every (layer, head) must keep the same number of
resident tokens, so the gathered cache keeps one shape; a change starts a new resident-set version.
Without ``set_keep`` (``keep is None``) nothing changes. ``resident_tokens`` then counts kept tokens.
With ``record_keep=True`` every installed mask is kept (``keep_history``) so a test can rebuild the
per-layer, per-head visibility (``token_visibility``) and check it against one masked forward.

``physical=True`` on ``page_out`` really moves the page's tensors to host memory (pinned when the
model is on CUDA); ``page_in`` moves them back. Logical paging (the default) only changes the
resident set, which is all RL rollouts need.
"""

from __future__ import annotations

import torch
from transformers import DynamicCache

from ..core.log import TokenLog


class PagedKV(TokenLog):
    def __init__(self, model, budget: int | None = None, strict: bool = True,
                 store_device: str | None = None, prefill_chunk: int | None = None,
                 last_logits_only: bool = False, record_keep: bool = False) -> None:
        super().__init__(budget=budget, strict=strict)
        self.model = model
        self.n_layers = model.config.num_hidden_layers
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        # Optional CPU backing holds every immutable page once while the live
        # resident cache stays on the model device. This avoids retaining two
        # GPU copies of an entire 120K log. It changes transfers, not visibility
        # or positions, and must not be used for HF wall-clock speed claims.
        self.store_device = torch.device(store_device) if store_device else self.device
        if prefill_chunk is not None and prefill_chunk < 1:
            raise ValueError('prefill_chunk must be positive')
        self.prefill_chunk, self.last_logits_only = prefill_chunk, last_logits_only
        self.kv: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}
        self._live: DynamicCache | None = None  # valid while the resident set is unchanged
        # token-level retention (D10)
        self.n_kv_heads = model.config.num_key_value_heads
        self.keep: torch.Tensor | None = None   # bool [layers, kv_heads, capacity]; None = keep all
        self.record_keep = record_keep
        self.keep_history: list[tuple[int, torch.Tensor]] = []  # (version, keep[:, :, :log_len]) on CPU
        self.keep_changes = 0
        self.keep_equalized = 0
        self._hidden_log: list = []  # [version, [[pid, hidden tokens], ...]] for the cost (core/costs.py)
        self._hidden_min: tuple[int, int] | None = None  # (version, min hidden resident tokens over heads)
        self._live_pos: list[torch.Tensor] | None = None  # per layer [kv_heads, n] positions of gathered keys
        self._live_start = 0  # log position of the first token appended to the live cache after gathering

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
            self.kv[p] = [(k.to(self.store_device, non_blocking=True), v.to(self.store_device, non_blocking=True))
                          for k, v in self.kv[p]]
        return pids

    def drop(self, pids) -> list[int]:
        pids = super().drop(pids)
        self._live = None
        for p in pids:
            self.kv[p] = []
        return pids

    # -- token-level retention (D10) -----------------------------------------------------------
    def keep_view(self) -> torch.Tensor:
        """Bool [layers, kv_heads, log_len] retention mask (all True when no mask was installed)."""
        if self.keep is None:
            return torch.ones(self.n_layers, self.n_kv_heads, self.log_len, dtype=torch.bool, device=self.device)
        return self.keep[:, :, :self.log_len]

    def visible_positions(self, pid: int | None = None) -> torch.Tensor:
        """Log positions of the resident pages (plus ``pid``) that still hold KV, in log order."""
        pid = self.table.current if pid is None else pid
        extra = {pid} if pid is not None else set()
        pages = [p for p in self.table.in_order(self.table.resident() | extra) if self.kv.get(p)]
        if not pages:
            return torch.zeros(0, dtype=torch.long, device=self.device)
        return torch.cat([torch.arange(self.table.pages[p].start, self.table.pages[p].end, device=self.device)
                          for p in pages])

    def _grow_keep(self) -> None:
        if self.keep is not None and self.keep.shape[-1] < self.log_len:
            extra = max(self.log_len - self.keep.shape[-1], self.keep.shape[-1])
            self.keep = torch.cat([self.keep, torch.ones(self.n_layers, self.n_kv_heads, extra, dtype=torch.bool,
                                                         device=self.device)], -1)

    def set_keep(self, keep: torch.Tensor) -> int:
        """Install a retention mask [layers, kv_heads, log_len] (or broadcastable). Tokens of the page
        being written must stay kept, and every (layer, head) must keep the same number of resident
        tokens. Starts a new version and returns it."""
        keep = keep.to(device=self.device, dtype=torch.bool).expand(self.n_layers, self.n_kv_heads, self.log_len)
        cur = self.table.current
        if cur is not None:
            page = self.table.pages[cur]
            if not keep[:, :, page.start:page.end].all():
                raise ValueError("tokens of the page being written must stay kept")
        pos = self.visible_positions()
        counts = keep[:, :, pos].sum(-1)
        if pos.numel() and not (counts == counts.flatten()[0]).all():
            raise ValueError("every (layer, head) must keep the same number of resident tokens")
        if self.keep is None:
            self.keep = torch.ones(self.n_layers, self.n_kv_heads, max(1024, 2 * self.log_len), dtype=torch.bool,
                                   device=self.device)
        self._grow_keep()
        self.keep[:, :, :self.log_len] = keep
        return self._keep_changed()

    def _keep_changed(self) -> int:
        version = self.table.bump()
        self._live = None
        self.keep_changes += 1
        # hidden tokens per page (layer 0, head 0: every head keeps the same number of visible tokens)
        row = ~self.keep[0, 0]
        self._hidden_log.append([version, [[p, int(row[pg.start:pg.end].sum())] for p, pg in self.table.pages.items()
                                           if pg.length and row[pg.start:pg.end].any()]])
        if self.record_keep:
            self.keep_history.append((version, self.keep[:, :, :self.log_len].cpu().clone()))
        return version

    def _equalize(self) -> None:
        """Page-level moves (forced evictions, a page coming back) can leave heads with different
        numbers of kept resident tokens. Hide the oldest surplus tokens of the heads that keep more
        (never the page being written); counted in ``keep_equalized``."""
        pos = self.visible_positions()
        keep = self.keep[:, :, pos]
        counts = keep.sum(-1)
        low = int(counts.min())
        if (counts == low).all():
            return
        cur = self.table.pages[self.table.current] if self.table.current is not None else None
        movable = keep.clone()
        if cur is not None:
            movable &= ~((pos >= cur.start) & (pos < cur.end))
        surplus = counts - low
        rank = movable.long().cumsum(-1)  # 1-based rank of each movable kept token, oldest first
        hide = movable & (rank <= surplus[..., None])
        if (hide.sum(-1) != surplus).any():
            raise ValueError("cannot equalize kept tokens across heads")
        self.keep[:, :, pos] = keep & ~hide
        self.keep_equalized += 1
        self._keep_changed()

    def record(self) -> dict:
        rec = super().record()
        if self.keep is not None:
            rec["hidden_pages"] = [list(x) for x in self._hidden_log]
            rec["keep_changes"], rec["keep_equalized"] = self.keep_changes, self.keep_equalized
        return rec

    def resident_tokens(self) -> int:
        if self.keep is None:
            return super().resident_tokens()
        v = self.table.version
        if self._hidden_min is None or self._hidden_min[0] != v:
            self._grow_keep()
            pos = self.visible_positions()
            hidden = (~self.keep[:, :, pos]).sum(-1)
            self._hidden_min = (v, int(hidden.min()) if pos.numel() else 0)
        return super().resident_tokens() - self._hidden_min[1]

    def cache_positions(self, layer: int) -> torch.Tensor:
        """[kv_heads, n] log positions of the live cache's keys of ``layer`` (column order)."""
        if self._live is None:
            raise ValueError("no live cache")
        new = torch.arange(self._live_start, self.log_len, device=self.device).expand(self.n_kv_heads, -1)
        return torch.cat([self._live_pos[layer], new], -1)

    def visibility_parts(self) -> dict:
        """Compact form of the per-layer, per-head visibility (needs ``record_keep=True``):
        ``page_of``/``seg_of`` [T], ``resident`` [S, P] and ``keep_at`` [S, layers, kv_heads, T] (the
        retention mask in force at each version)."""
        if not self.record_keep and self.keep_changes:
            raise ValueError("token visibility needs PagedKV(record_keep=True)")
        traj = self.trajectory()
        t, n_versions = len(self.tokens), len(self.table.snapshots)
        keep_at = torch.ones(n_versions, self.n_layers, self.n_kv_heads, t, dtype=torch.bool)
        history = sorted(self.keep_history, key=lambda x: x[0])
        for i, (v, k) in enumerate(history):
            end = history[i + 1][0] if i + 1 < len(history) else n_versions
            keep_at[v:end, :, :, :k.shape[-1]] = k
        return {"page_of": traj["page_of"], "seg_of": traj["seg_of"], "resident": traj["resident"],
                "keep_at": keep_at}

    @staticmethod
    def visibility_rows(parts: dict, layer: int, rows: torch.Tensor) -> torch.Tensor:
        """Bool [kv_heads, len(rows), T] for query positions ``rows``: causal, and the key's page is the
        query's own page, or it is resident at the query's version and the key is kept there."""
        page_of, seg = parts["page_of"], parts["seg_of"]
        t = page_of.shape[0]
        causal = torch.arange(t)[None, :] <= rows[:, None]
        own = page_of[rows][:, None] == page_of[None, :]
        res = parts["resident"][seg[rows][:, None], page_of[None, :]]
        kept = parts["keep_at"][seg[rows], layer].transpose(0, 1)  # [kv_heads, rows, T]
        return causal[None] & (own[None] | (res[None] & kept))

    def token_visibility(self) -> torch.Tensor:
        """Bool [layers, kv_heads, T, T] (small logs only; see ``visibility_rows``)."""
        parts = self.visibility_parts()
        rows = torch.arange(len(self.tokens))
        return torch.stack([self.visibility_rows(parts, layer, rows) for layer in range(self.n_layers)])

    # -- forward ---------------------------------------------------------------------------
    def _gather(self, pid: int) -> DynamicCache:
        cache = DynamicCache(config=self.model.config)
        pages = [p for p in self.table.in_order(self.table.resident() | {pid}) if self.kv[p]]
        self._live_pos = []
        if self.keep is not None and pages:
            self._grow_keep()
            self._equalize()
        if pages:
            pos = torch.cat([torch.arange(self.table.pages[p].start, self.table.pages[p].end, device=self.device)
                             for p in pages])
            for layer in range(self.n_layers):
                k = torch.cat([self.kv[p][layer][0].to(self.device) for p in pages], dim=-2)
                v = torch.cat([self.kv[p][layer][1].to(self.device) for p in pages], dim=-2)
                if self.keep is None:
                    self._live_pos.append(pos.expand(self.n_kv_heads, -1))
                else:
                    m = self.keep[layer][:, pos]  # [kv_heads, N]
                    n = int(m[0].sum())
                    idx = m.nonzero()[:, 1].view(self.n_kv_heads, n)  # row-major: per head, in log order
                    gi = idx[None, :, :, None].expand(k.shape[0], -1, -1, k.shape[-1])
                    k, v = k.gather(2, gi), v.gather(2, gi)
                    self._live_pos.append(pos[idx])
                cache.update(k, v, layer)
        else:
            self._live_pos = [torch.zeros(self.n_kv_heads, 0, dtype=torch.long, device=self.device)
                              for _ in range(self.n_layers)]
        self._live_start = self.log_len
        return cache

    @torch.no_grad()
    def feed(self, pid: int, ids: list[int]) -> torch.Tensor:
        """Append tokens; return [n,vocab], or [1,vocab] in last-logit-only mode."""
        if self.prefill_chunk and len(ids) > self.prefill_chunk:
            chunks = []
            for start in range(0, len(ids), self.prefill_chunk):
                part = self.feed(pid, ids[start:start + self.prefill_chunk])
                if self.last_logits_only:
                    chunks = [part]
                else:
                    chunks.append(part)
            return torch.cat(chunks, 0)
        n = len(ids)
        cache = self._live if self._live is not None else self._gather(pid)
        l_res = cache.get_seq_length()
        mask = torch.zeros(1, 1, n, l_res + n, dtype=self.dtype, device=self.device)
        mask[0, 0, :, l_res:] = torch.triu(
            torch.full((n, n), torch.finfo(self.dtype).min, dtype=self.dtype, device=self.device), 1)
        pos = torch.arange(self.log_len, self.log_len + n, device=self.device)[None]
        out = self.model(input_ids=torch.tensor([ids], device=self.device), position_ids=pos,
                         attention_mask=mask, past_key_values=cache, use_cache=True,
                         **({'logits_to_keep': 1} if self.last_logits_only else {}))
        for layer in range(self.n_layers):
            lay = cache.layers[layer]
            k_new = lay.keys[:, :, l_res:].to(self.store_device, copy=True)
            v_new = lay.values[:, :, l_res:].to(self.store_device, copy=True)
            if len(self.kv[pid]) < self.n_layers:
                self.kv[pid].append((k_new, v_new))
            else:
                k_old, v_old = self.kv[pid][layer]
                self.kv[pid][layer] = (torch.cat([k_old, k_new], -2), torch.cat([v_old, v_new], -2))
        self._live = cache
        self._record(pid, ids)
        if self.keep is not None:
            self._grow_keep()
        return out.logits[0]

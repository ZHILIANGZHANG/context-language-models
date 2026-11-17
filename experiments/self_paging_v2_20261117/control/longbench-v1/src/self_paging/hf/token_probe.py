"""Query and attention telemetry for token-level system baselines (D10, docs/baseline_protocol.md §7).

``ProbedPagedKV`` is a ``PagedKV`` that records, without changing any attention output:

* the post-norm, post-RoPE queries of the most recent ``window`` tokens of every layer
  (``window_queries``), for the question-aware scorers (TOVA, SnapKV, Quest);
* optionally (``accumulate=True``) the attention every fed query pays to every visible key, summed
  per (layer, KV head, log position) in ``acc``: H2O's accumulated attention. It is recomputed from
  the cached keys in float32, chunked over queries, at the instant of each feed (the visible set is
  the one the query really saw).

``window_attention(layer)`` returns the softmax attention of the window queries over the keys that
are visible *now* (resident pages, kept tokens), which is what SnapKV and TOVA score at
compression time. All of this is an HF analytic prototype: no latency claims.
"""

from __future__ import annotations

import math

import torch

from .logstore import PagedKV


class ProbedPagedKV(PagedKV):
    def __init__(self, model, budget=None, strict=True, *, window: int = 32, accumulate: bool = False,
                 query_chunk: int = 64, record_keep: bool = False):
        super().__init__(model, budget=budget, strict=strict, record_keep=record_keep)
        if window < 1:
            raise ValueError("window must be positive")
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        self.attn_layers = [layer.self_attn for layer in base.model.layers]
        if len(self.attn_layers) != self.n_layers:
            raise ValueError("expected one dense attention module per layer")
        if any(getattr(a, "sliding_window", None) is not None for a in self.attn_layers):
            raise ValueError("token telemetry requires dense attention in every layer")
        self.window, self.accumulate, self.query_chunk = window, accumulate, query_chunk
        self.groups = model.config.num_attention_heads // self.n_kv_heads
        self.head_dim = self.attn_layers[0].head_dim
        self.win_q: list[torch.Tensor | None] = [None] * self.n_layers  # [Hq, w, d] float32
        self.win_pos = torch.zeros(0, dtype=torch.long, device=self.device)
        self.acc: torch.Tensor | None = None  # [layers, kv_heads, capacity] float32

    # -- capture ---------------------------------------------------------------------------
    @torch.no_grad()
    def feed(self, pid: int, ids: list[int]) -> torch.Tensor:
        if self.prefill_chunk and len(ids) > self.prefill_chunk:
            out = None
            for start in range(0, len(ids), self.prefill_chunk):
                part = self.feed(pid, ids[start:start + self.prefill_chunk])
                out = part if out is None or self.last_logits_only else torch.cat([out, part], 0)
            return out
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

        keep_n = len(ids) if self.accumulate else min(len(ids), self.window)
        queries: dict[int, torch.Tensor] = {}

        def hook(index):
            def capture(module, args, kwargs):
                h = kwargs.get("hidden_states", args[0] if args else None)[:, -keep_n:]
                cos, sin = kwargs["position_embeddings"]
                q = module.q_norm(module.q_proj(h).view(*h.shape[:-1], -1, module.head_dim)).transpose(1, 2)
                q, _ = apply_rotary_pos_emb(q, q, cos[:, -keep_n:], sin[:, -keep_n:])
                queries[index] = q[0].float()
            return capture

        handles = [a.register_forward_pre_hook(hook(i), with_kwargs=True) for i, a in enumerate(self.attn_layers)]
        start = self.log_len
        try:
            logits = super().feed(pid, ids)
        finally:
            for h in handles:
                h.remove()
        if len(queries) != self.n_layers:
            raise RuntimeError("incomplete per-layer query telemetry")
        n = len(ids)
        qpos = torch.arange(start + n - keep_n, start + n, device=self.device)
        if self.accumulate:
            self._accumulate(queries, qpos)
        w = min(keep_n, self.window)
        for layer, q in queries.items():
            prev = self.win_q[layer]
            q = q[:, -w:]
            self.win_q[layer] = (q if prev is None else torch.cat([prev, q], 1))[:, -self.window:]
        self.win_pos = torch.cat([self.win_pos, qpos[-w:]])[-self.window:]
        return logits

    def _accumulate(self, queries, qpos) -> None:
        cap = max(1024, 2 * self.log_len)
        if self.acc is None:
            self.acc = torch.zeros(self.n_layers, self.n_kv_heads, cap, device=self.device)
        elif self.acc.shape[-1] < self.log_len:
            self.acc = torch.cat([self.acc, torch.zeros(self.n_layers, self.n_kv_heads, cap - self.acc.shape[-1],
                                                        device=self.device)], -1)
        for layer, q in queries.items():
            keys = self._live.layers[layer].keys[0].float()  # [Hkv, Lc, d]
            pos = self.cache_positions(layer)                # [Hkv, Lc]
            for a in range(0, q.shape[1], self.query_chunk):
                w = self._softmax(q[:, a:a + self.query_chunk], keys, pos, qpos[a:a + self.query_chunk])
                self.acc[layer].scatter_add_(1, pos, w.sum(1).view(self.n_kv_heads, self.groups, -1).sum(1))

    def _softmax(self, q, keys, pos, qpos) -> torch.Tensor:
        """[Hq, nq, Lc] attention of queries ``q`` [Hq, nq, d] over ``keys`` [Hkv, Lc, d] (causal by
        log position)."""
        k = keys.repeat_interleave(self.groups, 0)
        p = pos.repeat_interleave(self.groups, 0)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(p[:, None, :] > qpos[None, :, None], float("-inf"))
        return scores.softmax(-1)

    # -- read ------------------------------------------------------------------------------
    def visible_keys(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Keys [Hkv, n, d] (float32) and their log positions [Hkv, n] visible right now."""
        if self._live is None:
            self._live = self._gather(self.table.current)
        return self._live.layers[layer].keys[0].float(), self.cache_positions(layer)

    def window_attention(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Attention [Hq, w, n] of the window queries over the visible keys, and the keys' positions
        [Hkv, n]."""
        if self.win_q[layer] is None:
            raise ValueError("no queries observed yet")
        keys, pos = self.visible_keys(layer)
        return self._softmax(self.win_q[layer], keys, pos, self.win_pos), pos

    def all_keys(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Keys [Hkv, N, d] of every resident token (kept or not; Quest's chunk bounds are built over
        the whole stored KV) and positions [N]."""
        pages = [p for p in self.table.in_order(self.table.resident()) if self.kv.get(p)]
        if not pages:
            return (torch.zeros(self.n_kv_heads, 0, self.head_dim, device=self.device),
                    torch.zeros(0, dtype=torch.long, device=self.device))
        k = torch.cat([self.kv[p][layer][0].to(self.device) for p in pages], -2)[0].float()
        pos = torch.cat([torch.arange(self.table.pages[p].start, self.table.pages[p].end, device=self.device)
                         for p in pages])
        return k, pos

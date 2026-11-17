"""Token-level system baselines (D11, docs/baseline_protocol.md §1, §4).

The system, not the model, decides which tokens stay visible; the model only does the task
(``kv_tools=False``). All of them run on ``PagedKV``'s retention mask (D10) and are called like the
page-level pagers: before each assistant turn and before each tool block, with the number of tokens
about to be written, so the budget holds at every moment (strict: no prefill-then-compress peak).

| class | scores | granularity | reversible | sees the question |
|---|---|---|---|---|
| ``StreamingLLMPager`` | attention sinks + most recent tokens | shared by all layers and heads | no | no |
| ``H2OPager`` | attention accumulated over every query so far, plus a recent window | per layer, per KV head | no | no |
| ``TOVAPager`` | attention of the latest query, averaged over the heads of a layer | per layer | no | yes (latest query) |
| ``SnapKVPager`` | attention of an observation window (the most recent tokens) summed, then 1D pooled | per layer, per KV head | no | yes (window) |
| ``QuestPager`` | per 16-token chunk, the upper bound Σ_d max(q_d·min_d, q_d·max_d) of the window queries | per layer, per KV head | yes (reselects every call) | yes (window) |

Faithfulness notes (what differs from the papers, and why):
* compression happens between turns, never inside a generation, like our method (§4);
* every (layer, head) keeps the same number of tokens (one cache shape); PyramidKV's per-layer
  budgets are therefore not implemented;
* Quest's first two dense layers are not exempt (they would exceed the shared budget); it scores with
  the mean upper bound of the observation window, not a fresh query at every decoding step;
* GQA (Qwen3-8B: 32 query heads, 8 KV heads): per-KV-head scores sum the attention of the query heads
  that share the KV head (kvpress does the same);
* scores are recomputed from the cached keys in float32 by ``hf.token_probe.ProbedPagedKV``.
``protect_pinned`` (default True) keeps the structural pinned pages (system prompt with the tool
schemas, headers, block markers) like our method's budget does; it is a tunable (§5), and setting it
to False gives the flat-sequence variant of the papers. The page being written and the transient
gauge / acknowledgement pages are always kept.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .pagers import _overhead

ALWAYS_KEEP_KINDS = ("gauge", "ack")


# -- pure scoring and selection (unit-tested with hand-made tensors) --------------------------------
def select_keep(scores: torch.Tensor, keep_now: torch.Tensor, protect: torch.Tensor, target: int,
                reversible: bool) -> torch.Tensor:
    """Bool [L, H, N]: keep ``target`` tokens per (layer, head) by score, always keeping ``protect``.
    Irreversible methods only choose among tokens they still keep."""
    L, H, N = keep_now.shape
    s = scores.float().expand(L, H, N).clone()
    if not reversible:
        s[~keep_now] = float("-inf")
    s[:, :, protect] = float("inf")
    k = max(int(protect.sum()), min(int(target), N))
    if not reversible:
        k = min(k, int(keep_now[0, 0].sum()))
    out = torch.zeros(L, H, N, dtype=torch.bool, device=keep_now.device)
    if k > 0:
        out.scatter_(-1, s.topk(k, -1).indices, True)
    return out


def streaming_scores(pos: torch.Tensor, sink: int) -> torch.Tensor:
    s = pos.float().clone()
    s[pos < sink] = float("inf")
    return s[None, None]


def heads_to_kv(x: torch.Tensor, groups: int) -> torch.Tensor:
    """[Hq, ...] -> [Hkv, ...]: sum over the query heads sharing a KV head."""
    return x.view(x.shape[0] // groups, groups, *x.shape[1:]).sum(1)


def dense_columns(values: torch.Tensor, cols: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Scatter per-head column values [H, n] at log positions ``cols`` [H, n] into [H, N] over the
    candidate positions ``pos`` (sorted); positions not visible get 0."""
    idx = torch.searchsorted(pos, cols.contiguous())
    out = torch.zeros(values.shape[0], pos.numel(), device=values.device, dtype=values.dtype)
    return out.scatter_(1, idx.clamp(max=max(pos.numel() - 1, 0)), values)


def tova_layer_scores(attn_last: torch.Tensor, cols: torch.Tensor, pos: torch.Tensor, groups: int) -> torch.Tensor:
    """TOVA: attention of the latest query [Hq, n], averaged over all heads of the layer -> [1, N]."""
    per_kv = heads_to_kv(attn_last, groups) / groups
    return dense_columns(per_kv, cols, pos).mean(0, keepdim=True)


def snapkv_layer_scores(win_attn: torch.Tensor, cols: torch.Tensor, pos: torch.Tensor, groups: int,
                        kernel: int = 5, pooling: str = "avg") -> torch.Tensor:
    """SnapKV: window votes [Hq, w, n] summed over the window, per KV head, pooled along the kept
    sequence (columns are in log order per head) -> [Hkv, N]."""
    votes = heads_to_kv(win_attn.sum(1), groups) / groups  # [Hkv, n]
    if kernel > 1 and votes.shape[-1] > 1:
        pool = F.avg_pool1d if pooling == "avg" else F.max_pool1d
        kwargs = {"count_include_pad": False} if pooling == "avg" else {}
        votes = pool(votes[:, None], kernel, stride=1, padding=kernel // 2, **kwargs)[:, 0, :votes.shape[-1]]
    return dense_columns(votes, cols, pos)


def quest_upper_bound(q: torch.Tensor, kmin: torch.Tensor, kmax: torch.Tensor) -> torch.Tensor:
    """Σ_d max(q_d·kmin_d, q_d·kmax_d) for queries [Hq, w, d] and chunk bounds [Hq, C, d] -> [Hq, w, C]."""
    return q.clamp(min=0) @ kmax.transpose(-1, -2) + q.clamp(max=0) @ kmin.transpose(-1, -2)


def quest_layer_scores(keys: torch.Tensor, pos: torch.Tensor, q: torch.Tensor, groups: int,
                       chunk: int = 16) -> torch.Tensor:
    """Quest: chunks of ``chunk`` consecutive log positions; element-wise key min / max per chunk and
    KV head; a token scores its chunk's mean upper bound over the window queries -> [Hkv, N]."""
    hkv, n, d = keys.shape
    cid = torch.div(pos, chunk, rounding_mode="floor")
    uniq, inv = torch.unique(cid, return_inverse=True)
    c = uniq.numel()
    idx = inv[None, :, None].expand(hkv, n, d)
    kmin = torch.full((hkv, c, d), float("inf"), device=keys.device).scatter_reduce(1, idx, keys, "amin")
    kmax = torch.full((hkv, c, d), float("-inf"), device=keys.device).scatter_reduce(1, idx, keys, "amax")
    ub = quest_upper_bound(q, kmin.repeat_interleave(groups, 0), kmax.repeat_interleave(groups, 0))
    per_chunk = heads_to_kv(ub.mean(1), groups)  # [Hkv, C]
    return per_chunk[:, inv]


# -- pagers ----------------------------------------------------------------------------------
class TokenPager:
    reversible = False
    name = "token"

    def __init__(self, recent: int = 32, protect_pinned: bool = True, sink: int = 0):
        self.recent, self.protect_pinned, self.sink = recent, protect_pinned, sink

    def scores(self, log, pos: torch.Tensor, target: int) -> torch.Tensor:  # [L|1, H|1, N]
        raise NotImplementedError

    def recent_for(self, target: int) -> int:
        return self.recent

    def protect(self, log, pos: torch.Tensor, target: int) -> torch.Tensor:
        t = log.table
        prot = torch.zeros(pos.numel(), dtype=torch.bool, device=pos.device)
        for p in t.resident() | ({t.current} if t.current is not None else set()):
            page = t.pages[p]
            if p == t.current or page.kind in ALWAYS_KEEP_KINDS or (self.protect_pinned and page.pinned):
                prot |= (pos >= page.start) & (pos < page.end)
        r = self.recent_for(target)
        if r > 0:
            prot |= pos >= log.log_len - r
        if self.sink > 0:
            prot |= pos < self.sink
        return prot

    def __call__(self, st, incoming: int) -> None:
        if st.cfg.budget is None:
            return
        target = st.cfg.budget - incoming - _overhead(st) - st.cfg.min_gen_room
        hidden = self.compress(st.log, target)
        if hidden:
            st.meta["token_hidden"] = st.meta.get("token_hidden", 0) + hidden
            st.meta["token_compressions"] = st.meta.get("token_compressions", 0) + 1

    def compress(self, log, target: int) -> int:
        """Shrink (or, if reversible, reselect) the kept resident tokens to ``target`` per head.
        Returns the number of tokens per head newly hidden."""
        if log.keep is not None:
            log._equalize()  # page-level moves can leave heads with different counts
        before = log.resident_tokens()
        if not self.reversible and before <= target:
            return 0
        pos = log.visible_positions()
        if pos.numel() == 0:
            return 0
        keep_all = log.keep_view().clone()
        keep_now = keep_all[:, :, pos]
        sel = select_keep(self.scores(log, pos, target), keep_now, self.protect(log, pos, target),
                          target, self.reversible)
        if torch.equal(sel, keep_now):
            return 0
        keep_all[:, :, pos] = sel
        log.set_keep(keep_all)
        return max(0, before - log.resident_tokens())


class StreamingLLMPager(TokenPager):
    """Attention sinks (the first ``sink`` log tokens) plus the most recent tokens."""
    name = "streaming"

    def __init__(self, sink: int = 4, protect_pinned: bool = True):
        super().__init__(recent=0, protect_pinned=protect_pinned, sink=sink)

    def scores(self, log, pos, target):
        return streaming_scores(pos, self.sink)


class H2OPager(TokenPager):
    """Heavy hitters by accumulated attention, plus a recent window of ``recent_ratio`` × budget.
    Needs ``ProbedPagedKV(accumulate=True)``."""
    name = "h2o"

    def __init__(self, recent_ratio: float = 0.5, protect_pinned: bool = True):
        super().__init__(recent=0, protect_pinned=protect_pinned)
        self.recent_ratio = recent_ratio

    def recent_for(self, target):
        return max(0, int(self.recent_ratio * target))

    def scores(self, log, pos, target):
        if log.acc is None:
            raise ValueError("H2O needs ProbedPagedKV(accumulate=True)")
        return log.acc[:, :, pos]


class TOVAPager(TokenPager):
    """Keep the tokens the latest query attends to most (head-averaged per layer)."""
    name = "tova"

    def __init__(self, recent: int = 0, protect_pinned: bool = True):
        super().__init__(recent=recent, protect_pinned=protect_pinned)

    def scores(self, log, pos, target):
        out = []
        for layer in range(log.n_layers):
            attn, cols = log.window_attention(layer)
            out.append(tova_layer_scores(attn[:, -1], cols, pos, log.groups))
        return torch.stack(out)  # [L, 1, N]


class SnapKVPager(TokenPager):
    """Observation-window votes with 1D pooling; the window itself is always kept. ``window`` must
    not exceed the ``ProbedPagedKV`` window."""
    name = "snapkv"

    def __init__(self, window: int = 32, kernel: int = 5, pooling: str = "avg", protect_pinned: bool = True):
        super().__init__(recent=window, protect_pinned=protect_pinned)
        self.window, self.kernel, self.pooling = window, kernel, pooling

    def scores(self, log, pos, target):
        out = []
        for layer in range(log.n_layers):
            attn, cols = log.window_attention(layer)
            out.append(snapkv_layer_scores(attn[:, -self.window:], cols, pos, log.groups, self.kernel, self.pooling))
        return torch.stack(out)  # [L, Hkv, N]


class QuestPager(TokenPager):
    """Query-aware chunk selection by key bounds; reversible (tokens left out stay in the store and
    can be selected again at the next call)."""
    name = "quest"
    reversible = True

    def __init__(self, chunk: int = 16, window: int = 16, recent: int = 16, protect_pinned: bool = True):
        super().__init__(recent=recent, protect_pinned=protect_pinned)
        self.chunk, self.window = chunk, window

    def scores(self, log, pos, target):
        out = []
        for layer in range(log.n_layers):
            keys, kpos = log.all_keys(layer)
            s = quest_layer_scores(keys, kpos, log.win_q[layer][:, -self.window:], log.groups, self.chunk)
            out.append(dense_columns(s, kpos.expand(s.shape[0], -1), pos))
        return torch.stack(out)  # [L, Hkv, N]


TOKEN_PAGERS = {"streaming": StreamingLLMPager, "h2o": H2OPager, "tova": TOVAPager, "snapkv": SnapKVPager,
                "quest_token": QuestPager}


def make_token_log(method: str, model, budget, strict: bool = False, window: int = 32, record_keep: bool = False):
    """The ``PagedKV`` variant a token pager needs."""
    from ..hf.logstore import PagedKV
    from ..hf.token_probe import ProbedPagedKV

    if method == "streaming":
        return PagedKV(model, budget=budget, strict=strict, record_keep=record_keep)
    return ProbedPagedKV(model, budget=budget, strict=strict, window=window, accumulate=method == "h2o",
                         record_keep=record_keep)

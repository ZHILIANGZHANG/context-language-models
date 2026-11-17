"""Attention masks that reproduce self-paging in one forward pass.

Rule (docs/algorithm_design.md §4.1): query token i, processed under resident-set version s(i),
may attend key token j iff

    j <= i  and  (page(j) == page(i)  or  page(j) in R_{s(i)})

Because every token's KV is computed once and never recomputed, this mask over the whole log
gives exactly the same logits as running the paged engine step by step (P3).
"""

from __future__ import annotations

import torch


def resident_matrix(snapshots: list[set[int]], n_pages: int) -> torch.Tensor:
    """Bool [S, P]: resident[s, p] is True when page p is resident in version s."""
    res = torch.zeros(len(snapshots), n_pages, dtype=torch.bool)
    for s, pids in enumerate(snapshots):
        if pids:
            res[s, list(pids)] = True
    return res


def visibility(page_of: torch.Tensor, seg_of: torch.Tensor, resident: torch.Tensor) -> torch.Tensor:
    """Bool [T, T] visibility matrix (True = may attend)."""
    t = page_of.shape[0]
    q = torch.arange(t)[:, None]
    k = torch.arange(t)[None, :]
    causal = k <= q
    own = page_of[:, None] == page_of[None, :]
    vis = resident[seg_of[:, None], page_of[None, :]]
    return causal & (own | vis)


def additive_mask(vis: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """[1, 1, Q, K] additive float mask (0 = attend, dtype min = blocked), as HF eager/sdpa expect."""
    out = torch.zeros(vis.shape, dtype=dtype)
    out.masked_fill_(~vis, torch.finfo(dtype).min)
    return out[None, None]


def make_mask_mod(page_of: torch.Tensor, seg_of: torch.Tensor, resident: torch.Tensor,
                  doc_of: torch.Tensor | None = None):
    """FlexAttention ``mask_mod`` for training. With packing, page and version ids must be global
    across the packed trajectories and ``doc_of`` gives each token's trajectory."""

    def mask_mod(b, h, q, kv):
        ok = (kv <= q) & ((page_of[q] == page_of[kv]) | resident[seg_of[q], page_of[kv]])
        if doc_of is not None:
            ok = ok & (doc_of[q] == doc_of[kv])
        return ok

    return mask_mod

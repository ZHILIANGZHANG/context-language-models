"""Pack episode records into one training sequence and compute token log-probs in one pass.

Packing (docs/experiment_plan.md §5): trajectories are concatenated; page and version ids are made
global; the resident matrix becomes block-diagonal; ``doc_of`` keeps attention inside a trajectory;
positions restart at 0 for every trajectory (the same global positions the rollout used).

``attn="sdpa"``/``"eager"`` builds a dense additive mask (CPU tests, short sequences);
``attn="flex_attention"`` builds a FlexAttention BlockMask (GPU training). HF's flex_attention
implementation accepts the BlockMask directly as ``attention_mask`` (tested).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch
from torch.utils.checkpoint import checkpoint

from ..hf.masks import additive_mask, make_mask_mod, visibility


@dataclass
class Packed:
    input_ids: torch.Tensor     # [1, T]
    position_ids: torch.Tensor  # [1, T]
    page_of: torch.Tensor       # [T] global page ids
    seg_of: torch.Tensor        # [T] global version ids
    doc_of: torch.Tensor        # [T]
    resident: torch.Tensor      # [S_total, P_total] bool
    loss_mask: torch.Tensor     # [T] 1 where the token is a training target
    ctrl_mask: torch.Tensor     # [T]
    old_logprobs: torch.Tensor  # [T] behaviour log-probs (0 where not generated)
    adv: torch.Tensor           # [T] per-token advantage (RL); zeros for SFT
    n_docs: int


def pack(records: list[dict], advantages: list[list[float]] | None = None) -> Packed:
    ids, pos, page, seg, doc, lm, cm, olp, adv = [], [], [], [], [], [], [], [], []
    blocks = []
    p_off = s_off = 0
    for d, r in enumerate(records):
        if r.get("hidden_pages"):
            # token-level system baselines (agent/token_pagers.py) hide tokens per layer and head; the
            # page-level BlockMask below would let the trainer see them (docs/baseline_protocol.md §7, D12)
            raise ValueError("records with a token-level retention mask cannot be packed for training yet")
        n = len(r["tokens"])
        ids += r["tokens"]
        pos += list(range(n))
        page += [p + p_off for p in r["page_of"]]
        seg += [s + s_off for s in r["seg_of"]]
        doc += [d] * n
        lm += r["loss_mask"]
        cm += r.get("ctrl_mask", [0] * n)
        olp += r.get("logprobs", [0.0] * n)
        adv += advantages[d] if advantages is not None else [0.0] * n
        n_pages = len(r["pages"])
        res = torch.zeros(len(r["snapshots"]), n_pages, dtype=torch.bool)
        for s, pids in enumerate(r["snapshots"]):
            if pids:
                res[s, pids] = True
        blocks.append((s_off, p_off, res))
        p_off += n_pages
        s_off += len(r["snapshots"])
    resident = torch.zeros(s_off, p_off, dtype=torch.bool)
    for so, po, res in blocks:
        resident[so:so + res.shape[0], po:po + res.shape[1]] = res
    t = lambda x, dt=torch.long: torch.tensor(x, dtype=dt)  # noqa: E731
    return Packed(t(ids)[None], t(pos)[None], t(page), t(seg), t(doc), resident, t(lm), t(cm),
                  t(olp, torch.float32), t(adv, torch.float32), len(records))


@lru_cache(maxsize=1)
def _compiled_block_mask_builder():
    from torch.nn.attention.flex_attention import create_block_mask

    # Eager vmap indexing materializes int64 [L,L] temporaries (~8 GiB
    # at 32K). Fuse the identical predicate and block reductions instead.
    # Fullgraph prevents a silent fallback to the memory-heavy eager path.
    return torch.compile(create_block_mask, fullgraph=True, dynamic=True)


def attention_mask(p: Packed, attn: str, dtype: torch.dtype, device):
    if attn == "flex_attention":
        from torch.nn.attention.flex_attention import create_block_mask

        mod = make_mask_mod(p.page_of.to(device), p.seg_of.to(device), p.resident.to(device),
                            p.doc_of.to(device))
        t = p.input_ids.shape[1]
        builder = (_compiled_block_mask_builder() if torch.device(device).type == 'cuda' else create_block_mask)
        return builder(mod, None, None, t, t, device=device)
    vis = visibility(p.page_of, p.seg_of, p.resident) & (p.doc_of[:, None] == p.doc_of[None, :])
    return additive_mask(vis, dtype).to(device)


def token_logprobs(model, p: Packed, attn: str = "sdpa", chunk: int = 2048) -> torch.Tensor:
    """log π(token_t | prefix, mask) for every t ≥ 1 (index t holds the log-prob of token t; index 0
    is 0). The LM head and log-softmax run in checkpointed chunks to keep memory linear in T."""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    mask = attention_mask(p, attn, dtype, device)
    ids = p.input_ids.to(device)
    base = model.get_base_model() if hasattr(model, "get_base_model") else model  # PEFT
    # Torch 2.11 AUTO has no eligible custom-mask kernel for short sequences
    # on SM86. The tested Triton backend supports both short and long blocks;
    # this chooses a kernel, without changing visibility or attention math.
    # SM86 has 99 KiB shared memory per block. Default head_dim=128 kernels
    # request 114 KiB; use verified 32x32 tiles in forward and backward.
    kernel = {"kernel_options": {"BACKEND": "TRITON", "BLOCK_M": 32, "BLOCK_N": 32,
                                  "BLOCK_M1": 32, "BLOCK_N1": 32,
                                  "BLOCK_M2": 32, "BLOCK_N2": 32}} if attn == "flex_attention" else {}
    hidden = base.model(input_ids=ids, position_ids=p.position_ids.to(device),
                        attention_mask=mask, use_cache=False, **kernel).last_hidden_state[0]
    targets = ids[0, 1:]
    head = base.lm_head

    def piece(h, y):
        return torch.log_softmax(head(h).float(), -1).gather(1, y[:, None])[:, 0]

    outs = []
    for a in range(0, targets.shape[0], chunk):
        h, y = hidden[a:a + chunk], targets[a:a + chunk]
        outs.append(checkpoint(piece, h, y, use_reentrant=False) if torch.is_grad_enabled() else piece(h, y))
    lp = torch.cat(outs) if outs else hidden.new_zeros(0)
    # targets that cross a document boundary are never in the loss (the first token of a doc is a
    # system token, loss_mask 0), so no extra masking is needed here
    return torch.cat([lp.new_zeros(1), lp])

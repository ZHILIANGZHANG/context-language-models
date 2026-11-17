"""Read-only Qwen3 query/attention telemetry for the system-pager controls.

No attention output is replaced. Queries use the actual normalized projection
and global RoPE; page attention mass is computed at the instant of each feed,
not retroactively using a different resident set. HF analytical study only.
"""
from __future__ import annotations

from collections import deque
import math

import torch

from .logstore import PagedKV


class ObservedPagedKV(PagedKV):
    def __init__(self, model, budget=None, strict=True, *, window=16, attention_mass=False):
        super().__init__(model, budget=budget, strict=strict)
        if window < 1:
            raise ValueError('Positive observation window required')
        self.window, self.attention_mass = window, attention_mass
        self.recent_queries = {}
        self.attention_window = deque(maxlen=window)
        base = model.get_base_model() if hasattr(model, 'get_base_model') else model
        self.attention_layers = [layer.self_attn for layer in base.model.layers]
        if len(self.attention_layers) != self.n_layers:
            raise ValueError('Expected one dense attention module per layer')
        if any(getattr(layer, 'sliding_window', None) is not None for layer in self.attention_layers):
            raise ValueError('Telemetry currently requires dense Qwen3 attention')

    @torch.no_grad()
    def feed(self, pid, ids):
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
        queries = {}
        def hook(index):
            def capture(module, args, kwargs):
                h = kwargs.get('hidden_states', args[0] if args else None)
                cos, sin = kwargs['position_embeddings']
                h = h[:, -self.window:]
                q = module.q_norm(module.q_proj(h).view(*h.shape[:-1], -1, module.head_dim)).transpose(1,2)
                q, _ = apply_rotary_pos_emb(q, q, cos[:, -h.shape[1]:], sin[:, -h.shape[1]:])
                queries[index] = q.detach()
            return capture
        handles = [layer.register_forward_pre_hook(hook(i), with_kwargs=True)
                   for i, layer in enumerate(self.attention_layers)]
        start = self.log_len
        try:
            logits = super().feed(pid, ids)
        finally:
            for handle in handles:
                handle.remove()
        if len(queries) != self.n_layers:
            raise RuntimeError('Incomplete per-layer query telemetry')
        for layer, q in queries.items():
            previous = self.recent_queries.get(layer)
            self.recent_queries[layer] = (q if previous is None else torch.cat([previous,q],-2))[:, :, -self.window:]
        if self.attention_mass:
            self._record_attention(queries, start, len(ids))
        return logits

    def _record_attention(self, queries, start, n):
        pages = [p for p in self.table.in_order(self.table.resident()) if self.kv[p]]
        positions = torch.cat([torch.arange(self.table.pages[p].start,self.table.pages[p].end,device=self.device)
                               for p in pages])
        nq = min(n,self.window)
        query_positions = torch.arange(start+n-nq,start+n,device=self.device)
        causal = positions[None,:] <= query_positions[:,None]
        totals = torch.zeros(nq,len(pages),device=self.device,dtype=torch.float32)
        for layer,q in queries.items():
            k = torch.cat([self.kv[p][layer][0].to(self.device) for p in pages],-2)
            groups = q.shape[1]//k.shape[1]
            k = k.repeat_interleave(groups,dim=1)
            scores = torch.matmul(q.float(),k.float().transpose(-1,-2))/math.sqrt(q.shape[-1])
            weights = scores.masked_fill(~causal[None,None],float('-inf')).softmax(-1).mean((0,1))
            offset=0
            for j,p in enumerate(pages):
                length=self.table.pages[p].length
                totals[:,j] += weights[:,offset:offset+length].sum(-1)/self.n_layers
                offset+=length
        for row in totals.cpu().tolist():
            self.attention_window.append(dict(zip(pages,row)))

    @torch.no_grad()
    def keymean_scores(self):
        """Mean dot(q, mean page key), averaged over recent queries/layers/heads.

        This is the runbook's page-key approximation, not token-level Quest.
        Already-rotated keys retain their original global positions.
        """
        result={}
        for p, cache in self.kv.items():
            if not cache or self.table.pages[p].where=='dropped':
                continue
            total=0.0
            for layer,(key,_) in enumerate(cache):
                q=self.recent_queries[layer].float()
                representative=key.float().mean(-2,keepdim=True).to(q.device)
                representative=representative.repeat_interleave(q.shape[1]//representative.shape[1],dim=1)
                total+=float((q*representative).sum(-1).mean()/math.sqrt(q.shape[-1]))/self.n_layers
            result[p]=total
        return result

    def attention_scores(self):
        result={}
        for row in self.attention_window:
            for p,mass in row.items():
                result[p]=result.get(p,0.0)+mass
        return result

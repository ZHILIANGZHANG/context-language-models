"""StreamingLLM / H2O algorithms ported to Qwen3's QK-normalized GQA.

Sources pinned in docs/faithful_baselines.md. This is NOT the old turn-level pager.
No page store, protected headers, or resurrection: only compact live K/V tensors.
StreamingLLM caches unrotated keys and reassigns retained cache positions (pos-shift).
H2O caches global-RoPE keys and accumulates attention on retained positions only.

B denotes peak attended KV slots, including the incoming token. Retain B-1 between
forwards, reserving one slot; H2O heavy/recent split is floor/ceil((B-1)/2).
Forward chunks are bounded by free capacity; once full, update/evict every token.
Scores are FP32; GQA query-head attention is averaged within each KV-head group.
Actual outputs use SDPA for consistency with the native Full/model arms. No latency claim.
"""
from __future__ import annotations
import types
import hashlib
import torch
import torch.nn.functional as F
from ..core.log import TokenLog


def retention_indices(method, scores, length, capacity, heads, device, sink=4):
    if length <= capacity:
        return torch.arange(length, device=device).expand(heads, -1)
    if method == 'streaming':
        assert capacity >= sink
        return torch.cat((torch.arange(sink, device=device),
                          torch.arange(length-(capacity-sink), length, device=device))).expand(heads, -1)
    heavy = capacity // 2
    recent = capacity - heavy
    top = scores[:, :length-recent].topk(heavy, dim=-1).indices.sort(-1).values
    return torch.cat((top, torch.arange(length-recent, length, device=device).expand(heads, -1)), -1)


class FaithfulKV(TokenLog):
    def __init__(self, model, method, budget, record_trace=False):
        if method not in ('streaming', 'h2o') or budget < 8:
            raise ValueError('Expected streaming/h2o and budget >= 8')
        super().__init__(budget=budget, strict=False)
        self.model, self.method, self.capacity = model, method, budget-1
        self.base = model.get_base_model() if hasattr(model, 'get_base_model') else model
        self.layers = [x.self_attn for x in self.base.model.layers]
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.groups = model.config.num_attention_heads // model.config.num_key_value_heads
        self.hkv = model.config.num_key_value_heads
        self.state = [None] * len(self.layers)
        self.originals = []
        self.record_trace, self.trace = record_trace, []
        self.evictions = 0
        self.feed_stats = []
        self.native_mask = None
        self.stream_rope = None
        self.last_logits_only = False
        for layer in self.layers:
            if layer.sliding_window is not None:
                raise ValueError('Only dense Qwen3 supported')
        for i, layer in enumerate(self.layers):
            self.originals.append(layer.forward)
            def forward(module, hidden_states, position_embeddings, attention_mask, past_key_values=None, _i=i, **kwargs):
                return self._attention(_i, module, hidden_states, position_embeddings)
            layer.forward = types.MethodType(forward, layer)

    def close(self):
        for layer, original in zip(self.layers, self.originals):
            layer.forward = original
        self.originals = []
        self.state = [None] * len(self.layers)

    def resident_tokens(self):
        s = self.state[0]
        return 0 if s is None else s[0].shape[-2]

    def page_out(self, *args, **kwargs):
        raise RuntimeError('Original token policies do not use page eviction')

    page_in = page_out
    drop = page_out

    def _attention(self, i, module, hidden, rope):
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, rotate_half
        shape = (*hidden.shape[:-1], -1, module.head_dim)
        q = module.q_norm(module.q_proj(hidden).view(shape)).transpose(1, 2)
        k = module.k_norm(module.k_proj(hidden).view(shape)).transpose(1, 2)
        v = module.v_proj(hidden).view(shape).transpose(1, 2)
        n = hidden.shape[1]
        prev = self.state[i]
        if self.method == 'h2o':
            q, k = apply_rotary_pos_emb(q, k, *rope)
        else:
            cos, sin = rope
            q = q*cos.unsqueeze(1) + rotate_half(q)*sin.unsqueeze(1)
        if prev is not None:
            k, v = torch.cat((prev[0], k), -2), torch.cat((prev[1], v), -2)
        size = k.shape[-2]
        assert size <= self.budget
        pos_new = torch.arange(self.log_len, self.log_len+n, device=self.device).expand(self.hkv, -1)
        positions = pos_new if prev is None else torch.cat((prev[3], pos_new), -1)
        attention_k = k
        if self.method == 'streaming':
            cos, sin = self.stream_rope
            attention_k = k*cos.unsqueeze(1) + rotate_half(k)*sin.unsqueeze(1)
        kk = attention_k.repeat_interleave(self.groups, dim=1)
        vv = v.repeat_interleave(self.groups, dim=1)
        out = F.scaled_dot_product_attention(q, kk, vv, attn_mask=self.native_mask,
                                             dropout_p=0., scale=module.scaling)
        scores = None
        if self.method == 'h2o':
            # Same causal visibility as the output forward, never reconstructed from future keys.
            scores = torch.zeros(self.hkv, size, dtype=torch.float32, device=self.device)
            for start in range(0, n, 32):
                logits = q[:, :, start:start+32].float() @ kk.float().transpose(-1, -2)
                logits *= module.scaling
                logits += self.native_mask[:, :, start:start+32].float()
                attn = logits.softmax(-1)[0].view(self.hkv, self.groups, -1, size).mean(1)
                scores += attn.sum(1)
            if prev is not None:
                scores[:, :-n] += prev[2]
        idx = retention_indices(self.method, scores, size, self.capacity, self.hkv, self.device)
        gi = idx[None, :, :, None].expand(1, self.hkv, idx.shape[-1], k.shape[-1])
        self.state[i] = (k.gather(2, gi), v.gather(2, gi),
                         scores.gather(1, idx) if scores is not None else None,
                         positions.gather(1, idx))
        if i == 0 and size > self.capacity:
            self.evictions += size-self.capacity
        if self.record_trace:
            self.trace.append(dict(layer=i, at=self.log_len+n, positions=self.state[i][3].cpu().tolist()))
        return module.o_proj(out.transpose(1, 2).reshape(*hidden.shape[:-1], -1).contiguous()), None

    @torch.no_grad()
    def feed(self, pid, ids):
        if not ids:
            raise ValueError('Empty feed')
        outputs = []
        offset = 0
        while offset < len(ids):
            previous = self.resident_tokens()
            n = min(len(ids)-offset, 128, self.budget-previous)
            assert n >= 1
            chunk = ids[offset:offset+n]
            self.native_mask = torch.zeros(1, 1, n, previous+n, dtype=self.dtype, device=self.device)
            self.native_mask[:, :, :, previous:] = torch.triu(torch.full((n,n),
                torch.finfo(self.dtype).min, dtype=self.dtype, device=self.device), 1)
            start = previous if self.method == 'streaming' else self.log_len
            position_ids = torch.arange(start, start+n, device=self.device)[None]
            if self.method == 'streaming':
                dummy = torch.empty(1, 1, self.model.config.hidden_size, device=self.device, dtype=self.dtype)
                self.stream_rope = self.base.model.rotary_emb(dummy,
                    torch.arange(previous+n, device=self.device)[None])
            result = self.model(input_ids=torch.tensor([chunk], device=self.device),
                position_ids=position_ids, attention_mask=self.native_mask, use_cache=False)
            self.feed_stats.append([n, previous+n, self.table.pages[pid].kind])
            self.peak_resident = max(self.peak_resident, previous+n)
            self._record(pid, chunk)
            outputs.append(result.logits[0])
            offset += n
        return torch.cat(outputs, 0)

    def record(self):
        result = super().record()
        result.update(cache_algorithm=self.method, cache_evictions=self.evictions,
                      cache_capacity=self.capacity, peak_resident=self.peak_resident,
                      cache_final_state=[dict(layer=i, slots=s[0].shape[-2],
                          positions_sha256=hashlib.sha256(s[3].cpu().numpy().tobytes()).hexdigest(),
                          position_min=int(s[3].min()), position_max=int(s[3].max()),
                          keys_finite=bool(torch.isfinite(s[0]).all()), values_finite=bool(torch.isfinite(s[1]).all()),
                          positions_ordered=bool((s[3][:, 1:]>s[3][:, :-1]).all()))
                          for i,s in enumerate(self.state)],
                      feed_stats=self.feed_stats, retention_trace=self.trace)
        return result

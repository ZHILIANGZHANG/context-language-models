# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""KV-cache-aware inference FLOPs for ATIF-CTX trajectories.

Replay each trajectory's ordered per-turn PROMPTS through a prefix-cache model.
With vLLM ``--enable-prefix-caching`` the KV cache is reused ACROSS completion calls (content-addressed by token blocks), so a turn
only prefills the tokens AFTER the longest prefix already computed on an earlier
turn; appending is ~free, and a compaction/edit that rewrites earlier context
(which shifts every later token's position) breaks the shared prefix and
re-prefills the suffix. We report both:

    cache_aware_prefill = sum_t  new_tokens(M_t after longest cached prefix)
    naive_prefill       = sum_t  tokens(M_t)                 # full prompt every turn
    generation          = sum_t  completion_tokens           # each gen'd once
    FLOPs_linear = 2 * N_body * (prefill + generation)       # QKVO + MLP projections

We ALSO add the O(L^2) attention score/-value term the linear model omits, because
it is exactly the term long context inflates and compaction relieves. Attention
FLOPs depend on the per-turn sequence length L_t (not on token *sums*), so we
accumulate them per turn while walking the prefix-cache trie:

    attn_pairs_naive_t       = L_t^2 / 2                      # full causal attention
    attn_pairs_cacheaware_t  = (L_t^2 - matched_t^2) / 2      # only NEW query rows are
                                                              # recomputed, but they still
                                                              # attend over the cached prefix
    attn_pairs_decode        ~= (gen/turns)*sum_t L_t + gen^2/(2*turns)   # decode attends
                                                              # over the growing context
    FLOPs_attn = 4 * n_layers * hidden * (attn_pairs_prefill + attn_pairs_decode)
    FLOPs = FLOPs_linear + FLOPs_attn

Prefix caching reduces attention less than it reduces the linear term (it skips the
cached QUERY positions but the new suffix still attends over all cached KEYS), so
including attention INCREASES the measured benefit of keeping L_t small (compaction).

Prefill source, in priority order (per trajectory):
  1. server usage ``prompt_tokens`` + ``prompt_tokens_details.cached_tokens``
     (vLLM prefix-cache hits): EXACT, and cache_aware = prompt_tokens -
     cached_tokens directly — no trie needed.
  2. real ``prompt_token_ids``: prefix matching at KV-block
     granularity.
  3. the MODEL's own tokenizer over the reconstructed per-turn context (accurate
     for open models; pass ``tokenizer=<HF path/name/obj>`` or a resolvable
     ``model``).
  4. tiktoken o200k (a proxy, mainly for closed API models whose tokenizer we
     cannot load), then chars/4.
Generation: per-step ``completion_tokens`` -> run total -> tokenizer estimate.

FORMULA NOTE: ``2*N_body*tokens`` counts only the linear projection matmuls
(QKV/O + MLP). The O(L^2) attention score/-value compute is added separately (see
above) whenever the model geometry (n_layers, hidden) is resolvable via
``model_key`` (``ATTN_GEOMETRY``) or explicit ``n_layers``/``hidden_size``; if it is
not, the headline falls back to the linear-only term (which under-states
long-context cost and therefore UNDER-states compaction's benefit). N_body is the
non-embedding text-transformer params (embeddings / vision tower / MTP head
excluded).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, Iterable

from ..agent_trajectory_format.models import Trajectory, segment_context_before

# Non-embedding transformer body params (attention + MLP projections), text-only.
N_BODY_PARAMS: dict[str, float] = {"9b": 6.9196e9, "27b": 24.3532e9}
# Full-attention geometry (number of O(L^2) layers, attn_width). Qwen3.5/3.6 are
# hybrid models: only one layer in four is full attention; the other GatedDeltaNet
# layers are linear in sequence length and already belong in N_BODY_PARAMS. Charging
# all 32/64 layers as quadratic overstates long-context attention by exactly 4x.
# attn_width = num_attention_heads * head_dim (under GQA this need not equal hidden).
ATTN_GEOMETRY: dict[str, tuple[int, int]] = {"9b": (8, 4096), "27b": (16, 6144)}
DEFAULT_BLOCK_SIZE = 16  # vLLM default KV-cache block

_ENCODER: Any = None
_ENCODER_TRIED = False


def _encoder() -> Any:
    global _ENCODER, _ENCODER_TRIED
    if _ENCODER_TRIED:
        return _ENCODER
    _ENCODER_TRIED = True
    try:
        import tiktoken

        _ENCODER = tiktoken.get_encoding("o200k_base")
    except Exception:  # tiktoken missing or vocab download blocked offline
        _ENCODER = None
    return _ENCODER


def _tok_len(text: str) -> int:
    enc = _encoder()
    if enc is None:
        return (len(text) + 3) // 4  # chars/4 fallback
    return len(enc.encode(text, disallowed_special=()))


# --- text token counter resolution -----------------------------------------
# When neither server usage nor real token ids are available we still need a
# per-text token count. Prefer the MODEL's own tokenizer (accurate for open
# models); fall back to tiktoken (a proxy, mainly for closed API models whose
# tokenizer we cannot load) and finally chars/4.
_LITELLM_PROVIDER_PREFIXES = (
    "openai/", "anthropic/", "gemini/", "vertex_ai/", "bedrock/", "azure/",
    "azure_ai/", "cohere/", "mistral/", "together_ai/", "fireworks_ai/",
    "groq/", "deepseek/", "xai/", "openrouter/", "ollama/", "hosted_vllm/",
)
_HF_COUNTER_CACHE: dict[str, Any] = {}  # spec -> count_fn | False (tried, failed)


def _hf_ref_from_model(model: Any) -> str | None:
    """A loadable HF tokenizer reference from a model string: a local path, or an
    ``org/name`` repo id (NOT a litellm ``provider/served-name`` route)."""
    if not isinstance(model, str) or not model:
        return None
    import os

    if os.path.exists(model):
        return model
    low = model.lower()
    if any(low.startswith(p) for p in _LITELLM_PROVIDER_PREFIXES):
        return None  # provider-routed served name, not an HF repo id
    return model if "/" in model else None


def _hf_counter(name_or_path: str):
    """A ``str -> n_tokens`` counter from an HF tokenizer, or ``None`` if
    transformers is missing / the tokenizer can't be loaded (offline)."""
    try:
        from transformers import AutoTokenizer
    except Exception:  # transformers not installed
        return None
    try:
        tok = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)
    except Exception:  # not found / download blocked offline
        return None
    return lambda s: len(tok.encode(s, add_special_tokens=False))


def resolve_text_counter(model: Any = None, tokenizer: Any = None) -> tuple[Any, str]:
    """Return ``(count_fn, source_label)`` for counting a text's tokens. Priority:
    explicit callable/tokenizer -> the model's HF tokenizer (by object, path, or
    repo id) -> tiktoken o200k -> chars/4."""
    if callable(tokenizer):
        return tokenizer, "custom"
    # A str tokenizer is an HF path/repo-id spec, NOT an object. Check it BEFORE
    # the ``hasattr("encode")`` object branch: Python ``str`` also has ``.encode``
    # (bytes encoding), so a string would otherwise wrongly take the object path
    # and crash on the ``add_special_tokens`` kwarg.
    if isinstance(tokenizer, str):
        spec: Any = tokenizer
    elif tokenizer is not None and hasattr(tokenizer, "encode"):
        name = getattr(tokenizer, "name_or_path", "obj")
        return (lambda s: len(tokenizer.encode(s, add_special_tokens=False))), f"hf:{name}"
    else:
        spec = _hf_ref_from_model(model)
    if spec:
        cached = _HF_COUNTER_CACHE.get(spec)
        if cached is None:
            cached = _hf_counter(spec) or False
            _HF_COUNTER_CACHE[spec] = cached
        if cached:
            return cached, f"hf:{spec}"

    return _tok_len, ("tiktoken_o200k" if _encoder() is not None else "approx_chars4")


def _msg_text(m: dict[str, Any]) -> str:
    """Stable per-message identity + tokenizable text (role + content + tool call
    + tool_call_id)."""
    import json

    parts = [str(m.get("role", ""))]
    reasoning = m.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning:
        # Qwen thinking is serialized into the assistant turn by the chat template
        # even when visible `content` is empty. Omitting it undercounts exactly the
        # verbosity this metric is meant to expose and can also create false cache hits.
        parts.append(reasoning)
    c = m.get("content")
    if isinstance(c, str):
        parts.append(c)
    elif c is not None:
        parts.append(json.dumps(c, sort_keys=True, default=str))
    if m.get("tool_calls"):
        parts.append(json.dumps(m["tool_calls"], sort_keys=True, default=str))
    if m.get("tool_call_id"):
        parts.append(str(m["tool_call_id"]))
    return "\n".join(parts)


# A "turn" is a list of cache units ``(key, token_weight)``; a prefix of units is
# a cache hit iff the keys match a previously-seen turn's leading units.
Units = list[tuple[Hashable, int]]


def _units_from_token_ids(ids: list[int], block_size: int) -> Units:
    """Block-aligned units over real token ids (matches vLLM block caching)."""
    return [
        (tuple(ids[i : i + block_size]), len(ids[i : i + block_size]))
        for i in range(0, len(ids), block_size)
    ]


def _units_from_messages(messages: list[dict[str, Any]], count_text: Any = _tok_len) -> Units:
    """Per-message units (key = message text, weight = its token length under
    ``count_text``)."""
    return [((t := _msg_text(m)), count_text(t)) for m in messages]


@dataclass
class PrefillResult:
    n_turns: int = 0
    n_reprefill_turns: int = 0  # turns whose cached prefix < previous full prompt
    naive_prefill_tokens: int = 0
    cache_aware_prefill_tokens: int = 0
    # freeze mode: only inserted tokens are prefilled; survivors keep frozen KV.
    # This is the "no-reprefill" regime where context edits cost only the
    # newly-written tokens, not the suffix after the edit point.
    freeze_prefill_tokens: int = 0
    # O(L^2) attention "pair counts" (causal (query,key) pairs), accumulated per
    # turn so absolute attention FLOPs can be applied later with model geometry.
    # naive = full L_t^2/2 every turn; cache-aware = only the NEW query rows
    # ((L_t^2 - matched_t^2)/2) since cached query positions are not recomputed.
    naive_attn_pairs: float = 0.0
    cache_aware_attn_pairs: float = 0.0
    freeze_attn_pairs: float = 0.0

    @property
    def cache_hit_tokens(self) -> int:
        return self.naive_prefill_tokens - self.cache_aware_prefill_tokens

    @property
    def cache_saved_frac(self) -> float:
        return self.cache_hit_tokens / self.naive_prefill_tokens if self.naive_prefill_tokens else 0.0


def accumulate_prefill(turns: Iterable[Units]) -> PrefillResult:
    """Walk ordered per-turn units through a trie prefix-cache; accumulate the
    naive vs cache-aware vs freeze prefill token counts.

    The freeze mode models a KV-reuse engine that never re-prefills survivors:
    on each turn it diffs the new token sequence against the previous turn and
    only computes KV for genuinely inserted tokens (SequenceMatcher 'insert' /
    'replace' on the new side). Survivors keep their frozen KV."""
    from difflib import SequenceMatcher as SM

    res = PrefillResult()
    root: dict[Hashable, Any] = {}
    prev_total = 0
    prev_keys: list[Hashable] | None = None  # for freeze diff
    for units in turns:
        keys = [k for k, _ in units]
        weights = [w for _, w in units]
        total = sum(weights)
        res.naive_prefill_tokens += total

        # --- prefix-cache (trie) accounting ---
        node = root
        matched_tokens = 0
        for key, w in units:
            nxt = node.get(key)
            if nxt is None:
                break
            node = nxt
            matched_tokens += w
        res.cache_aware_prefill_tokens += total - matched_tokens

        # --- freeze accounting (difflib on consecutive turns) ---
        if prev_keys is None:
            # first turn: everything is fresh
            freeze_inserted = total
            freeze_kept = 0
        else:
            ops = SM(a=prev_keys, b=keys, autojunk=False).get_opcodes()
            freeze_inserted = sum(
                sum(weights[j1:j2])
                for tag, i1, i2, j1, j2 in ops
                if tag in ("insert", "replace")
            )
            freeze_kept = sum(
                sum(weights[j1:j2])
                for tag, i1, i2, j1, j2 in ops
                if tag == "equal"
            )
        res.freeze_prefill_tokens += freeze_inserted

        # attention pairs
        res.naive_attn_pairs += total * total / 2.0
        res.cache_aware_attn_pairs += (total * total - matched_tokens * matched_tokens) / 2.0
        # freeze: inserted tokens attend to the full sequence (including frozen
        # prefix), but survivors do NOT recompute their attention.
        res.freeze_attn_pairs += (total * total - freeze_kept * freeze_kept) / 2.0

        if matched_tokens < prev_total:
            res.n_reprefill_turns += 1
        prev_total = total
        prev_keys = keys
        res.n_turns += 1
        node = root
        for key, _ in units:
            node = node.setdefault(key, {})
    return res


# ---------------------------------------------------------------------------
# Trajectory adapters
# ---------------------------------------------------------------------------
def _agent_output_text(step: Any) -> str:
    import json

    parts: list[str] = []
    if isinstance(step.message, str):
        parts.append(step.message)
    if step.reasoning_content:
        parts.append(step.reasoning_content)
    for tc in step.tool_calls or []:
        args = tc.arguments if isinstance(tc.arguments, str) else json.dumps(tc.arguments or {}, default=str)
        parts.append(f"{tc.function_name} {args}")
    return "\n".join(parts)


def _agent_steps(traj: Trajectory) -> list[tuple[Any, int, Any]]:
    """(segment, index_in_segment, step) for every agent step, in run order."""
    return [(seg, k, st) for seg in traj.segments for k, st in enumerate(seg.steps) if st.source == "agent"]


def _prefill_from_usage(steps: list[tuple[Any, int, Any]]) -> dict[str, Any] | None:
    """EXACT prefill from server usage. ``prompt_tokens`` gives naive; the vLLM
    ``cached_tokens`` (prompt tokens served from the prefix cache) gives cache-aware
    directly as ``prompt_tokens - cached_tokens``. Needs both on every agent step."""
    pt = [st.metrics.prompt_tokens if st.metrics else None for _, _, st in steps]
    ct = [st.metrics.cached_tokens if st.metrics else None for _, _, st in steps]
    if not steps or any(p is None for p in pt) or any(c is None for c in ct):
        return None
    naive = sum(int(p) for p in pt)
    cache_aware = sum(max(int(p) - int(c), 0) for p, c in zip(pt, ct))
    n_reprefill = 0
    prev = 0
    naive_attn_pairs = 0.0
    cache_aware_attn_pairs = 0.0
    for p, c in zip(pt, ct):
        p_i, c_i = int(p), min(int(c), int(p))
        if c_i < prev:
            n_reprefill += 1
        prev = p_i
        naive_attn_pairs += p_i * p_i / 2.0
        cache_aware_attn_pairs += (p_i * p_i - c_i * c_i) / 2.0
    # freeze accounting from usage: diff consecutive prompt_tokens to find inserted
    freeze = 0
    freeze_attn_pairs = 0.0
    prev_pt = None
    for p, c in zip(pt, ct):
        p_i, c_i = int(p), min(int(c), int(p))
        if prev_pt is None:
            inserted = p_i  # first turn: full prefill
        elif c_i < prev_pt:
            # approximate: for edits fall back to cache_aware (conservative upper bound)
            inserted = max(p_i - c_i, 0)  # edit turn: prefill the non-cached part
        else:
            inserted = max(p_i - prev_pt, 0)  # append: only new tokens
        freeze += inserted
        # Survivors keep frozen KV and do not recompute attention; only the inserted
        # rows attend, over the full sequence. Same shape as accumulate_prefill's
        # (total^2 - kept^2)/2, with `inserted` standing in for the difflib diff.
        kept = max(p_i - inserted, 0)
        freeze_attn_pairs += (p_i * p_i - kept * kept) / 2.0
        prev_pt = p_i
    return {"naive": naive, "cache_aware": cache_aware, "freeze": freeze,
            "source": "server_usage_cached",
            "n_reprefill": n_reprefill, "n_turns": len(steps),
            "naive_attn_pairs": naive_attn_pairs, "cache_aware_attn_pairs": cache_aware_attn_pairs,
            "freeze_attn_pairs": freeze_attn_pairs}


def _prefill_from_token_ids(steps: list[tuple[Any, int, Any]], block_size: int) -> dict[str, Any] | None:
    """Cache-aware prefill by block-level prefix matching over real server token
    ids. Needs ``prompt_token_ids`` on every agent step."""
    turns: list[Units] = []
    for _, _, st in steps:
        ids = st.metrics.prompt_token_ids if st.metrics else None
        if not ids:
            return None
        turns.append(_units_from_token_ids(list(ids), block_size))
    pf = accumulate_prefill(turns)
    return {
        "naive": pf.naive_prefill_tokens,
        "cache_aware": pf.cache_aware_prefill_tokens,
        "freeze": pf.freeze_prefill_tokens,
        "source": "server_token_ids",
        "n_reprefill": pf.n_reprefill_turns,
        "n_turns": pf.n_turns,
        "naive_attn_pairs": pf.naive_attn_pairs,
        "cache_aware_attn_pairs": pf.cache_aware_attn_pairs,
        "freeze_attn_pairs": pf.freeze_attn_pairs,
    }


def _prefill_from_text(steps: list[tuple[Any, int, Any]], count_text: Any, source: str) -> dict[str, Any]:
    """Fallback: message-level prefix matching over the reconstructed per-turn
    context, counting tokens with ``count_text`` (the model's tokenizer when
    resolvable, else tiktoken). Used only when no server token info is present."""
    turns = [_units_from_messages(segment_context_before(seg, k), count_text) for seg, k, _ in steps]
    pf = accumulate_prefill(turns)
    return {
        "naive": pf.naive_prefill_tokens,
        "cache_aware": pf.cache_aware_prefill_tokens,
        "freeze": pf.freeze_prefill_tokens,
        "source": source,
        "n_reprefill": pf.n_reprefill_turns,
        "n_turns": pf.n_turns,
        "naive_attn_pairs": pf.naive_attn_pairs,
        "cache_aware_attn_pairs": pf.cache_aware_attn_pairs,
        "freeze_attn_pairs": pf.freeze_attn_pairs,
    }


def _generation_tokens(steps: list[tuple[Any, int, Any]], total: int | None, count_text: Any) -> tuple[int, str]:
    """Prefer per-step ``completion_tokens`` (exact); else the run total from
    usage.json (exact sum); else a ``count_text`` estimate of the agent output."""
    per_step = [st.metrics.completion_tokens if st.metrics else None for _, _, st in steps]
    if steps and all(c is not None for c in per_step):
        return sum(int(c) for c in per_step), "server_usage_per_step"
    if total is not None:
        return int(total), "usage_total"
    return sum(count_text(_agent_output_text(st)) for _, _, st in steps), "estimate_from_output"


def resolve_n_body(
    model_key: str | None = None, n_body: float | None = None, *, required: bool = False
) -> float | None:
    """Resolve the non-embedding param count for the ``2*N`` FLOPs term:
    explicit ``n_body`` -> ``N_BODY_PARAMS[model_key]``. Raises on an unknown
    ``model_key`` (catch typos), and — when ``required`` — on a fully missing
    size. Returns ``None`` only when nothing is given and not required."""
    if n_body is not None:
        return float(n_body)
    if model_key is not None:
        if model_key not in N_BODY_PARAMS:
            raise ValueError(
                f"unknown flops_model_key={model_key!r}; known keys: {sorted(N_BODY_PARAMS)}. "
                "Pass a known key or an explicit flops_n_body=<non-embedding params>."
            )
        return N_BODY_PARAMS[model_key]
    if required:
        raise ValueError(
            "FLOPs model size is required but was not provided. Pass flops_model_key "
            f"(one of {sorted(N_BODY_PARAMS)}) or flops_n_body=<non-embedding params>; "
            "or set cost_metric='usd' for a priced API model, or emit_ctx_trajectory=False."
        )
    return None


def resolve_attn_geometry(
    model_key: str | None = None,
    n_layers: int | None = None,
    hidden_size: int | None = None,
) -> tuple[int, int] | None:
    """(n_layers, hidden_size) for the O(L^2) attention term: explicit values ->
    ``ATTN_GEOMETRY[model_key]`` -> None (caller falls back to linear-only)."""
    if n_layers and hidden_size:
        return int(n_layers), int(hidden_size)
    if model_key and model_key in ATTN_GEOMETRY:
        return ATTN_GEOMETRY[model_key]
    return None


def attention_flops_from_prefill(
    pf: PrefillResult, gen_tokens: int, n_layers: int, hidden_size: int
) -> dict[str, float]:
    """Absolute attention FLOPs from a walked :class:`PrefillResult`.

    Uses the per-turn attention pair counts (score + value, causal) plus a decode
    term. Coefficient ``4*n_layers*hidden`` per pair makes a full length-L prefill
    cost ``2*n_layers*hidden*L^2`` (score and value matmuls, causal-halved),
    matching the O(L^2) attention model. Decode pairs are approximated with the mean
    per-turn generation (``gen/turns``) attending over each turn's prompt length
    (``sum_t L_t == naive_prefill_tokens``); decode attention is cache-independent.
    """
    decode_pairs = 0.0
    if pf.n_turns and gen_tokens:
        g = gen_tokens / pf.n_turns
        decode_pairs = g * pf.naive_prefill_tokens + (gen_tokens * gen_tokens) / (2.0 * pf.n_turns)
    coef = 4.0 * n_layers * hidden_size
    return {
        "naive_attn_flops": coef * (pf.naive_attn_pairs + decode_pairs),
        "cache_aware_attn_flops": coef * (pf.cache_aware_attn_pairs + decode_pairs),
        "decode_attn_pairs": decode_pairs,
    }


def compute_trajectory_flops(
    traj: Trajectory,
    *,
    n_body: float | None = None,
    model_key: str | None = None,
    model: str | None = None,
    tokenizer: Any = None,
    block_size: int = DEFAULT_BLOCK_SIZE,
    completion_tokens_total: int | None = None,
    aux_prefill_tokens: int = 0,
    aux_gen_tokens: int = 0,
    n_aux_calls: int = 0,
    n_layers: int | None = None,
    hidden_size: int | None = None,
) -> dict[str, Any]:
    """KV-cache-aware + naive prefill token counts and (if ``n_body``/``model_key``
    given) FLOPs for one trajectory. Prefill source is picked per trajectory:
    server usage+``cached_tokens`` (exact) -> real ``prompt_token_ids`` (block-level)
    -> the model's own tokenizer (``tokenizer``/``model``) -> tiktoken -> chars/4,
    over the reconstructed context. When the model geometry is resolvable
    (``model_key`` in ``ATTN_GEOMETRY`` or explicit ``n_layers``/``hidden_size``) the
    headline FLOPs INCLUDE the O(L^2) attention term; otherwise they are linear-only.
    Returns a JSON-ready dict.

    ``aux_*`` fold in AUXILIARY LLM calls that are NOT trajectory agent steps — e.g.
    summary/compaction calls a harness makes. Those bill real tokens but do
    not appear as steps, so without this they'd be omitted from the local cost estimate.
    Each is a one-off full prefill (a fresh, largely cache-reset prompt), so its input is
    added to BOTH naive and cache-aware prefill, and its output to generation."""
    n_body = resolve_n_body(model_key, n_body)
    count_text, text_source = resolve_text_counter(model=model, tokenizer=tokenizer)
    steps = _agent_steps(traj)
    prefill = (
        _prefill_from_usage(steps)
        or _prefill_from_token_ids(steps, block_size)
        or _prefill_from_text(steps, count_text, text_source)
    )
    generation, gen_source = _generation_tokens(steps, completion_tokens_total, count_text)

    naive, cached = prefill["naive"], prefill["cache_aware"]
    freeze = prefill.get("freeze", cached)  # fallback to cache_aware if freeze not computed
    aux_prefill_tokens = int(aux_prefill_tokens or 0)
    aux_gen_tokens = int(aux_gen_tokens or 0)
    # Auxiliary (summary/compaction) calls: one-off full prefills -> add to both naive and
    # cache-aware (no cache hit for their fresh prompt); their output adds to generation.
    naive += aux_prefill_tokens
    cached += aux_prefill_tokens
    freeze += aux_prefill_tokens
    generation += aux_gen_tokens
    cache_hit = naive - cached
    geom = resolve_attn_geometry(model_key, n_layers, hidden_size)
    # attention FLOPs (added when geometry is known and we have per-turn pair counts)
    attn_ca = attn_nv = attn_fr = 0.0
    if geom and prefill.get("naive_attn_pairs") is not None:
        nl, hd = geom
        pf = PrefillResult(
            n_turns=int(prefill.get("n_turns") or 0),
            naive_prefill_tokens=naive,
            cache_aware_prefill_tokens=cached,
            freeze_prefill_tokens=freeze,
            naive_attn_pairs=float(prefill["naive_attn_pairs"]),
            cache_aware_attn_pairs=float(prefill["cache_aware_attn_pairs"]),
            freeze_attn_pairs=float(prefill.get("freeze_attn_pairs", prefill["cache_aware_attn_pairs"])),
        )
        af = attention_flops_from_prefill(pf, generation, nl, hd)
        attn_nv, attn_ca = af["naive_attn_flops"], af["cache_aware_attn_flops"]
        # freeze attention: same formula but with freeze pairs. Decode attention is
        # cache-independent, so it is added here exactly as for naive/cache-aware.
        attn_fr = 4.0 * nl * hd * (pf.freeze_attn_pairs + af["decode_attn_pairs"])
    includes_attn = bool(geom) and (attn_ca or attn_nv)
    out: dict[str, Any] = {
        "formula": (
            "2*N_body*(prefill_tokens+completion_tokens) + 4*n_layers*hidden*attn_pairs"
            if includes_attn else "2*N_body*(prefill_tokens + completion_tokens)"
        ),
        "includes_attention": bool(includes_attn),
        "prefill_source": prefill["source"],
        "generation_source": gen_source,
        "block_size": block_size,
        "n_llm_calls": len(steps),
        "n_aux_llm_calls": int(n_aux_calls or 0),
        "aux_prefill_tokens": aux_prefill_tokens,
        "aux_gen_tokens": aux_gen_tokens,
        "n_reprefill_turns": prefill.get("n_reprefill"),
        "completion_tokens": generation,
        "cache_aware_prefill_tokens": cached,
        "freeze_prefill_tokens": freeze,
        "naive_prefill_tokens": naive,
        "cache_hit_tokens": cache_hit,
        "cache_saved_frac": round(cache_hit / naive, 4) if naive else 0.0,
        "freeze_saved_frac": round((naive - freeze) / naive, 4) if naive else 0.0,
        "n_body_params": n_body,
        "attn_geometry": ({"n_layers": geom[0], "hidden_size": geom[1]} if geom else None),
        "cache_aware_attn_flops": attn_ca or None,
        "freeze_attn_flops": attn_fr or None,
        "naive_attn_flops": attn_nv or None,
        # linear-only (projections), kept for decomposition/back-compat
        "cache_aware_flops_linear": None,
        "freeze_flops_linear": None,
        "naive_flops_linear": None,
        # headline: linear + attention when geometry is known, else linear-only
        "cache_aware_flops": None,
        "freeze_flops": None,
        "naive_flops": None,
    }
    if n_body:
        lin_ca = 2.0 * n_body * (cached + generation)
        lin_fr = 2.0 * n_body * (freeze + generation)
        lin_nv = 2.0 * n_body * (naive + generation)
        out["cache_aware_flops_linear"] = lin_ca
        out["freeze_flops_linear"] = lin_fr
        out["naive_flops_linear"] = lin_nv
        out["cache_aware_flops"] = lin_ca + attn_ca
        out["freeze_flops"] = lin_fr + attn_fr
        out["naive_flops"] = lin_nv + attn_nv
    return out


def attach_flops(
    traj: Trajectory,
    *,
    n_body: float | None = None,
    model_key: str | None = None,
    model: str | None = None,
    tokenizer: Any = None,
    block_size: int = DEFAULT_BLOCK_SIZE,
    completion_tokens_total: int | None = None,
    aux_prefill_tokens: int = 0,
    aux_gen_tokens: int = 0,
    n_aux_calls: int = 0,
    key: str = "kv_cache_flops",
) -> dict[str, Any]:
    """Compute FLOPs and stash them under ``trajectory.final_metrics.extra[key]``
    (creating ``final_metrics``/``extra`` as needed). Returns the metric dict."""
    from ..agent_trajectory_format.models import FinalMetrics

    result = compute_trajectory_flops(
        traj, n_body=n_body, model_key=model_key, model=model, tokenizer=tokenizer,
        block_size=block_size, completion_tokens_total=completion_tokens_total,
        aux_prefill_tokens=aux_prefill_tokens, aux_gen_tokens=aux_gen_tokens,
        n_aux_calls=n_aux_calls,
    )
    if traj.final_metrics is None:
        traj.final_metrics = FinalMetrics()
    if traj.final_metrics.extra is None:
        traj.final_metrics.extra = {}
    traj.final_metrics.extra[key] = result
    return result


# ---------------------------------------------------------------------------
# Unified compute-cost headline: USD for priced API (closed) models, FLOPs for
# open-weight models. One consistent field, both numbers kept.
# ---------------------------------------------------------------------------
def classify_model_kind(cost_usd: float | None, cost_metric: str = "auto") -> str:
    """"api" (closed, priced) vs "open" (open-weight). ``cost_metric`` forces it
    ("usd"/"api" or "flops"/"open"); "auto" treats a positive litellm-priced
    ``cost_usd`` as an API model and everything else (served/open) as open."""
    m = (cost_metric or "auto").lower()
    if m in ("usd", "api", "closed", "cost"):
        return "api"
    if m in ("flops", "open", "open_weights"):
        return "open"
    return "api" if (cost_usd is not None and cost_usd > 0) else "open"


def estimate_usd(prefill_tokens: float | None, gen_tokens: float | None,
                 in_rate: float | None, out_rate: float | None) -> float | None:
    """USD = prefill_tokens*in_rate + gen_tokens*out_rate, or None if rates unknown."""
    if in_rate is None or out_rate is None or prefill_tokens is None:
        return None
    return float(prefill_tokens) * in_rate + float(gen_tokens or 0) * out_rate


def summarize_compute_cost(
    flops: dict[str, Any],
    *,
    cost_usd: float | None = None,
    model: str | None = None,
    cost_metric: str = "auto",
    in_rate: float | None = None,
    out_rate: float | None = None,
) -> dict[str, Any]:
    """One comparable ``compute_cost`` headline per trajectory, reporting BOTH costs:

    * ``usd_api``  — from the provider's returned usage (the ``cost_usd`` passed in;
      reflects the provider's own prefix-cache billing).
    * ``usd_local_cache_aware`` / ``usd_local_naive`` — a provider-INDEPENDENT
      estimate from the LOCAL tokenizer prefix-match (``kv_cache_flops``): re-prefill
      tokens after the longest shared prefix (so an unedited/append turn ~= free, an
      edit re-bills its suffix) vs the full-prompt-every-turn upper bound.

    For open-weight models the headline stays FLOPs; for API models it is USD
    (preferring the provider value, falling back to the local estimate)."""
    kind = classify_model_kind(cost_usd, cost_metric)
    ca_flops = flops.get("cache_aware_flops")
    gen = flops.get("completion_tokens")
    usd_local_ca = estimate_usd(flops.get("cache_aware_prefill_tokens"), gen, in_rate, out_rate)
    usd_local_naive = estimate_usd(flops.get("naive_prefill_tokens"), gen, in_rate, out_rate)
    if kind == "api":
        metric, unit = "usd", "USD"
        value = cost_usd if cost_usd is not None else usd_local_ca
    else:
        metric, unit, value = "flops", "FLOP", ca_flops
    return {
        "model": model,
        "kind": kind,
        "metric": metric,
        "unit": unit,
        "value": value,
        "cost_usd": cost_usd,
        # dual cost reporting: provider-billed vs local prefix-match estimate
        "usd_api": cost_usd,
        "usd_local_cache_aware": usd_local_ca,
        "usd_local_naive": usd_local_naive,
        "rates_usd_per_token": ({"input": in_rate, "output": out_rate} if in_rate is not None else None),
        "cache_aware_prefill_tokens": flops.get("cache_aware_prefill_tokens"),
        "naive_prefill_tokens": flops.get("naive_prefill_tokens"),
        "cache_saved_frac": flops.get("cache_saved_frac"),
        "n_reprefill_turns": flops.get("n_reprefill_turns"),
        "cache_aware_flops": ca_flops,
        "naive_flops": flops.get("naive_flops"),
        "flops_source": flops.get("prefill_source"),
    }


def attach_compute_cost(
    traj: Trajectory,
    *,
    cost_usd: float | None = None,
    model: str | None = None,
    cost_metric: str = "auto",
    n_body: float | None = None,
    model_key: str | None = None,
    in_rate: float | None = None,
    out_rate: float | None = None,
    key: str = "compute_cost",
) -> dict[str, Any]:
    """Attach the unified ``compute_cost`` headline (and set the typed
    ``FinalMetrics.total_cost_usd``). Reuses ``kv_cache_flops`` if already attached,
    else computes it first. Pass ``in_rate``/``out_rate`` (USD/token) to also emit the
    local prefix-match USD estimate alongside the provider-reported one."""
    from ..agent_trajectory_format.models import FinalMetrics

    extra = traj.final_metrics.extra if traj.final_metrics else None
    flops = (extra or {}).get("kv_cache_flops")
    if flops is None:
        flops = attach_flops(traj, n_body=n_body, model_key=model_key)
    summary = summarize_compute_cost(
        flops, cost_usd=cost_usd, model=model, cost_metric=cost_metric,
        in_rate=in_rate, out_rate=out_rate,
    )
    if traj.final_metrics is None:
        traj.final_metrics = FinalMetrics()
    if traj.final_metrics.extra is None:
        traj.final_metrics.extra = {}
    traj.final_metrics.extra[key] = summary
    if cost_usd is not None:
        traj.final_metrics.total_cost_usd = cost_usd
    return summary


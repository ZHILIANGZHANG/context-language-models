"""D10 (token-level retention in PagedKV) and D11 (token-level system baselines),
docs/baseline_protocol.md §7. Tiny random Qwen3 on CPU."""

from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from self_paging.agent.token_pagers import (TOKEN_PAGERS, make_token_log, quest_layer_scores,  # noqa: E402
                                            quest_upper_bound, select_keep, snapkv_layer_scores,
                                            streaming_scores, tova_layer_scores)
from self_paging.hf.logstore import PagedKV  # noqa: E402
from self_paging.hf.token_probe import ProbedPagedKV  # noqa: E402

VOCAB = 101
_REF: dict = {}


def _ref_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    """Exact attention with the per-layer, per-head visibility in ``_REF``, chunked over queries
    (the logs of the toy-tokenizer episodes are ~7K tokens: no T×T float masks)."""
    parts, layer = _REF["parts"], module.layer_idx
    groups = query.shape[1] // key.shape[1]
    k = key.repeat_interleave(groups, 1)[0]
    v = value.repeat_interleave(groups, 1)[0]
    q = query[0]
    t = q.shape[1]
    out = torch.empty_like(q)
    weights = torch.zeros(q.shape[0], t, t, dtype=q.dtype) if _REF["weights"] else None
    for a in range(0, t, 512):
        rows = torch.arange(a, min(t, a + 512))
        vis = PagedKV.visibility_rows(parts, layer, rows).repeat_interleave(groups, 0)
        s = (q[:, rows] @ k.transpose(-1, -2)) * scaling
        w = s.masked_fill(~vis, float("-inf")).softmax(-1)
        out[:, rows] = w @ v
        if weights is not None:
            weights[:, rows] = w
    return out.transpose(0, 1)[None].contiguous(), (None if weights is None else weights[None])


transformers.AttentionInterface.register("token_vis_ref", _ref_attention)


def tiny(attn="eager", dtype=torch.float64, vocab=VOCAB, layers=3):
    cfg = transformers.Qwen3Config(vocab_size=vocab, hidden_size=64, intermediate_size=128, num_hidden_layers=layers,
                                   num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                                   max_position_embeddings=1 << 15, rope_theta=10000.0, tie_word_embeddings=False)
    torch.manual_seed(0)
    model = transformers.Qwen3ForCausalLM(cfg).to(dtype).eval()
    model.set_attn_implementation(attn)
    return model


def reference(model, kv, output_attentions=False):
    """One forward over the whole log with the per-layer, per-head visibility of ``kv``."""
    _REF.update(parts=kv.visibility_parts(), weights=output_attentions)
    t = len(kv.tokens)
    old = model.config._attn_implementation
    model.set_attn_implementation("token_vis_ref")
    try:
        with torch.no_grad():
            out = model(input_ids=torch.tensor([kv.tokens]), position_ids=torch.arange(t)[None],
                        attention_mask=torch.zeros(1, 1, 1, 1), use_cache=False,
                        output_attentions=output_attentions)
    finally:
        model.set_attn_implementation(old)
        _REF.clear()
    return out


def random_keep(kv: PagedKV, rng: random.Random, frac: float) -> torch.Tensor:
    """Random mask: every (layer, head) keeps the same number of visible tokens, chosen
    independently per head; the page being written is kept."""
    keep = kv.keep_view().clone()
    pos = kv.visible_positions()
    cur = kv.table.pages[kv.table.current]
    free = [i for i, p in enumerate(pos.tolist()) if not cur.start <= p < cur.end and keep[0, 0, p]]
    n_hide = int(len(free) * frac)
    for layer in range(keep.shape[0]):
        for head in range(keep.shape[1]):
            row_free = [i for i in range(len(pos)) if keep[layer, head, pos[i]]
                        and not cur.start <= int(pos[i]) < cur.end]
            for i in rng.sample(row_free, min(n_hide, len(row_free))):
                keep[layer, head, pos[i]] = False
    return keep


def token_episode(kv: PagedKV, seed: int, page_moves: bool = True) -> list[torch.Tensor]:
    rng = random.Random(seed)
    logits = []

    def feed(pid, n, one=False):
        ids = [rng.randrange(VOCAB) for _ in range(n)]
        if one:
            for t in ids:
                logits.append(kv.feed(pid, [t]))
        else:
            logits.append(kv.feed(pid, ids))

    feed(kv.open("system", pinned=True), 6)
    for _ in range(7):
        feed(kv.open("tool"), rng.randint(3, 8))
        feed(kv.open("assistant"), rng.randint(2, 4), one=True)
        if rng.random() < 0.8:
            kv.set_keep(random_keep(kv, rng, rng.uniform(0.1, 0.4)))
        if page_moves:
            movable = [p for p, pg in kv.table.pages.items()
                       if pg.where == "gpu" and not pg.pinned and p != kv.table.current]
            if movable and rng.random() < 0.4:
                kv.page_out(rng.sample(movable, 1))
            host = [p for p, pg in kv.table.pages.items() if pg.where == "host"]
            if host and rng.random() < 0.4:
                kv.page_in(rng.sample(host, 1))
    return logits


# -- D10 ---------------------------------------------------------------------------------------
def test_all_true_mask_is_bit_identical():
    model = tiny("sdpa")
    a, b = PagedKV(model), PagedKV(model)
    rng = random.Random(0)
    for kv in (a, b):
        kv.feed(kv.open("system", pinned=True), [1, 2, 3, 4])
    out = []
    for step in range(5):
        ids = [rng.randrange(VOCAB) for _ in range(4)]
        for kv in (a, b):
            kv.feed(kv.open("tool"), ids)
        b.set_keep(b.keep_view().clone())  # installs an all-true mask: a new version, same visibility
        out.append((a.feed(a.open("assistant"), [7]), b.feed(b.open("assistant"), [7])))
    assert all(torch.equal(x, y) for x, y in out)
    assert b.keep_changes == 5 and b.resident_tokens() == a.resident_tokens()


@pytest.mark.parametrize("attn", ["eager", "sdpa"])
@pytest.mark.parametrize("seed", [0, 1])
def test_token_mask_equals_masked_single_pass(attn, seed):
    model = tiny(attn)
    kv = PagedKV(model, record_keep=True)
    stepwise = torch.cat(token_episode(kv, seed), 0)
    assert kv.keep_changes > 2
    vis = kv.token_visibility()
    assert not torch.equal(vis[:, 0], vis[:, 1]) or not torch.equal(vis[0], vis[-1])  # per-head / per-layer
    full = reference(model, kv).logits[0]
    torch.testing.assert_close(stepwise, full, rtol=0, atol=1e-6)


def test_token_mask_matters_and_budget_counts_kept_tokens():
    model = tiny()
    kv = PagedKV(model, record_keep=True)
    token_episode(kv, 3, page_moves=False)
    t = len(kv.tokens)
    with torch.no_grad():
        plain = model(input_ids=torch.tensor([kv.tokens]), position_ids=torch.arange(t)[None]).logits[0]
    assert (reference(model, kv).logits[0] - plain).abs().max() > 1e-4
    kept = kv.keep_view()[:, :, kv.visible_positions()].sum(-1)
    assert (kept == kept.flatten()[0]).all() and kv.resident_tokens() == int(kept.flatten()[0])
    assert kv.resident_tokens() < kv.table.resident_tokens()


def test_set_keep_validates():
    kv = PagedKV(tiny())
    kv.feed(kv.open("system", pinned=True), [1, 2, 3, 4])
    kv.feed(kv.open("tool"), [5, 6, 7])
    keep = kv.keep_view().clone()
    keep[0, 0, 1] = False
    with pytest.raises(ValueError, match="same number"):
        kv.set_keep(keep)
    keep = kv.keep_view().clone()
    keep[:, :, 5] = False  # inside the page being written
    with pytest.raises(ValueError, match="being written"):
        kv.set_keep(keep)


def test_equalize_after_page_move():
    model = tiny()
    kv = PagedKV(model, record_keep=True)
    kv.feed(kv.open("system", pinned=True), [1, 2, 3])
    a = kv.open("tool")
    kv.feed(a, [4, 5, 6, 7])
    b = kv.open("tool")
    kv.feed(b, [8, 9, 10, 11])
    kv.feed(kv.open("assistant"), [12])
    keep = kv.keep_view().clone()
    keep[:, 0, 4] = False  # head 0 hides a token of page a ...
    keep[:, 1, 8] = False  # ... head 1 one of page b
    kv.set_keep(keep)
    kv.page_out([a])  # head 1 now keeps one more resident token than head 0
    kv.feed(kv.open("assistant"), [13])
    assert kv.keep_equalized == 1
    kept = kv.keep_view()[:, :, kv.visible_positions()].sum(-1)
    assert (kept == kept.flatten()[0]).all()
    stepwise = kv.feed(kv.open("assistant"), [14])
    torch.testing.assert_close(stepwise, reference(model, kv).logits[0, -1:], rtol=0, atol=1e-6)


def test_h2o_accumulation_equals_reference_attention():
    model = tiny(layers=2)
    kv = ProbedPagedKV(model, window=4, accumulate=True, record_keep=True)
    token_episode(kv, 5, page_moves=False)
    out = reference(model, kv, output_attentions=True)
    groups = model.config.num_attention_heads // model.config.num_key_value_heads
    for layer, att in enumerate(out.attentions):
        expect = att[0].sum(1).view(-1, groups, att.shape[-1]).sum(1).float()  # [Hkv, T]
        torch.testing.assert_close(kv.acc[layer, :, :expect.shape[-1]], expect, rtol=1e-5, atol=1e-5)


def test_window_queries_reproduce_last_row_attention():
    model = tiny(layers=2)
    kv = ProbedPagedKV(model, window=3, record_keep=True)
    token_episode(kv, 6, page_moves=False)
    kv.feed(kv.open("assistant"), [5])  # the episode may end with a mask change; query under the current set
    out = reference(model, kv, output_attentions=True)
    for layer in range(2):
        attn, cols = kv.window_attention(layer)  # visible now = what the last query saw
        last_ref = out.attentions[layer][0, :, -1]  # [Hq, T]
        groups = attn.shape[0] // cols.shape[0]
        got = torch.zeros_like(last_ref, dtype=torch.float32)
        got.scatter_(1, cols.repeat_interleave(groups, 0), attn[:, -1])
        torch.testing.assert_close(got, last_ref.float(), rtol=1e-5, atol=1e-5)


# -- D11: scorers on hand-made tensors -----------------------------------------------------------
def test_select_keep_protects_and_respects_irreversibility():
    keep_now = torch.ones(1, 2, 6, dtype=torch.bool)
    keep_now[0, :, 1] = False
    scores = torch.tensor([[[0., 9., 1., 5., 2., 3.], [4., 9., 3., 0., 1., 2.]]])
    protect = torch.tensor([True, False, False, False, False, False])
    sel = select_keep(scores, keep_now, protect, 3, reversible=False)
    assert sel[0, 0].tolist() == [True, False, False, True, False, True]
    assert sel[0, 1].tolist() == [True, False, True, False, False, False] or sel[0, 1].sum() == 3
    assert (sel.sum(-1) == 3).all() and not sel[:, :, 1].any()  # the hidden token never comes back
    rev = select_keep(scores, keep_now, protect, 3, reversible=True)
    assert rev[:, :, 1].all()  # reversible methods may reselect it


def test_streaming_keeps_sinks_and_recent():
    pos = torch.arange(10)
    sel = select_keep(streaming_scores(pos, 2), torch.ones(1, 1, 10, dtype=torch.bool),
                      torch.zeros(10, dtype=torch.bool), 5, False)
    assert sel[0, 0].nonzero().flatten().tolist() == [0, 1, 7, 8, 9]


def test_tova_keeps_what_the_last_query_attends():
    attn = torch.full((4, 6), 0.1)
    attn[:, 2] = 0.8
    attn[0, 4] = 0.9  # one head only: averaging over heads keeps column 2 first
    pos = torch.arange(6)
    s = tova_layer_scores(attn, pos.expand(2, -1), pos, groups=2)
    assert s.shape == (1, 6) and int(s.argmax()) == 2 and s[0, 4] > s[0, 0]


def test_snapkv_pooling_keeps_neighbours_of_a_vote():
    w = torch.zeros(2, 3, 9)  # [Hq, window, n]
    w[:, :, 4] = 1.0
    w[:, :, 0] = 0.4
    pos = torch.arange(9)
    flat = snapkv_layer_scores(w, pos.expand(1, -1), pos, groups=2, kernel=1)
    pooled = snapkv_layer_scores(w, pos.expand(1, -1), pos, groups=2, kernel=3)
    top = lambda s: set(s[0].topk(3).indices.tolist())  # noqa: E731
    assert 0 in top(flat) and top(pooled) == {3, 4, 5}


def test_quest_bound_is_an_upper_bound_and_finds_the_chunk():
    torch.manual_seed(0)
    q = torch.randn(2, 1, 8)
    keys = 0.1 * torch.randn(1, 64, 8)
    target = 37
    keys[0, target] = 4 * q[0, 0] / q[0, 0].norm()  # chunk 2 (positions 32-47) holds an aligned key
    pos = torch.arange(64)
    kmin = keys.view(1, 4, 16, 8).amin(2)
    kmax = keys.view(1, 4, 16, 8).amax(2)
    ub = quest_upper_bound(q, kmin.expand(2, -1, -1), kmax.expand(2, -1, -1))  # [Hq, 1, C]
    exact = torch.einsum("hwd,cnd->hwcn", q, keys.view(4, 16, 8)).amax(-1)
    assert (ub >= exact - 1e-5).all()
    s = quest_layer_scores(keys, pos, q[:1], groups=1, chunk=16)
    assert int(s[0].argmax()) // 16 == 2 and torch.equal(s[0, 32:48], s[0, 32].expand(16))


# -- D11: pagers in episodes -------------------------------------------------------------------------
def _episode(method, model, tok, ratio=0.5, record_keep=True, **kw):
    from self_paging.agent.episode import EpisodeConfig, ScriptedPolicy, run_episode
    from self_paging.agent.pagers import rule_task_policy
    from self_paging.data.build_t1_sft import t1_budget
    from self_paging.envs.t1_warehouse import T1Config, WarehouseEnv

    env = WarehouseEnv(10000, T1Config(n_decisions=1, warmup=2, gap=(1, 2), events_per_page=(2, 3)))
    budget = t1_budget(env, tok, ratio)
    log = make_token_log(method, model, budget, window=8, record_keep=record_keep)
    rec = run_episode(env, log, ScriptedPolicy(tok, rule_task_policy), tok,
                      EpisodeConfig(budget=budget, max_turns=20, kv_tools=False, min_gen_room=16, struct_reserve=32),
                      pager=TOKEN_PAGERS[method](**kw))
    return rec, log


@pytest.fixture(scope="module")
def toy():
    from self_paging.train.common import load_tokenizer

    tok = load_tokenizer("toy")
    return tok, tiny(vocab=tok.vocab_size, layers=2)


@pytest.mark.parametrize("method", list(TOKEN_PAGERS))
def test_token_pagers_hold_the_budget_and_match_single_pass(method, toy):
    tok, model = toy
    rec, log = _episode(method, model, tok)
    m = rec["meta"]
    assert m["token_compressions"] > 0 and not m["over_budget"] and rec["peak_resident"] <= rec["budget"]
    assert m["forced_evictions"] == 0
    hist = [k[:, :, :log.keep_history[0][1].shape[-1]] for _, k in log.keep_history]
    restored = any(((~a) & b).any() for a, b in zip(hist, hist[1:]))
    assert restored == (method == "quest_token")  # only Quest is reversible
    # the logged behaviour log-probs equal one forward with the per-layer, per-head visibility
    logits = reference(model, log).logits[0]
    lp = torch.log_softmax(logits.float(), -1)
    toks = torch.tensor(rec["tokens"])
    idx = [i for i, (l, b) in enumerate(zip(rec["loss_mask"], rec["logprobs"])) if l and i > 0]
    ref = lp[torch.tensor(idx) - 1, toks[idx]]
    torch.testing.assert_close(torch.tensor([rec["logprobs"][i] for i in idx]), ref, rtol=0, atol=1e-5)


def test_token_pagers_choose_different_tokens(toy):
    tok, model = toy
    finals = {}
    for method in TOKEN_PAGERS:
        _, log = _episode(method, model, tok, record_keep=False)
        finals[method] = log.keep_view()[:, :, log.visible_positions()].clone()
    names = list(finals)
    assert all(not torch.equal(finals[a], finals[b]) for i, a in enumerate(names) for b in names[i + 1:])
    # granularity: StreamingLLM shares one selection; TOVA one per layer; SnapKV / H2O differ per head
    assert all(torch.equal(finals["streaming"][0, 0], r) for r in finals["streaming"].flatten(0, 1))
    assert torch.equal(finals["tova"][0, 0], finals["tova"][0, 1])
    assert any(not torch.equal(finals[m][0, 0], finals[m][0, 1]) for m in ("snapkv", "h2o"))


def test_token_sanity_runs_both_modes(toy):
    from self_paging.eval.token_sanity import make_item, run_item

    tok, model = toy
    item = make_item(random.Random(0), tok, 600, 0.3)
    assert item[0] in item[2]
    for mode in ("oneshot", "stream"):
        for method in ("full", *TOKEN_PAGERS):
            r = run_item(model, tok, method, *item, 0.3, 128, mode, max_new=3)
            assert r["tokens"] > 600 and isinstance(r["correct"], bool)
            if method != "full" and mode == "stream":
                assert r["peak_resident"] <= r["budget"]


@pytest.mark.parametrize("method", ["snapkv", "quest_token"])
def test_cost_counts_only_visible_tokens(method, toy):
    from self_paging.core.costs import visible_counts

    tok, model = toy
    rec, log = _episode(method, model, tok)
    parts = log.visibility_parts()
    rows = torch.arange(len(rec["tokens"]))
    expect = PagedKV.visibility_rows(parts, 0, rows)[0].sum(-1)  # layer 0, head 0
    assert visible_counts(rec) == expect.tolist()
    plain = visible_counts({k: v for k, v in rec.items() if k != "hidden_pages"})
    assert sum(plain) > sum(expect.tolist())

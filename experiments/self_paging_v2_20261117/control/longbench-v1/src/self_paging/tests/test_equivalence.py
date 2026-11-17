"""P1–P3 on the real Qwen3 architecture (tiny random weights, CPU, no download).

The core claim of the method: running the paged engine step by step (gather resident KV, global
positions) gives the same logits as one forward over the whole log with the visibility mask.
If this fails, single-pass training would not match inference and the design does not hold.
"""

from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from self_paging.core.pages import PagingError  # noqa: E402
from self_paging.hf.logstore import PagedKV  # noqa: E402
from self_paging.hf.masks import additive_mask, make_mask_mod, visibility  # noqa: E402

VOCAB = 101


def tiny_qwen3(attn: str = "eager", dtype=torch.float64):
    cfg = transformers.Qwen3Config(
        vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=4096,
        rope_theta=10000.0, tie_word_embeddings=False)
    torch.manual_seed(0)
    model = transformers.Qwen3ForCausalLM(cfg).to(dtype).eval()
    model.set_attn_implementation(attn)
    return model


def random_episode(kv: PagedKV, seed: int, physical: bool = False) -> list[torch.Tensor]:
    """System page, then tool pages and assistant pages (decoded one token at a time), with random
    kv_out / kv_in between turns. Returns the logits of every fed token, in log order."""
    rng = random.Random(seed)
    logits = []

    def feed(pid, n, one_by_one=False):
        ids = [rng.randrange(VOCAB) for _ in range(n)]
        if one_by_one:
            for t in ids:
                logits.append(kv.feed(pid, [t]))
        else:
            logits.append(kv.feed(pid, ids))

    feed(kv.open("system", pinned=True), 7)
    for _ in range(8):
        feed(kv.open("tool"), rng.randint(3, 9))
        feed(kv.open("assistant"), rng.randint(2, 5), one_by_one=True)
        movable = [p for p, pg in kv.table.pages.items()
                   if pg.where == "gpu" and not pg.pinned and p != kv.table.current]
        if movable and rng.random() < 0.7:
            kv.page_out(rng.sample(movable, rng.randint(1, len(movable))), physical=physical)
        host = [p for p, pg in kv.table.pages.items() if pg.where == "host"]
        if host and rng.random() < 0.5:
            kv.page_in(rng.sample(host, rng.randint(1, len(host))))
    return logits


def single_pass(model, traj) -> torch.Tensor:
    vis = visibility(traj["page_of"], traj["seg_of"], traj["resident"])
    mask = additive_mask(vis, next(model.parameters()).dtype)
    t = traj["tokens"].shape[0]
    with torch.no_grad():
        out = model(input_ids=traj["tokens"][None], position_ids=torch.arange(t)[None],
                    attention_mask=mask, use_cache=False)
    return out.logits[0]


@pytest.mark.parametrize("attn", ["eager", "sdpa"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_paged_engine_equals_masked_single_pass(attn, seed):
    model = tiny_qwen3(attn)
    kv = PagedKV(model)
    stepwise = torch.cat(random_episode(kv, seed), dim=0)
    traj = kv.trajectory()
    assert traj["resident"].shape[0] > 1, "episode should contain paging actions"
    assert (~visibility(traj["page_of"], traj["seg_of"], traj["resident"])).any()
    full = single_pass(model, traj)
    assert stepwise.shape == full.shape
    # sdpa runs in float64 end to end, so the two paths agree to rounding. HF's eager attention
    # computes softmax in float32 (`softmax(..., dtype=torch.float32)`), so rows with different
    # numbers of masked keys round differently at ~1e-7.
    atol = 1e-9 if attn == "sdpa" else 1e-6
    torch.testing.assert_close(stepwise, full, rtol=0, atol=atol)


def test_masking_matters():
    """Sanity check that the test can fail: ignoring the resident set changes the logits."""
    model = tiny_qwen3()
    kv = PagedKV(model)
    stepwise = torch.cat(random_episode(kv, 3), dim=0)
    traj = kv.trajectory()
    t = traj["tokens"].shape[0]
    with torch.no_grad():
        causal = model(input_ids=traj["tokens"][None], position_ids=torch.arange(t)[None]).logits[0]
    assert (stepwise - causal).abs().max() > 1e-4


def test_physical_paging_is_bit_identical():
    model = tiny_qwen3()
    logical, physical = PagedKV(model), PagedKV(model)
    a = random_episode(logical, 4)
    b = random_episode(physical, 4, physical=True)
    for x, y in zip(a, b):
        assert torch.equal(x, y)
    for p in logical.kv:
        for (k1, v1), (k2, v2) in zip(logical.kv[p], physical.kv[p]):
            assert torch.equal(k1, k2) and torch.equal(v1, v2)


def test_page_in_restores_exact_kv():
    model = tiny_qwen3()
    kv = PagedKV(model)
    kv.feed(kv.open("system", pinned=True), [1, 2, 3])
    obs = kv.open("tool")
    kv.feed(obs, [4, 5, 6, 7])
    before = [(k.clone(), v.clone()) for k, v in kv.kv[obs]]
    kv.feed(kv.open("assistant"), [8])
    kv.page_out([obs], physical=True)
    kv.feed(kv.open("assistant"), [9])
    kv.page_in([obs])
    for (k0, v0), (k1, v1) in zip(before, kv.kv[obs]):
        assert torch.equal(k0, k1) and torch.equal(v0, v1)


def test_float32_tolerance():
    model = tiny_qwen3("sdpa", torch.float32)
    kv = PagedKV(model)
    stepwise = torch.cat(random_episode(kv, 5), dim=0)
    torch.testing.assert_close(stepwise, single_pass(model, kv.trajectory()), rtol=1e-4, atol=1e-4)


def test_flex_mask_mod_matches_dense_visibility():
    from torch.nn.attention.flex_attention import create_mask

    kv = PagedKV(tiny_qwen3())
    random_episode(kv, 6)
    traj = kv.trajectory()
    t = traj["tokens"].shape[0]
    mod = make_mask_mod(traj["page_of"], traj["seg_of"], traj["resident"])
    flex = create_mask(mod, None, None, t, t, device="cpu")[0, 0]
    assert torch.equal(flex, visibility(traj["page_of"], traj["seg_of"], traj["resident"]))


def test_illegal_actions():
    kv = PagedKV(tiny_qwen3(), budget=12)
    sys = kv.open("system", pinned=True)
    kv.feed(sys, [1, 2, 3])
    with pytest.raises(PagingError):
        kv.page_out([sys])  # pinned
    obs = kv.open("tool")
    kv.feed(obs, [4, 5, 6])
    with pytest.raises(PagingError):
        kv.page_out([obs])  # still being written
    ans = kv.open("assistant")
    kv.feed(ans, [7])
    with pytest.raises(PagingError):
        kv.page_in([obs])  # not in host
    with pytest.raises(PagingError):
        kv.feed(ans, [1] * 10)  # over budget
    kv.page_out([obs])
    with pytest.raises(PagingError):
        kv.page_out([obs])  # already out


def test_hf_flex_attention_accepts_block_mask():
    """Training path: HF's own `flex_attention` implementation takes our BlockMask as
    `attention_mask`, so single-pass training needs no change to the model code."""
    from torch.nn.attention.flex_attention import create_block_mask

    dense_model = tiny_qwen3("sdpa", torch.float32)
    kv = PagedKV(dense_model)
    random_episode(kv, 7)
    traj = kv.trajectory()
    t = traj["tokens"].shape[0]
    ref = single_pass(dense_model, traj)
    flex_model = tiny_qwen3("flex_attention", torch.float32)
    block_mask = create_block_mask(make_mask_mod(traj["page_of"], traj["seg_of"], traj["resident"]),
                                   None, None, t, t, device="cpu")
    with torch.no_grad():
        out = flex_model(input_ids=traj["tokens"][None], position_ids=torch.arange(t)[None],
                         attention_mask=block_mask, use_cache=False).logits[0]
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)

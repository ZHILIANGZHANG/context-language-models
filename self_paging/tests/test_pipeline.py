"""End-to-end checks of data → SFT → RL on CPU (tiny random Qwen3, character tokenizer).

The most important one is ``test_rollout_logprobs_match_training_pass``: the log-probs recorded
while rolling out through the paged engine equal the log-probs the trainer computes in one masked
forward. That is P3 applied to the actual RL pipeline (otherwise GRPO's ratios would be wrong).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from self_paging.agent.episode import EpisodeConfig, ModelPolicy, ScriptedPolicy, run_episode  # noqa: E402
from self_paging.agent.pagers import BeladyPager, RecentPager  # noqa: E402
from self_paging.core.costs import episode_cost, visible_counts  # noqa: E402
from self_paging.core.log import TokenLog  # noqa: E402
from self_paging.core.protocol import (Tok, escape, parse_pages, parse_tool_calls,  # noqa: E402
                                       render_tool_call)
from self_paging.data.build_t1_sft import build, t1_budget  # noqa: E402
from self_paging.envs.t1_warehouse import T1Config, WarehouseEnv, generate, rule_read  # noqa: E402
from self_paging.hf.logstore import PagedKV  # noqa: E402
from self_paging.hf.masks import visibility  # noqa: E402
from self_paging.oracle.t1_oracle import T1Oracle  # noqa: E402
from self_paging.testing.toy_tokenizer import ToyTokenizer  # noqa: E402
from self_paging.train.advantage import group_advantages, token_advantages  # noqa: E402
from self_paging.train.batch import pack, token_logprobs  # noqa: E402
from self_paging.train.common import tiny_qwen3  # noqa: E402
from self_paging.train.grpo import GRPOConfig, train_step  # noqa: E402
from self_paging.train.sft import sft_loss, train_sft  # noqa: E402

TOK = Tok(ToyTokenizer())
SMALL = T1Config(n_decisions=2, warmup=4, gap=(2, 3), events_per_page=(2, 3))
MID = T1Config(n_decisions=3, warmup=10, gap=(3, 8), events_per_page=(3, 6))


# -- protocol ---------------------------------------------------------------------------------
def test_protocol_parsing():
    assert parse_pages("3,5-7") == [3, 5, 6, 7]
    assert parse_pages([4, 2, 4]) == [2, 4]
    with pytest.raises(ValueError):
        parse_pages("x")
    text = "hi\n" + render_tool_call("kv_out", {"pages": "1-2"}) + "\n<tool_call>\nnot json\n</tool_call>"
    calls, spans, bad = parse_tool_calls(text)
    assert calls == [{"name": "kv_out", "arguments": {"pages": "1-2"}}] and bad == 1
    assert text[spans[0][0]:spans[0][1]].startswith("<tool_call>")
    assert "[[PAGE" not in escape("x [[PAGE 3 tool=a]] <|im_end|>") and "<|" not in escape("<|im_end|>")


# -- environment ------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(6))
def test_t1_env_rule_reader_and_need_sets(seed):
    ep = generate(seed, MID)
    assert ep.pages == generate(seed, MID).pages, "not deterministic"
    assert ep.decisions, "no decision generated"
    for d in ep.decisions:
        prefix = ep.pages[: d.page + 1]
        assert rule_read(prefix) == d.answer, "rule reader must solve from text alone"
        need = sorted(ep.need[d.page])
        assert rule_read([ep.pages[i] for i in need]) == d.answer, "need set must be sufficient"
        for p in need:
            if p != d.page:
                rest = [ep.pages[i] for i in need if i != p]
                assert rule_read(rest) != d.answer, "need set must be minimal"


def test_env_protocol():
    env = WarehouseEnv(0, SMALL)
    text, meta, done = env.step("store", {"shelf": 1})
    assert text.startswith("ERROR")
    while not env.done:
        text, meta, done = env.step("next_events", {})
        if env.pending is not None:
            assert env.step("next_events", {})[0].startswith("ERROR")
            env.step("store", {"shelf": env.pending.answer})
    assert env.score() == 1.0


# -- oracle data ------------------------------------------------------------------------------
def test_oracle_sft_data():
    kept, stats = build(TOK, range(4), [1.0, 0.6], ["paging", "notes"], MID, max_turns=80, max_future_share=1.0)
    assert kept, stats
    budgeted = [r for r in kept if r["budget"] is not None]
    assert budgeted, stats
    for r in kept:
        n = len(r["tokens"])
        assert len(r["loss_mask"]) == len(r["ctrl_mask"]) == len(r["logprobs"]) == len(r["page_of"]) == n
        assert r["reward"] == 1.0
        if r["budget"] is not None:
            assert r["peak_resident"] <= r["budget"]
    assert any(sum(r["ctrl_mask"]) > 0 for r in budgeted), "budgeted oracle episodes should page"
    assert all(sum(r["ctrl_mask"]) == 0 for r in kept if r["budget"] is None)


def test_costs_match_dense_visibility():
    env = WarehouseEnv(1, SMALL)
    b = t1_budget(env, TOK, 0.6)
    rec = run_episode(env, TokenLog(b), ScriptedPolicy(TOK, T1Oracle("paging")), TOK, EpisodeConfig(budget=b))
    log_vis = visibility(torch.tensor(rec["page_of"]), torch.tensor(rec["seg_of"]),
                         _resident(rec)).sum(1).tolist()
    assert visible_counts(rec) == log_vis
    c = episode_cost(rec, batch=32)
    assert c["cost"] > 0 and c["generated_tokens"] == sum(rec["loss_mask"])


def _resident(rec):
    res = torch.zeros(len(rec["snapshots"]), len(rec["pages"]), dtype=torch.bool)
    for s, pids in enumerate(rec["snapshots"]):
        if pids:
            res[s, pids] = True
    return res


# -- RL consistency ---------------------------------------------------------------------------
def _gen_positions(rec):
    return [i for i, m in enumerate(rec["loss_mask"]) if m]


def test_rollout_logprobs_match_training_pass():
    model = tiny_qwen3(TOK.vocab_size).eval()
    env = WarehouseEnv(2, SMALL)
    b = t1_budget(env, TOK, 0.6)
    # scripted oracle through the paged engine: real paging happens, log-probs come from the engine
    rec = run_episode(env, PagedKV(model, budget=b, strict=False), ScriptedPolicy(TOK, T1Oracle("paging")),
                      TOK, EpisodeConfig(budget=b))
    assert len(rec["snapshots"]) > 2, "episode should contain paging"
    with torch.no_grad():
        lp = token_logprobs(model, pack([rec]), attn="sdpa")
    pos = _gen_positions(rec)
    got = torch.tensor([rec["logprobs"][i] for i in pos])
    torch.testing.assert_close(lp[pos], got, rtol=1e-4, atol=1e-4)


def test_sampled_rollout_logprobs_match():
    model = tiny_qwen3(TOK.vocab_size).eval()
    env = WarehouseEnv(3, SMALL)
    gen = torch.Generator().manual_seed(0)
    rec = run_episode(env, PagedKV(model, budget=None, strict=False), ModelPolicy(TOK, generator=gen), TOK,
                      EpisodeConfig(max_turns=3, max_new_tokens=24))
    with torch.no_grad():
        lp = token_logprobs(model, pack([rec]), attn="sdpa")
    pos = _gen_positions(rec)
    torch.testing.assert_close(lp[pos], torch.tensor([rec["logprobs"][i] for i in pos]), rtol=1e-4, atol=1e-4)


# -- training ---------------------------------------------------------------------------------
def test_sft_overfits_oracle_episodes():
    kept, _ = build(TOK, range(2), [0.6], ["paging"], SMALL, max_turns=60, max_future_share=1.0)
    assert kept
    model = tiny_qwen3(TOK.vocab_size)
    before = sft_loss(model, kept[:1], "sdpa")[0].item()
    train_sft(model, kept[:1], epochs=30, lr=3e-3, max_tokens=10**6, attn="sdpa", warmup_frac=0.0)
    after = sft_loss(model, kept[:1], "sdpa")[0].item()
    assert after < before * 0.8, (before, after)


def test_group_advantages():
    a_task, a_eff = group_advantages([1, 1, 0, 1], [10.0, 20.0, 5.0, 30.0])
    assert a_task == [0.25, 0.25, -0.75, 0.25]
    assert a_eff[2] == 0.0 and a_eff[0] > 0 > a_eff[3]
    _, none = group_advantages([1, 0], [1.0, 2.0])
    assert none == [0.0, 0.0]


def test_grpo_step_moves_policy_towards_positive_advantage():
    model = tiny_qwen3(TOK.vocab_size)
    gen = torch.Generator().manual_seed(1)
    recs = []
    for seed in (4, 5):
        rec = run_episode(WarehouseEnv(seed, SMALL), PagedKV(model.eval(), strict=False),
                          ModelPolicy(TOK, generator=gen), TOK, EpisodeConfig(max_turns=2, max_new_tokens=16))
        rec["cost"] = 1.0
        recs.append(rec)
    recs[0]["reward"], recs[1]["reward"] = 1.0, 0.0
    cfg = GRPOConfig(lr=5e-3, attn="sdpa", clip_low=10.0, clip_high=10.0)

    def mean_lp(r):
        with torch.no_grad():
            lp = token_logprobs(model, pack([r]), attn="sdpa")
        pos = _gen_positions(r)
        return float(lp[pos].mean())

    good0, bad0 = mean_lp(recs[0]), mean_lp(recs[1])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    for _ in range(3):
        stats = train_step(model, opt, [recs], cfg)
    assert stats["groups_kept"] == 1
    assert mean_lp(recs[0]) > good0 and mean_lp(recs[1]) < bad0
    assert token_advantages(recs[0], 0.5, 0.0, 0.25)[_gen_positions(recs[0])[0]] == 0.5


# -- system pagers ----------------------------------------------------------------------------
def _task_only(st):
    """Does the task from the visible text only; never touches KV (pagers do)."""
    env, t = st.env, st.log.table
    if env.pending is not None:
        by_pid = {pid: env.ep.pages[e] for e, pid in st.env_page.items()}
        visible = [by_pid[p] for p in t.in_order(t.resident()) if p in by_pid]
        return render_tool_call("store", {"shelf": rule_read(visible)})
    return render_tool_call("next_events", {})


@pytest.mark.parametrize("seed", range(3))
def test_belady_pager_keeps_what_is_needed(seed):
    env = WarehouseEnv(seed, MID)
    b = t1_budget(env, TOK, 0.6)
    rec = run_episode(env, TokenLog(b), ScriptedPolicy(TOK, _task_only), TOK,
                      EpisodeConfig(budget=b, kv_tools=False, max_turns=80), pager=BeladyPager())
    assert rec["reward"] == 1.0
    env2 = WarehouseEnv(seed, MID)
    rec2 = run_episode(env2, TokenLog(b), ScriptedPolicy(TOK, _task_only), TOK,
                       EpisodeConfig(budget=b, kv_tools=False, max_turns=80), pager=RecentPager())
    assert rec2["reward"] <= rec["reward"]

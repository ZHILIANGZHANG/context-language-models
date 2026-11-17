"""SFT v2 / RL v2 changes (docs/rl_design.md §2, §4): page digests, tool permissions, realizable
page-ins, hygiene, counterfactual credit and v2 advantages."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from self_paging.agent.episode import EpisodeConfig, ScriptedPolicy, run_episode  # noqa: E402
from self_paging.core.log import TokenLog  # noqa: E402
from self_paging.core.protocol import Tok, render_tool_call  # noqa: E402
from self_paging.envs.t1_warehouse import T1Config, WarehouseEnv, page_digest, rule_read  # noqa: E402
from self_paging.testing.toy_tokenizer import ToyTokenizer  # noqa: E402
from self_paging.train.advantage import (advantages_v2, counterfactual_credit, has_signal,  # noqa: E402
                                         hygiene_penalty)

TOK = Tok(ToyTokenizer())
CFG = T1Config(n_decisions=3, warmup=10, gap=(3, 8), events_per_page=(3, 6))


def test_page_digest_and_header():
    text = ("[step 3] Facilities notice FN-1: shelf S5 QUARANTINED (x).\n"
            "[step 4] Inbound: item I-1 ...\n    ACTION: STORE I-1 -> S9\n    RESULT: OK\n"
            "[step 5] Outbound: ... S12 is now EMPTY.\n[step 6] Telemetry: zone=A.")
    assert page_digest(text) == "steps 3-6 | quarantine S5 | store S9 | dispatch S12"
    assert page_digest("[step 1] Telemetry: x.\n[step 2] Telemetry: y.") == "steps 1-2 | telemetry only"
    env = WarehouseEnv(0, CFG)
    _, meta, _ = env.step("next_events", {})
    assert meta["digest"].startswith("steps ")
    rec = run_episode(WarehouseEnv(0, CFG), TokenLog(None), ScriptedPolicy(TOK, _reader()), TOK,
                      EpisodeConfig(max_turns=80))
    text = TOK.decode(rec["tokens"])
    assert "[[PAGE" in text and "tool=next_events | steps " in text
    off = T1Config(**{**CFG.__dict__, "digest": False})
    rec2 = run_episode(WarehouseEnv(0, off), TokenLog(None), ScriptedPolicy(TOK, _reader()), TOK,
                       EpisodeConfig(max_turns=80))
    assert "tool=next_events | steps" not in TOK.decode(rec2["tokens"])


def test_unoffered_kv_tools_are_not_executed():
    def evil(st):
        return render_tool_call("kv_out", {"pages": "2"}) + "\n" + render_tool_call("next_events", {})
    rec = run_episode(WarehouseEnv(0, CFG), TokenLog(None), ScriptedPolicy(TOK, evil), TOK,
                      EpisodeConfig(max_turns=3, kv_tools=False))
    assert rec["meta"]["kv_out"] == 0 and rec["meta"]["unoffered_kv_calls"] == 3
    assert rec["meta"]["invalid"] >= 3


def _reader(evict_quarantine=False, recall=False):
    """Rule-reader policy. Optionally pages out quarantine pages right after reading them, and
    optionally pages every host page back in before answering."""
    state = {"recalled": set()}

    def fn(st):
        env, t = st.env, st.log.table
        by_pid = {pid: env.ep.pages[e] for e, pid in st.env_page.items()}
        if env.pending is not None:
            host = [p for p in by_pid if t.pages[p].where == "host"]
            if recall and host and env.pending.page not in state["recalled"]:
                state["recalled"].add(env.pending.page)
                return render_tool_call("kv_in", {"pages": ",".join(map(str, host))})
            visible = [by_pid[p] for p in t.in_order(t.resident()) if p in by_pid]
            return render_tool_call("store", {"shelf": rule_read(visible)})
        out = [p for p, txt in by_pid.items() if evict_quarantine and "QUARANTINED" in txt
               and t.pages[p].where == "gpu"]
        call = render_tool_call("next_events", {})
        return (render_tool_call("kv_out", {"pages": ",".join(map(str, out))}) + "\n" + call) if out else call
    return fn


def _delayed_seed():
    for seed in range(200):
        env = WarehouseEnv(seed, CFG)
        if any(d.kind == "delayed" for d in env.ep.decisions):
            return seed
    raise AssertionError("no delayed decision found")


def test_counterfactual_credit_blames_eviction_and_rewards_recall():
    seed = _delayed_seed()
    env = WarehouseEnv(seed, CFG)
    rec = run_episode(env, TokenLog(None), ScriptedPolicy(TOK, _reader(evict_quarantine=True)), TOK,
                      EpisodeConfig(max_turns=80))
    assert rec["reward"] < 1.0, "evicting the quarantine notice must cost a delayed decision"
    credit, stats = counterfactual_credit(rec, env)
    assert stats["blamed_kv_out"] >= 1
    neg = [i for i, c in enumerate(credit) if c < 0]
    assert neg and all(rec["ctrl_mask"][i] for i in neg), "blame lands on kv_out control tokens only"

    env2 = WarehouseEnv(seed, CFG)
    rec2 = run_episode(env2, TokenLog(None), ScriptedPolicy(TOK, _reader(evict_quarantine=True, recall=True)),
                       TOK, EpisodeConfig(max_turns=80))
    assert rec2["reward"] == 1.0
    credit2, stats2 = counterfactual_credit(rec2, env2)
    assert stats2["credited_kv_in"] >= 1 and stats2["blamed_kv_out"] == 0
    assert any(c > 0 for c in credit2)


def test_hygiene_and_v2_advantages():
    assert hygiene_penalty({}) == 0.0
    assert hygiene_penalty({"forced_evictions": 100, "truncated_pages": 100, "invalid": 100,
                            "kv_errors": 100}) == -0.25
    mk = lambda reward, cost, hyg=0.0: {"reward": reward, "cost": cost, "hygiene": hyg,  # noqa: E731
                                         "loss_mask": [0, 1, 1, 1], "ctrl_mask": [0, 0, 1, 0],
                                         "credit": [0.0, 0.0, 0.0, 0.0]}
    same = [mk(1.0, 10.0), mk(1.0, 20.0)]
    assert not has_signal(advantages_v2(same, w_eff=0.0, w_credit=1.0))
    assert has_signal(advantages_v2(same, w_eff=0.25, w_credit=1.0)), "efficiency signal keeps the group"
    a = advantages_v2([mk(1.0, 10.0), mk(1.0, 10.0, hyg=-0.1)], w_eff=0.0, w_credit=1.0)
    assert a[0][1] > 0 > a[1][1]
    g = [mk(0.5, 10.0), mk(0.5, 10.0)]
    g[0]["credit"] = [0.0, 0.0, -0.2, 0.0]
    a = advantages_v2(g, w_eff=0.0, w_credit=1.0)
    assert a[0][2] == pytest.approx(-0.2) and has_signal(a)


def test_grpo_step_with_reward_v2():
    from self_paging.agent.episode import ModelPolicy
    from self_paging.hf.logstore import PagedKV
    from self_paging.train.common import tiny_qwen3
    from self_paging.train.grpo import GRPOConfig, score_v2, train_step

    model = tiny_qwen3(TOK.vocab_size)
    gen = torch.Generator().manual_seed(0)
    cfg = GRPOConfig(attn="sdpa", lr=1e-3)
    group = []
    for _ in range(2):
        env = WarehouseEnv(1, CFG)
        rec = run_episode(env, PagedKV(model.eval(), strict=False), ModelPolicy(TOK, generator=gen), TOK,
                          EpisodeConfig(max_turns=2, max_new_tokens=12))
        rec["cost"] = 1.0
        score_v2(rec, env, cfg)
        group.append(rec)
    group[0]["reward"] = 1.0
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    stats = train_step(model, opt, [group], cfg, w_eff=0.0)
    assert stats["groups_kept"] == 1 and stats["optimizer_step"]


def test_pool_rows_set_turn_limits():
    import random
    from self_paging.train.grpo import GRPOConfig, collect_groups

    seen = []

    def fake_rollout(model, tok, seed, ratio, cfg, gen):
        seen.append((seed, ratio, cfg.max_turns, cfg.max_new_tokens, cfg.t1.n_decisions))
        return [{"reward": float(i), "hygiene": 0.0, "cost": 1.0, "credit": None} for i in range(cfg.group)]

    row = {"seed": 10001, "ratio": 0.4, "max_turns": 97, "max_new_tokens": 512,
           "t1_config": {**CFG.__dict__, "gap": list(CFG.gap), "events_per_page": list(CFG.events_per_page),
                         "kinds": list(CFG.kinds)}}
    cfg = GRPOConfig(pool=[row], group=2)
    _, kept = collect_groups(None, TOK, cfg, random.Random(0), None, fake_rollout, 1)
    assert kept and seen[0] == (10001, 0.4, 97, 512, CFG.n_decisions)

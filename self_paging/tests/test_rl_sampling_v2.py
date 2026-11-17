"""RL pool strata, adaptive prompt sampler, budget curriculum and OOD v2 (docs/rl_design.md §3).
Model-free except the last test, which runs the GRPO sampling loop with a fake rollout."""

import json
import random
from dataclasses import asdict

import pytest

from self_paging.data.rl_preparation import config, ood_config_v2
from self_paging.data.rl_strata import (LONG_GAP_PAGES, annotate, decision_profile, select_profile_stratified,
                                        select_stratified, shares)
from self_paging.envs.t1_warehouse import generate
from self_paging.train.sampler import PromptSampler, bucket_of, group_signal


def _row(seed, ratio, split="train", eligible=True, cfg=None):
    cfg = cfg or config(seed)
    return {"split": split, "seed": seed, "ratio": ratio, "prompt_id": f"t1-{split}-{seed}-{ratio:g}",
            "t1_config": json.loads(json.dumps(asdict(cfg))), "n_decisions": len(generate(seed, cfg).decisions),
            "train_eligible": eligible, "weight": 1.0}


def test_decision_profile_finds_key_events():
    for seed in range(10000, 10020):
        ep = generate(seed, config(seed))
        prof = decision_profile(ep)
        assert [p["kind"] for p in prof] == [d.kind for d in ep.decisions]
        for d, p in zip(ep.decisions, prof):
            assert p["gap_pages"] >= 0 and p["gap_steps"] > 0
            assert f"[step {d.key_step}] " in ep.pages[d.page - p["gap_pages"]]
    r = annotate(_row(10003, 0.4))
    assert r["n_delayed"] + r["n_explicit"] + r["n_implicit"] == r["n_decisions"]
    assert (r["stratum"] == "long_delayed") == (r["n_long_delayed"] > 0)
    assert all(p["gap_pages"] < LONG_GAP_PAGES for p in r["decision_profile"]
               if p["kind"] == "delayed") or r["n_long_delayed"]


def _synthetic(n_worlds=300):
    """Annotated-looking rows with a natural delayed share of ~1/3 (fast: no world generation)."""
    rng = random.Random(0)
    rows = []
    for w in range(n_worlds):
        n = 2 + w % 5
        nd = sum(rng.random() < 1 / 3 for _ in range(n))
        long_ = nd > 0 and rng.random() < 0.3
        for ratio in (1.0, 0.6, 0.4, 0.25):
            rows.append({"split": "train", "ratio": ratio, "prompt_id": f"w{w}-{ratio}", "n_decisions": n,
                         "n_delayed": nd, "train_eligible": True,
                         "stratum": "long_delayed" if long_ else ("delayed" if nd else "no_delayed")})
    return rows


def test_stratified_selection_meets_shares_and_keeps_quotas():
    rows = _synthetic()
    natural = shares([r for r in rows if r["ratio"] == 1.0])
    assert natural["delayed_decision_share"] < 0.37
    sel, rep = select_stratified(rows, 400)
    assert len(sel) == 400 and not any(rep["gaps"].values())
    assert sum(r["ratio"] == 1.0 for r in sel) == 40 and sum(r["ratio"] == 0.25 for r in sel) == 120
    for ratio, v in rep["by_ratio"].items():
        assert v["delayed_target_met"] and v["long_target_met"], ratio
        counts = list(v["selected"]["n_decisions"].values())
        assert max(counts) - min(counts) <= 1  # decision counts stay balanced
    assert len({r["prompt_id"] for r in sel}) == 400
    assert select_stratified(rows, 400)[0] == sel  # deterministic
    prof = select_profile_stratified(sel, 24)
    assert {(r["ratio"], r["stratum"]) for r in prof} == {(r, s) for r in (1.0, 0.6, 0.4, 0.25)
                                                         for s in ("long_delayed", "delayed", "no_delayed")}


def test_unreachable_share_is_reported_not_hidden():
    rows = [dict(r, n_delayed=0, stratum="no_delayed") for r in _synthetic(60)]
    sel, rep = select_stratified(rows, 100)
    assert len(sel) == 100
    assert not rep["by_ratio"]["0.6"]["delayed_target_met"] and not rep["by_ratio"]["0.6"]["long_target_met"]


def _groups(std):
    return [{"reward": 0.5 + (std if i % 2 else -std), "hygiene": 0.0} for i in range(4)]


def test_sampler_downweights_quiet_buckets_but_keeps_exploring():
    pool = [{"seed": i, "ratio": 0.4, "n_decisions": 3, "n_delayed": i % 2, "t1_config": {"n_decisions": 3}}
            for i in range(20)]
    s = PromptSampler(pool, explore=0.1, ema=1.0)
    quiet, busy = bucket_of(pool[0]), bucket_of(pool[1])
    assert group_signal(_groups(0.0)) == 0.0 and group_signal(_groups(0.0), has_other_signal=True) == 0.25
    assert group_signal(_groups(0.5)) == 1.0
    s.update(pool[0], _groups(0.0))
    s.update(pool[1], _groups(0.5))
    rng = random.Random(0)
    draws = [bucket_of(s.sample(rng)) for _ in range(4000)]
    share = draws.count(quiet) / len(draws)
    assert 0.03 < share < 0.08  # ≈ explore × half the pool mass = 0.05, never 0
    summ = s.summary()
    assert summ["sampler_buckets_quiet"] == 1 and abs(summ["sampler_quiet_mass"] - 0.05) < 1e-6
    assert busy in s.signal and s.signal[busy] == 1.0
    flat = PromptSampler(pool, adaptive=False)
    flat.update(pool[0], _groups(0.0))
    assert 0.45 < [bucket_of(flat.sample(rng)) for _ in range(2000)].count(quiet) / 2000 < 0.55


def test_sampler_curriculum_and_lazy_annotation():
    pool = [_row(10000 + i, r) for i in range(3) for r in (0.6, 0.25)]
    assert "n_delayed" not in pool[0]
    s = PromptSampler(pool, ratio_start={0.25: 30})
    assert "n_delayed" in s.pool[0]  # annotated on start-up for older pools
    rng = random.Random(1)
    assert {s.sample(rng, 0)["ratio"] for _ in range(200)} == {0.6}
    assert {s.sample(rng, 30)["ratio"] for _ in range(200)} == {0.6, 0.25}
    with pytest.raises(ValueError):
        PromptSampler(pool, ratio_start={0.6: 5, 0.25: 5}).sample(rng, 0)


def test_grpo_collect_groups_uses_sampler_and_curriculum():
    from self_paging.train.grpo import GRPOConfig, allowed_ratios, collect_groups, make_sampler

    pool = [_row(10000 + i, r) for i in range(4) for r in (0.6, 0.4, 0.25)]
    seen = []

    def fake(model, tok, seed, ratio, cfg, gen):
        seen.append(ratio)
        return [{"reward": float(i % 2), "hygiene": 0.0, "cost": 1.0, "credit": None} for i in range(cfg.group)]

    cfg = GRPOConfig(pool=pool, group=2, ratio_start={0.25: 10})
    sampler = make_sampler(cfg)
    collect_groups(None, None, cfg, random.Random(0), None, fake, 30, sampler, step=0)
    assert set(seen) == {0.6, 0.4} and sum(sampler.seen.values()) == 30
    seen.clear()
    collect_groups(None, None, cfg, random.Random(0), None, fake, 60, sampler, step=10)
    assert 0.25 in seen
    plain = GRPOConfig(ratios=(0.25, 0.6), ratio_start={0.25: 5})
    assert allowed_ratios(plain, 0) == [0.6] and allowed_ratios(plain, 5) == [0.25, 0.6]


def test_ood_v2_is_harder_but_shorter_than_v1():
    for seed in range(210000, 210010):
        v2, v1 = ood_config_v2(seed), config(seed, True, ood_version=1)
        assert 7 <= v2.n_decisions <= 9 and v2.events_per_page == (6, 10)
        assert config(seed, True) == v2
        e2, e1 = generate(seed, v2), generate(seed, v1)
        assert len(e2.decisions) >= 7
        assert sum(map(len, e2.pages)) < 0.75 * sum(map(len, e1.pages))

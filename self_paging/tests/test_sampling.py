import random

import pytest

from self_paging.train.grpo import GRPOConfig, collect_groups, sample_seed


def test_replacement_sampling_and_heldout_exclusion():
    cfg = GRPOConfig(group=2, seed_range=(89999, 90102), max_sampling_rounds=3)
    sampled_seeds = []

    def rollout(model, tok, seed, ratio, cfg, gen):
        sampled_seeds.append(seed)
        # Initial two groups have no signal; replacements must supply it.
        return [{"reward": 0.}, {"reward": float(len(sampled_seeds) > 2)}]

    sampled, kept = collect_groups(None, None, cfg, random.Random(0), None, rollout, 2)
    assert len(sampled) == 4 and len(kept) == 2
    assert all(s in (89999, 90100, 90101) for s in sampled_seeds)


def test_sampling_exhaustion_is_bounded_and_empty_range_rejected():
    cfg = GRPOConfig(group=2, max_sampling_rounds=2)
    sampled, kept = collect_groups(None, None, cfg, random.Random(0), None,
                                  lambda *a: [{"reward": 0.}, {"reward": 0.}], 3)
    assert len(sampled) == 6 and not kept
    cfg.seed_range = (90000, 90005)
    with pytest.raises(ValueError, match="held-out"):
        sample_seed(random.Random(0), cfg)

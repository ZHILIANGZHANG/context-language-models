"""RL prompt pools (docs/data_and_training.md §3.5).

RL does not train on fixed trajectories: it samples prompts and rolls out on-line. A prompt is
(environment, seed, difficulty config, budget ratio). This module builds the pools GRPO samples from
and the fixed pools evaluation runs on.

Stage 1, ``build`` (no model, CPU): for every (seed, ratio) run the rule reader as the task policy
twice, once with ``BeladyPager`` and once with ``RecentPager``:
  * Belady fails            → the budget is infeasible: dropped;
  * Belady − Recent ≥ δ     → "paging-sensitive": which pages stay resident decides success; weighted up;
  * ``oracle_ok`` records whether the KV-tool oracle also passes the SFT gates (information only:
    the RL policy is a model, and the oracle can fail a gate just because its own text is long).

Stage 2, ``profile`` (optional, needs the policy, e.g. the SFT checkpoint): sample k episodes per
prompt, record the pass rate, keep prompts whose pass rate is inside [lo, hi] (groups with all-equal
rewards give GRPO no signal), and keep a small share of the rest at low weight.

``eval`` writes fixed evaluation pools: in-distribution (seeds ≥ 100000, same config sampler) and
out-of-distribution (more decisions, longer gaps, tighter budgets).

    python -m self_paging.data.build_rl_pool build --tokenizer $M --out pools/t1_train.jsonl --seeds 0 5000
    python -m self_paging.data.build_rl_pool profile --tokenizer $M --model ckpt/sft --pool pools/t1_train.jsonl \\
        --out pools/t1_train_profiled.jsonl --samples 4
    python -m self_paging.data.build_rl_pool eval --tokenizer $M --out-dir pools/
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import asdict

from ..agent.episode import EpisodeConfig, ScriptedPolicy, run_episode
from ..agent.pagers import BeladyPager, RecentPager, rule_task_policy
from ..core.log import TokenLog
from ..envs.t1_warehouse import T1Config, WarehouseEnv
from ..oracle.t1_oracle import T1Oracle
from ..train.common import load_tokenizer, read_jsonl, write_jsonl
from .build_t1_sft import passes, t1_budget, varied_config


def config_to_dict(cfg: T1Config) -> dict:
    return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(cfg).items()}


def dict_to_config(d: dict) -> T1Config:
    return T1Config(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})


def ood_config(seed: int) -> T1Config:
    """Out-of-distribution difficulty: more decisions (7-9) than training, still within the 32K scope.
    Same generator as ``rl_preparation.ood_config_v2`` (the v1 OOD did not fit 32K)."""
    from .rl_preparation import ood_config_v2

    return ood_config_v2(seed)


def profile_prompt(seed: int, ratio: float, cfg: T1Config, tok, max_turns: int = 120) -> dict:
    """Model-free difficulty profile of one prompt."""

    def run(pager=None, oracle=False):
        env = WarehouseEnv(seed, cfg)
        b = t1_budget(env, tok, ratio)
        if oracle:
            rec = run_episode(env, TokenLog(b), ScriptedPolicy(tok, T1Oracle("paging", seed=seed)), tok,
                              EpisodeConfig(budget=b, max_turns=max_turns))
        else:
            rec = run_episode(env, TokenLog(b), ScriptedPolicy(tok, rule_task_policy), tok,
                              EpisodeConfig(budget=b, max_turns=max_turns, kv_tools=False), pager=pager)
        return env, b, rec

    env, budget, bel = run(BeladyPager())
    _, _, rec = run(RecentPager())
    _, _, orc = run(oracle=True)
    return {"env": "t1", "seed": seed, "ratio": ratio, "t1_config": config_to_dict(cfg), "budget": budget,
            "n_pages": len(env.ep.pages), "n_decisions": len(env.ep.decisions),
            "page_tokens": sum(len(tok.encode(p)) for p in env.ep.pages),
            "rule_belady": bel["reward"], "rule_recent": rec["reward"],
            "sensitivity": round(bel["reward"] - rec["reward"], 4),
            "oracle_ok": passes(orc, 1.0), "pass_rate": None}


def build_pool(tok, seeds, ratios, vary: bool = True, base: T1Config | None = None, min_sensitivity: float = 0.25,
               sensitive_weight: float = 2.0, config_fn=None) -> tuple[list[dict], Counter]:
    base = base or T1Config()
    pool, stats = [], Counter()
    for seed in seeds:
        cfg = config_fn(seed) if config_fn else (varied_config(seed, base) if vary else base)
        for ratio in ratios:
            e = profile_prompt(seed, ratio, cfg, tok)
            stats["total"] += 1
            if e["rule_belady"] < 1.0:  # no resident set within budget lets even a perfect reader succeed
                stats["infeasible"] += 1
                continue
            e["sensitive"] = e["sensitivity"] >= min_sensitivity
            e["weight"] = sensitive_weight if e["sensitive"] else 1.0
            stats["sensitive" if e["sensitive"] else "insensitive"] += 1
            pool.append(e)
    return pool, stats


def profile_with_model(pool: list[dict], model, tok, samples: int = 4, lo: float = 0.1, hi: float = 0.9,
                       keep_rest: float = 0.1, seed: int = 0, max_turns: int = 30, max_new_tokens: int = 512):
    """Pass rate of the current policy on every prompt; keep [lo, hi], a ``keep_rest`` share of the rest."""
    import torch

    from ..agent.episode import ModelPolicy
    from ..hf.logstore import PagedKV

    rng = random.Random(seed)
    gen = torch.Generator(device=next(model.parameters()).device).manual_seed(seed)
    out, stats = [], Counter()
    for e in pool:
        cfg = dict_to_config(e["t1_config"])
        rewards = []
        for _ in range(samples):
            env = WarehouseEnv(e["seed"], cfg)
            b = t1_budget(env, tok, e["ratio"])
            rec = run_episode(env, PagedKV(model, budget=b, strict=False), ModelPolicy(tok, generator=gen), tok,
                              EpisodeConfig(budget=b, max_turns=max_turns, max_new_tokens=max_new_tokens))
            rewards.append(rec["reward"])
        e = dict(e, pass_rate=sum(r >= 1.0 for r in rewards) / samples, mean_reward=sum(rewards) / samples)
        if lo <= e["pass_rate"] <= hi:
            out.append(e)
            stats["kept"] += 1
        elif rng.random() < keep_rest:
            out.append(dict(e, weight=e["weight"] * 0.25))
            stats["kept_low_weight"] += 1
        else:
            stats["dropped_" + ("too_hard" if e["pass_rate"] < lo else "too_easy")] += 1
    return out, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="model-free training pool")
    b.add_argument("--tokenizer", default="toy")
    b.add_argument("--out", required=True)
    b.add_argument("--seeds", type=int, nargs=2, default=[0, 2000])
    b.add_argument("--ratios", type=float, nargs="+", default=[0.25, 0.4, 0.6])
    b.add_argument("--vary", action=argparse.BooleanOptionalAction, default=True)
    b.add_argument("--min-sensitivity", type=float, default=0.25)
    b.add_argument("--sensitive-weight", type=float, default=2.0)
    p = sub.add_parser("profile", help="filter a pool by the policy's pass rate")
    p.add_argument("--tokenizer")
    p.add_argument("--model", required=True)
    p.add_argument("--pool", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--keep", type=float, nargs=2, default=[0.1, 0.9], metavar=("LO", "HI"))
    p.add_argument("--attn", default="sdpa")
    e = sub.add_parser("eval", help="fixed evaluation pools (in-distribution and out-of-distribution)")
    e.add_argument("--tokenizer", default="toy")
    e.add_argument("--out-dir", required=True)
    e.add_argument("--n", type=int, default=200)
    a = ap.parse_args()

    if a.cmd == "build":
        tok = load_tokenizer(a.tokenizer)
        pool, stats = build_pool(tok, range(*a.seeds), a.ratios, vary=a.vary, min_sensitivity=a.min_sensitivity,
                                 sensitive_weight=a.sensitive_weight)
        write_jsonl(a.out, pool)
        print(json.dumps(dict(stats)), f"-> {a.out}")
    elif a.cmd == "profile":
        import torch

        from ..train.common import load_model

        from ..train.common import tiny_qwen3

        tok = load_tokenizer(a.tokenizer or (None if a.model == "tiny" else a.model))
        base = tiny_qwen3(tok.vocab_size) if a.model == "tiny" else a.model
        model = load_model(base, a.attn, device="cuda" if torch.cuda.is_available() else "cpu").eval()
        pool, stats = profile_with_model(read_jsonl(a.pool), model, tok, a.samples, *a.keep)
        write_jsonl(a.out, pool)
        print(json.dumps(dict(stats)), f"-> {a.out}")
    else:
        tok = load_tokenizer(a.tokenizer)
        ind, s1 = build_pool(tok, range(100000, 100000 + a.n), [0.25, 0.4, 0.6], vary=True)
        ood, s2 = build_pool(tok, range(200000, 200000 + a.n), [0.15, 0.25], config_fn=ood_config)
        write_jsonl(f"{a.out_dir}/t1_eval_indist.jsonl", ind)
        write_jsonl(f"{a.out_dir}/t1_eval_ood.jsonl", ood)
        print("in-distribution", dict(s1), "| out-of-distribution", dict(s2))


if __name__ == "__main__":
    main()

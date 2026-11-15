"""Evaluate on T1 (docs/experiment_plan.md §3): model-managed paging vs system baselines.

Methods:
  model    the model manages KV with kv_out / kv_in / note (zero-shot, SFT or RL checkpoint)
  full     no budget
  recent   the system keeps the most recent pages (StreamingLLM at page level); no KV tools
  belady   the system pages with future knowledge (T1 upper bound); no KV tools
  oracle   the scripted oracle does everything (sanity check: should score 1.0)

    python -m self_paging.eval.t1_eval --model /path/Qwen3-8B --tokenizer /path/Qwen3-8B \\
        --methods full recent belady model --ratios 0.25 0.4 0.6 --seeds 100000 100200 --out eval/t1.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import torch

from ..agent.episode import EpisodeConfig, ModelPolicy, ScriptedPolicy, run_episode
from ..agent.pagers import BeladyPager, RecentPager
from ..core.costs import episode_cost
from ..core.log import TokenLog
from ..data.build_t1_sft import t1_budget
from ..envs.t1_warehouse import T1Config, WarehouseEnv
from ..hf.logstore import PagedKV
from ..oracle.t1_oracle import T1Oracle
from ..train.common import append_jsonl, load_model, load_tokenizer, tiny_qwen3

METHODS = ("model", "full", "recent", "belady", "oracle")


def run_one(method, model, tok, seed, ratio, t1cfg, temperature=0.6, top_p=0.95, top_k=20,
            max_turns=60, max_new_tokens=1024, gen=None) -> dict:
    env = WarehouseEnv(seed, t1cfg)
    budget = None if method == "full" else t1_budget(env, tok, ratio)
    pager = {"recent": RecentPager(), "belady": BeladyPager()}.get(method)
    cfg = EpisodeConfig(budget=budget, max_turns=max_turns, max_new_tokens=max_new_tokens,
                        kv_tools=method in ("model", "oracle"))
    if method == "oracle":
        rec = run_episode(env, TokenLog(budget), ScriptedPolicy(tok, T1Oracle("paging", seed=seed)), tok, cfg)
    else:
        policy = ModelPolicy(tok, temperature=temperature, top_p=top_p, top_k=top_k, generator=gen)
        rec = run_episode(env, PagedKV(model, budget=budget, strict=False), policy, tok, cfg, pager=pager)
    c = episode_cost(rec)
    return {"method": method, "seed": seed, "ratio": ratio, "budget": budget, "reward": rec["reward"],
            "peak_resident": rec["peak_resident"], "tokens": len(rec["tokens"]), **c, **rec["meta"]}


def aggregate(rows) -> dict:
    by = defaultdict(list)
    for r in rows:
        by[(r["method"], r["ratio"])].append(r)
    out = {}
    for (m, ratio), rs in sorted(by.items()):
        n = len(rs)
        out[f"{m}@{ratio}"] = {"n": n, "reward": sum(r["reward"] for r in rs) / n,
                               "cost": sum(r["cost"] for r in rs) / n,
                               "peak_resident": sum(r["peak_resident"] for r in rs) / n,
                               "invalid": sum(r.get("invalid", 0) for r in rs) / n,
                               "truncated": sum(r.get("truncated_tokens", 0) for r in rs) / n}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF path / checkpoint, or 'tiny'")
    ap.add_argument("--tokenizer")
    ap.add_argument("--methods", nargs="+", default=["full", "recent", "belady", "model"], choices=METHODS)
    ap.add_argument("--ratios", type=float, nargs="+", default=[0.25, 0.4, 0.6])
    ap.add_argument("--seeds", type=int, nargs=2, default=[100000, 100200])
    ap.add_argument("--n-decisions", type=int, default=4)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tok = load_tokenizer(a.tokenizer or (None if a.model == "tiny" else a.model))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = tiny_qwen3(tok.vocab_size) if a.model == "tiny" else a.model
    model = load_model(base, a.attn, device=device).eval()
    gen = torch.Generator(device=device).manual_seed(0)
    rows = []
    for seed in range(*a.seeds):
        for method in a.methods:
            for ratio in ([1.0] if method == "full" else a.ratios):
                row = run_one(method, model, tok, seed, ratio, T1Config(n_decisions=a.n_decisions),
                              temperature=a.temperature, gen=gen)
                append_jsonl(a.out, row)
                rows.append(row)
    print(json.dumps(aggregate(rows), indent=1))


if __name__ == "__main__":
    main()

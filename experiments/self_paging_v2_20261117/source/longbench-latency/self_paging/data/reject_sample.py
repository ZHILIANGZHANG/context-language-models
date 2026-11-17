"""Rejection sampling under budget pressure (docs/data_and_training.md §2.4).

Run the current model (zero-shot, or after one SFT round) on training seeds at tight budgets; keep
only episodes that succeed with a legal protocol, no truncation and no forced eviction; at most
``--keep-per-seed`` per seed. Output records are SFT-ready (same format as the oracle data).

    python -m self_paging.data.reject_sample --model /path/Qwen3-8B --tokenizer /path/Qwen3-8B \\
        --out data/t1_reject.jsonl --seeds 0 500 --ratios 0.25 0.5 --samples 8
"""

from __future__ import annotations

import argparse

import torch

from ..agent.episode import EpisodeConfig, ModelPolicy, run_episode
from ..data.build_t1_sft import t1_budget
from ..envs.t1_warehouse import T1Config, WarehouseEnv
from ..hf.logstore import PagedKV
from ..train.common import append_jsonl, load_model, load_tokenizer, tiny_qwen3

GATES = ("invalid", "truncated_tokens", "forced_evictions", "kv_errors", "over_budget")


def accept(rec: dict) -> bool:
    return rec["reward"] >= 1.0 and not any(rec["meta"].get(k, 0) for k in GATES)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs=2, default=[0, 100])
    ap.add_argument("--ratios", type=float, nargs="+", default=[0.25, 0.5])
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--keep-per-seed", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--n-decisions", type=int, default=4)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--yarn", type=float, default=0.0)
    a = ap.parse_args()
    tok = load_tokenizer(a.tokenizer or (None if a.model == "tiny" else a.model))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(tiny_qwen3(tok.vocab_size) if a.model == "tiny" else a.model, a.attn, device=device,
                       yarn_factor=a.yarn).eval()
    gen = torch.Generator(device=device).manual_seed(0)
    total = kept = 0
    for seed in range(*a.seeds):
        got = 0
        for ratio in a.ratios:
            for _ in range(a.samples):
                if got >= a.keep_per_seed:
                    break
                env = WarehouseEnv(seed, T1Config(n_decisions=a.n_decisions))
                b = t1_budget(env, tok, ratio)
                rec = run_episode(env, PagedKV(model, budget=b, strict=False),
                                  ModelPolicy(tok, temperature=a.temperature, top_p=0.8, top_k=20, generator=gen),
                                  tok, EpisodeConfig(budget=b))
                total += 1
                if accept(rec):
                    rec["meta"].update(seed=seed, ratio=ratio, source="reject_sample")
                    append_jsonl(a.out, rec)
                    got += 1
                    kept += 1
    print(f"kept {kept}/{total} sampled episodes -> {a.out}")


if __name__ == "__main__":
    main()

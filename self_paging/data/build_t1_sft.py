"""Build T1 SFT data with the oracle (docs/data_and_training.md §2.2, §2.5, §2.6).

For every seed and every budget ratio (same task, several budgets: "一题多预算"), run the oracle on a
model-free token log and keep the episode only if it passes the quality gates:

  reward == 1, no invalid call, no truncation, no forced eviction, no KV error, never over budget,
  and every stored answer is supported by the visible text (``oracle_unsupported`` == 0).

Optionally drop episodes whose evictions were mostly future-informed (``--max-future-share``).
Budget ratio 1.0 means "no budget" (the no-paging examples, ~15% of the mix by default).

    python -m self_paging.data.build_t1_sft --tokenizer /path/Qwen3-8B --out data/t1_sft.jsonl \\
        --seeds 0 1000 --ratios 1.0 0.6 0.45 --modes paging notes
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

from ..agent.episode import EpisodeConfig, ScriptedPolicy, run_episode
from ..core.log import TokenLog
from ..envs.t1_warehouse import T1Config, WarehouseEnv
from ..oracle.hinted_teacher import HintedTeacher
from ..oracle.t1_oracle import T1Oracle
from ..train.common import load_tokenizer, write_jsonl

GATES = ("invalid", "truncated_tokens", "forced_evictions", "kv_errors", "over_budget", "oracle_unsupported")


def t1_budget(env: WarehouseEnv, tok, ratio: float, overhead: int | None = None) -> int | None:
    """Budget = pinned overhead + ratio × total page text. ``overhead`` defaults to the system
    prompt plus a per-page allowance for headers, gauges and the oracle's own turns."""
    if ratio >= 1.0:
        return None
    pages = [len(tok.encode(p)) for p in env.ep.pages]
    if overhead is None:
        per_page = len(tok.encode("\n<tool_response>\n[[PAGE 999 tool=next_events]]\n\n</tool_response>")) + 8
        overhead = len(tok.encode(env.system)) * 2 + 900 + per_page * len(pages)
    return int(overhead + ratio * sum(pages))


def passes(rec: dict, max_future_share: float) -> bool:
    m = rec["meta"]
    if rec["reward"] < 1.0 or any(m.get(k, 0) for k in GATES):
        return False
    obs, fut = m.get("observable_evictions", 0), m.get("future_informed_evictions", 0)
    return (obs + fut) == 0 or fut / (obs + fut) <= max_future_share


def build(tok, seeds, ratios, modes, cfg: T1Config, max_turns: int, max_future_share: float, teacher=None):
    kept, stats = [], Counter()
    for seed in seeds:
        for ratio in ratios:
            for mode in (["paging"] if ratio >= 1.0 else modes):
                env = WarehouseEnv(seed, cfg)
                budget = t1_budget(env, tok, ratio)
                oracle = T1Oracle(mode, seed=seed)
                fn = oracle if teacher is None else HintedTeacher(teacher, tok, oracle)
                rec = run_episode(env, TokenLog(budget), ScriptedPolicy(tok, fn), tok,
                                  EpisodeConfig(budget=budget, max_turns=max_turns))
                stats[(ratio, mode, "total")] += 1
                if passes(rec, max_future_share):
                    rec["meta"].update(seed=seed, ratio=ratio, mode=mode, source="t1_oracle")
                    kept.append(rec)
                    stats[(ratio, mode, "kept")] += 1
    return kept, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="toy", help="HF tokenizer path (Qwen3-8B) or 'toy'")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs=2, default=[0, 100], metavar=("START", "END"))
    ap.add_argument("--ratios", type=float, nargs="+", default=[1.0, 0.6, 0.45])
    ap.add_argument("--modes", nargs="+", default=["paging", "notes"])
    ap.add_argument("--n-decisions", type=int, default=4)
    ap.add_argument("--max-turns", type=int, default=80)
    ap.add_argument("--max-future-share", type=float, default=1.0)
    ap.add_argument("--teacher", help="HF model that writes the prose (Qwen3-8B or Qwen3.5-9B); default: templates")
    a = ap.parse_args()
    tok = load_tokenizer(a.tokenizer)
    teacher = None
    if a.teacher:
        import torch

        from ..train.common import load_model
        teacher = load_model(a.teacher, "sdpa", device="cuda" if torch.cuda.is_available() else "cpu").eval()
    kept, stats = build(tok, range(*a.seeds), a.ratios, a.modes, T1Config(n_decisions=a.n_decisions),
                        a.max_turns, a.max_future_share, teacher=teacher)
    write_jsonl(a.out, kept)
    print(json.dumps({f"{r}/{m}/{k}": v for (r, m, k), v in sorted(stats.items())}, indent=1))
    print(f"kept {len(kept)} episodes -> {a.out}")


if __name__ == "__main__":
    main()

"""Single-turn long-context QA for the token-level baselines (D11 acceptance, docs/baseline_protocol.md
§6, §7): do our implementations reproduce the trends of their papers?

Task: pass-key retrieval in a haystack (RULER-style NIAH, generated here, no download); the question
comes after the haystack. Two modes:

* ``oneshot`` (the papers' setting): prefill the whole prompt with no budget, compress once to the
  budget with the question tokens as the observation window, then decode. The first answer token is
  computed after compression (the last prompt token is fed after it). Expected: SnapKV, TOVA and
  Quest ≫ StreamingLLM; H2O in between. This checks the implementations;
* ``stream`` (our agent protocol, budget strict): the haystack arrives in pages of ``page_tokens``;
  before each page the kept KV is compressed to ``budget − page − margin``. Irreversible methods
  must choose before the question exists, so they fall towards the needle-survival rate, while Quest
  (reversible) can bring the needle back once the question arrives.

Full is the ceiling. If the oneshot trend does not hold on Qwen3-8B, suspect the implementation
before reporting any baseline number.

    python -m self_paging.eval.token_sanity --model $M --tokenizer $M --out eval/token_sanity.jsonl \\
        --lengths 4000 8000 --budgets 0.1 0.25 --n 50
"""

from __future__ import annotations

import argparse
import json
import random
import re

import torch

from ..agent.token_pagers import TOKEN_PAGERS, make_token_log
from ..core.protocol import IM_END, IM_START, NO_THINK
from ..hf.logstore import PagedKV

FILLER = ("The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. ")


def make_item(rng: random.Random, tok, length: int, depth: float):
    key = str(rng.randint(10000, 99999))
    needle = f" The pass key is {key}. Remember it. {key} is the pass key. "
    question = "What is the pass key? Answer with the number only."
    reps = max(1, length // max(1, len(tok.encode(FILLER))))
    sentences = [FILLER] * reps
    sentences.insert(int(depth * reps), needle)
    head = f"{IM_START}user\n"
    tail = f"\n\n{question}{IM_END}\n{IM_START}assistant\n{NO_THINK}"
    return key, head, "".join(sentences), tail


def run_item(model, tok, method: str, key, head, haystack, tail, budget_ratio: float, page_tokens: int,
             mode: str = "oneshot", max_new: int = 12, margin: int = 32, window: int = 32):
    head_ids, body_ids, tail_ids = tok.encode(head), tok.encode(haystack), tok.encode(tail)
    total = len(head_ids) + len(body_ids) + len(tail_ids) + max_new
    # smallest budget a method can hold: the pinned head, the question, the answer, the recent window, and
    # in stream mode the incoming page plus the previous one (the page being written is always kept)
    floor = len(head_ids) + len(tail_ids) + max_new + margin + window + (2 * page_tokens if mode == "stream" else 0)
    budget = None if method == "full" else max(floor, int(budget_ratio * total))
    log = PagedKV(model) if method == "full" else make_token_log(method, model, budget, window=window)
    pager = None if method == "full" else TOKEN_PAGERS[method]()

    def room(incoming):
        if pager is not None:
            pager.compress(log, budget - incoming - margin)

    log.feed(log.open("task", pinned=True), head_ids)
    for a in range(0, len(body_ids), page_tokens):
        chunk = body_ids[a:a + page_tokens]
        if mode == "stream":
            room(len(chunk))
        log.feed(log.open("tool"), chunk)
    if mode == "stream":
        room(len(tail_ids) + max_new)
        logits = log.feed(log.open("header"), tail_ids)[-1]
    else:  # the question is the observation window; the last prompt token is fed after compression
        q = log.open("header")
        log.feed(q, tail_ids[:-1])
        room(1 + max_new)
        logits = log.feed(log.open("header"), tail_ids[-1:])[-1]
    out = log.open("assistant")
    ids = []
    for _ in range(max_new):
        t = int(logits.argmax())
        ids.append(t)
        if t == tok.im_end:
            break
        logits = log.feed(out, [t])[-1]
    text = tok.decode(ids)
    return {"correct": key in re.findall(r"\d+", text), "answer": text, "budget": budget,
            "peak_resident": log.peak_resident, "tokens": log.log_len}


def main() -> None:
    from ..train.common import load_model, load_tokenizer

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--methods", nargs="+", default=["full", *TOKEN_PAGERS])
    ap.add_argument("--lengths", type=int, nargs="+", default=[4000, 8000])
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.1, 0.25])
    ap.add_argument("--modes", nargs="+", choices=["oneshot", "stream"], default=["oneshot", "stream"])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--page-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    tok = load_tokenizer(a.tokenizer)
    model = load_model(a.model, "sdpa", device="cuda" if torch.cuda.is_available() else "cpu").eval()
    summary: dict = {}
    with open(a.out, "w") as f:
        for length in a.lengths:
            for mode in a.modes:
                for i in range(a.n):
                    rng = random.Random(f"{a.seed}-{length}-{i}")
                    depth = rng.random()
                    item = make_item(rng, tok, length, depth)
                    for method in a.methods:
                        for ratio in ([1.0] if method == "full" else a.budgets):
                            r = run_item(model, tok, method, *item, ratio, a.page_tokens, mode)
                            row = dict(length=length, mode=mode, item=i, depth=round(depth, 3), method=method,
                                       ratio=ratio, **r)
                            f.write(json.dumps(row) + "\n")
                            k = f"{length}/{mode}/{method}@{ratio:g}"
                            summary.setdefault(k, []).append(r["correct"])
    for k, v in summary.items():
        print(f"{k}: {sum(v) / len(v):.3f} (n={len(v)})")


if __name__ == "__main__":
    main()

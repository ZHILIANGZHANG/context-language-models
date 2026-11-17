"""Compose real QA documents into page streams for KV-management SFT data (docs/real_data_sft.md).

A *stream* is one training episode: real paragraphs arrive page by page; questions about them are
asked at different times; the model answers each with its question id. Text, questions and answers
are the datasets' own; only the arrangement is ours.

Design (agreed 2026-11-17, aimed at generalization: no surface feature may predict an action):

* **Sources**: wiki-style questions (HotpotQA, 2WikiMultihopQA, MuSiQue, mixed) share one stream,
  so other questions' paragraphs are real distractors; Qasper papers keep reading order.
* **Length**: a target length per stream, log-uniform in 2K-24K tokens; wiki streams add questions
  until they reach it, Qasper papers are cut there (questions whose evidence is cut are dropped).
* **Pages, two styles**: ``para`` = whole paragraphs up to a per-stream page size drawn
  log-uniform in 128-1024 tokens (mixed topics allowed; small pages would make the label stubs of
  paged-out pages cost as much as the pages);
  ``chunk`` = fixed 256/512/768/1024-token slices that cut through paragraphs, like LongBench's
  pages. Evidence pages = pages overlapping an evidence paragraph.
* **Questions** are asked before their evidence ("first") or after it ("last"), at random times;
  in 30% of streams the "first" questions are all asked up front. Asking and answering are
  separate: a "last" question is due when asked; a "first" question is due when its evidence is
  complete or (60%) at a later page, announced by the environment ("answer q3 now"), so its
  evidence may have been paged out by then.
* **Budget**: absolute KV slots; log-uniform 8-90% of the stream cost, or (25%) an evaluation-like
  value from {2048, 3072, 4096, 6144, 8192}; ~18% of streams have no pressure (no kv_out).
* **Balance**: many candidates are generated and planned with the observable-only teacher
  (``qa_policy.simulate``); streams are then selected so that in every cell (mode x source x length
  bucket x page style) about half of the questions need a page-in.

    python -m self_paging.data.qa_streams --tokenizer $M --out streams.jsonl --n 3000 \\
        --hotpot hotpot_train_v1.1.json --2wiki train.json --musique musique_ans_v1.0_train.jsonl \\
        --qasper qasper-train-v0.3.json --longbench-zip LongBench/data.zip
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass

from .qa_sources import LOADERS, Exclusion, QADoc, clean, norm

WIKI = ("hotpot", "2wiki", "musique")
EVAL_BUDGETS = (2048, 3072, 4096, 6144, 8192)
LEN_BUCKETS = (4000, 8000, 16000)


@dataclass
class StreamConfig:
    length: tuple[int, int] = (2000, 24000)     # target stream tokens, log-uniform
    max_items: int = 16
    page_target: tuple[int, int] = (128, 1024)   # "para" style: whole paragraphs up to a per-stream size
    chunk_sizes: tuple[int, ...] = (256, 512, 768, 1024)
    p_chunk: float = 0.4                         # page style "chunk" vs "para"
    p_shuffle: float = 0.6
    p_first: float = 0.5
    p_front: float = 0.3                         # all "first" questions asked together before page 0
    p_defer: float = 0.6                         # a "first" question is due later than its evidence
    p_no_pressure: float = 0.18
    budget_frac: tuple[float, float] = (0.08, 0.9)
    p_eval_budget: float = 0.25
    p_qasper: float = 0.3
    system_tokens: int = 700
    structure_per_page: int = 40


def len_bucket(n: int) -> str:
    for b in LEN_BUCKETS:
        if n < b:
            return f"<{b // 1000}K"
    return f">={LEN_BUCKETS[-1] // 1000}K"


def _loguniform(rng, lo, hi):
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def para_text(t: str, x: str) -> str:
    return f"{t}: {x}" if t else x


# -- flat composition: paragraphs + questions with evidence paragraphs ---------------------------
def merge_wiki(rng, docs: list[QADoc], cfg: StreamConfig) -> dict:
    paras, owner, seen = [], [], {}
    for d, doc in enumerate(docs):
        for j, (t, x) in enumerate(doc.paragraphs):
            key = norm(t + " " + x)
            if key not in seen:
                seen[key] = len(paras)
                paras.append((t, x))
                owner.append([])
            owner[seen[key]].append((d, j))
    order = list(range(len(paras)))
    if rng.random() < cfg.p_shuffle:
        rng.shuffle(order)
    else:  # loose per-document blocks
        blocks: dict[int, list[int]] = {}
        for i in order:
            blocks.setdefault(owner[i][0][0], []).append(i)
        keys = list(blocks)
        rng.shuffle(keys)
        order = []
        for k in keys:
            rng.shuffle(blocks[k])
            order += blocks[k]
        for _ in range(len(order) // 4):
            a, b = rng.randrange(len(order)), rng.randrange(len(order))
            order[a], order[b] = order[b], order[a]
    pos = {old: new for new, old in enumerate(order)}
    where = {dj: pos[i] for i, os_ in enumerate(owner) for dj in os_}
    questions = [dict(qid=q.qid, question=q.question, answers=q.answers, source=doc.source,
                      evidence_paras=sorted({where[(d, e)] for e in q.evidence}))
                 for d, doc in enumerate(docs) for q in doc.questions]
    return dict(kind="wiki", sources=sorted({d.source for d in docs}), docs=[d.doc_id for d in docs],
                paras=[paras[i] for i in order], questions=questions)


def flat_qasper(doc: QADoc) -> dict:
    return dict(kind="qasper", sources=["qasper"], docs=[doc.doc_id], title=doc.meta.get("title"),
                paras=list(doc.paragraphs),
                questions=[dict(qid=q.qid, question=q.question, answers=q.answers, source="qasper",
                                evidence_paras=list(q.evidence)) for q in doc.questions])


# -- pagination ----------------------------------------------------------------------------------
def paginate_para(rng, flat: dict, count, cfg: StreamConfig) -> tuple[list[dict], dict[int, list[int]]]:
    """Whole paragraphs, grouped until the page reaches a per-stream target size."""
    target = _loguniform(rng, *cfg.page_target)
    pages, page_of, cur, size = [], {}, [], 0
    paras = flat["paras"]
    for p, (t, x) in enumerate(paras):
        n = count(para_text(t, x))
        if cur and size + n > target:
            pages.append(cur)
            cur, size = [], 0
        cur.append(p)
        size += n
    if cur:
        pages.append(cur)
    for g, members in enumerate(pages):
        for p in members:
            page_of[p] = [g]
    return ([dict(text="\n\n".join(para_text(*paras[p]) for p in members), titles=[paras[p][0] for p in members])
             for members in pages], page_of)


def paginate_chunk(rng, flat: dict, tok, cfg: StreamConfig) -> tuple[list[dict], dict[int, list[int]]]:
    """Fixed-size token slices of the whole stream text; cuts are moved (at most 3 tokens) so that no
    character is split."""
    size = rng.choice(cfg.chunk_sizes)
    sep = tok.encode("\n\n")
    ids, spans = [], []
    for k, (t, x) in enumerate(flat["paras"]):
        if k:
            ids += sep
        a = len(ids)
        ids += tok.encode(para_text(t, x))
        spans.append((a, len(ids)))
    cuts = [0]
    while cuts[-1] < len(ids):
        c = min(len(ids), cuts[-1] + size)
        for _ in range(3):
            if c >= len(ids) or "�" not in tok.decode(ids[c - 1:c + 1]):
                break
            c += 1
        cuts.append(min(c, len(ids)))
    pages = [dict(text=tok.decode(ids[a:b]), titles=[]) for a, b in zip(cuts, cuts[1:])]
    page_of = {}
    for p, (a, b) in enumerate(spans):
        page_of[p] = [g for g, (x, y) in enumerate(zip(cuts, cuts[1:])) if x < b and a < y]
    for g, (x, y) in enumerate(zip(cuts, cuts[1:])):
        pages[g]["titles"] = [flat["paras"][p][0] for p, (a, b) in enumerate(spans) if x < b and a < y]
    return pages, page_of


def paginate(rng, flat: dict, tok, cfg: StreamConfig) -> dict:
    style = "chunk" if rng.random() < cfg.p_chunk else "para"
    pages, page_of = (paginate_chunk(rng, flat, tok, cfg) if style == "chunk"
                      else paginate_para(rng, flat, lambda t: len(tok.encode(t)), cfg))
    s = {k: v for k, v in flat.items() if k != "paras"}
    s["style"] = style
    s["pages"] = pages
    s["questions"] = [dict({k: v for k, v in q.items() if k != "evidence_paras"},
                           evidence_pages=sorted({g for e in q["evidence_paras"] for g in page_of[e]}))
                      for q in flat["questions"]]
    return s


def finalize(rng, s: dict, count, cfg: StreamConfig, target: int) -> dict | None:
    lengths, total = [], 0
    for p in s["pages"]:
        n = count(p["text"])
        if total + n > target and lengths:
            break
        lengths.append(n)
        total += n
    s["pages"] = s["pages"][:len(lengths)]
    s["questions"] = [q for q in s["questions"] if q["evidence_pages"] and max(q["evidence_pages"]) < len(lengths)]
    if not s["questions"] or len(lengths) < 3:
        return None
    n = len(lengths)
    front = rng.random() < cfg.p_front
    for q in s["questions"]:
        first, last = min(q["evidence_pages"]), max(q["evidence_pages"])
        if rng.random() < cfg.p_first and first > 0:
            q["mode"], q["ask_after"] = "first", -1 if front else rng.randint(-1, first - 1)
            q["due_after"] = rng.randint(last, n - 1) if rng.random() < cfg.p_defer else last
        else:
            q["mode"], q["ask_after"] = "last", rng.randint(last, n - 1)
            q["due_after"] = q["ask_after"]
    s["front_loaded"] = front
    s["questions"].sort(key=lambda q: (q["ask_after"], q["qid"]))
    s["page_tokens"], s["page_cap"], s["stream_tokens"] = lengths, max(lengths), total
    need = total + cfg.system_tokens + cfg.structure_per_page * n
    # smallest workable budget: system, a label line for every page, the two largest pages, a turn
    floor = cfg.system_tokens + 24 * n + 2 * s["page_cap"] + 4 * cfg.structure_per_page + 80
    if rng.random() < cfg.p_no_pressure or floor >= need:
        s["budget"] = None
    else:
        eval_like = [b for b in EVAL_BUDGETS if floor <= b < need]
        if eval_like and rng.random() < cfg.p_eval_budget:
            s["budget"] = rng.choice(eval_like)
        else:
            lo = max(cfg.budget_frac[0], floor / need)
            s["budget"] = int(_loguniform(rng, lo, max(lo * 1.01, cfg.budget_frac[1])) * need)
        if s["budget"] >= need:
            s["budget"] = None
    s["pressure"] = s["budget"] is not None
    return s


def schedule(rng, s: dict) -> dict:
    """Question-level targets (2026-11-17, item 1): every question of a pressure stream draws
    "page-in needed" or "not needed" with probability 1/2, independently of where it sits; then the
    timing is chosen to realize it. A "first" question gets an immediate (not needed) or deferred
    (needed) due time; while it is open its evidence is protected, so a deferred one needs a page-in
    only under real competition, which is the honest policy. A "last" question is asked at a time
    drawn uniformly among the times at which its evidence is still all resident (not needed) or no
    longer all resident (needed), read off a plan with the questions placed so far (one at a time,
    in order of evidence completion). Unreachable targets fall back to a uniform time; the final plan
    records what really happens."""
    from .qa_policy import simulate

    n = len(s["pages"])
    if not s.get("budget"):
        return s
    for q in s["questions"]:
        q["target_recall"] = rng.random() < 0.5
        if q["mode"] == "first":
            last = max(q["evidence_pages"])
            q["due_after"] = rng.randint(min(n - 1, last + 1), n - 1) if q["target_recall"] and last < n - 1 else last
    placed = [q for q in s["questions"] if q["mode"] == "first"]
    lasts = sorted((q for q in s["questions"] if q["mode"] == "last"), key=lambda q: max(q["evidence_pages"]))
    for q in lasts:  # one at a time: earlier page-ins change what is resident later
        timeline = simulate(dict(s, questions=placed), residency=True)["timeline"]
        ev = set(q["evidence_pages"])
        times = list(range(max(ev), n))
        ok = [t for t in times if (not ev <= timeline[t]) == q["target_recall"]]
        q["ask_after"] = q["due_after"] = rng.choice(ok or times)
        placed.append(q)
    s["questions"].sort(key=lambda q: (q["ask_after"], q["qid"]))
    return s


def candidates(docs_by_source: dict[str, list[QADoc]], n: int, tok, cfg: StreamConfig, seed: int = 0):
    """Yield up to ``n`` finalized and planned streams (``plan`` attached; infeasible ones skipped)."""
    from .qa_policy import Infeasible, simulate

    rng = random.Random(seed)
    count = lambda t: len(tok.encode(t))  # noqa: E731
    wiki = [d for s in WIKI for d in docs_by_source.get(s, [])]
    qasper = list(docs_by_source.get("qasper", []))
    rng.shuffle(wiki)
    rng.shuffle(qasper)
    wi, made = 0, 0
    while made < n and (wi < len(wiki) or qasper):
        target = int(_loguniform(rng, *cfg.length))
        if qasper and (rng.random() < cfg.p_qasper or wi >= len(wiki)):
            flat = flat_qasper(qasper.pop())
        else:
            docs, tokens = [], 0
            while wi < len(wiki) and (tokens < target or len(docs) < 2) and len(docs) < cfg.max_items:
                d = wiki[wi]
                wi += 1
                docs.append(d)
                tokens += sum(count(para_text(t, x)) for t, x in d.paragraphs)
            if len(docs) < 2:
                break
            flat = merge_wiki(rng, docs, cfg)
        s = finalize(rng, paginate(rng, flat, tok, cfg), count, cfg, target)
        if s is None:
            continue
        s["stream_id"] = f"c{seed}-{made:06d}"
        try:
            s = schedule(rng, s)
            plan = simulate(s)
        except Infeasible:
            continue
        s["plan"] = plan
        for q in s["questions"]:
            q.update(plan["outcome"][q["qid"]])
        made += 1
        yield s


def due_quartile(s: dict, q: dict) -> str:
    return f"due{min(3, 4 * q['due_after'] // max(1, len(s['pages'])))}"


def cell(s: dict, q: dict) -> tuple:
    """Balance cell: no surface feature in it may predict a page-in."""
    return (q["mode"], due_quartile(s, q), len_bucket(s["stream_tokens"]), s["style"])


def select_balanced(cands: list[dict], n: int, seed: int = 0, p_no_pressure: float = 0.18,
                    min_cell: int = 20) -> tuple[list[dict], dict]:
    """Pick ``n`` streams so that inside every cell (mode x due quartile x length x page style)
    about half of the questions need a page-in.
    A pressure stream is taken when its questions move their cells towards 50% on balance."""
    rng = random.Random(seed)
    pool = list(cands)
    rng.shuffle(pool)
    free = [s for s in pool if not s["pressure"]][:round(n * p_no_pressure)]
    counts: dict[tuple, list[int]] = {}

    def add(s):
        for q in s["questions"]:
            c = counts.setdefault(cell(s, q), [0, 0])
            c[0] += q["recall"]
            c[1] += 1

    for s in free:
        add(s)
    chosen = list(free)
    for s in pool:
        if len(chosen) >= n:
            break
        if not s["pressure"]:
            continue
        score = 0
        for q in s["questions"]:
            r, t = counts.get(cell(s, q), [0, 0])
            if t < min_cell:
                score += 1
                continue
            share = r / t
            score += 1 if (q["recall"] and share < 0.5) or (not q["recall"] and share > 0.5) else -1
        if score >= 0:
            chosen.append(s)
            add(s)
    stats = {"cells": {"/".join(k): dict(recall=v[0], n=v[1], share=round(v[0] / v[1], 3))
                       for k, v in sorted(counts.items())},
             "candidates": len(cands), "chosen": len(chosen)}
    return chosen, stats


def load_sources(paths: dict[str, str], exclusion: Exclusion | None, limit: int | None = None):
    docs, why = {}, {}
    for src, path in paths.items():
        if not path:
            continue
        lst = docs.setdefault(src, [])
        for doc in LOADERS[src](path):
            d, w = clean(doc, exclusion)
            for k, v in w.items():
                why[f"{src}/{k}"] = why.get(f"{src}/{k}", 0) + v
            if d is not None:
                lst.append(d)
            if limit and len(lst) >= limit:
                break
    return docs, why


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for src in LOADERS:
        ap.add_argument(f"--{src}", help=f"official {src} training file")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--oversample", type=float, default=4.0, help="candidates per selected stream")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-per-source", type=int)
    ap.add_argument("--longbench-zip", help="LongBench v1 data.zip: exclude its questions and paragraphs")
    ap.add_argument("--exclude-jsonl", nargs="*", default=[], help="more evaluation files (question field)")
    a = ap.parse_args()
    from ..train.common import load_tokenizer

    tok = load_tokenizer(a.tokenizer)
    ex = Exclusion()
    if a.longbench_zip:
        ex.add_longbench_zip(a.longbench_zip)
    for p in a.exclude_jsonl:
        ex.add_jsonl(p)
    docs, why = load_sources({s: getattr(a, s) for s in LOADERS}, ex, a.limit_per_source)
    cands = list(candidates(docs, int(a.n * a.oversample), tok, StreamConfig(), a.seed))
    chosen, stats = select_balanced(cands, a.n, a.seed)
    with open(a.out, "w") as f:
        for s in chosen:
            f.write(json.dumps(s) + "\n")
    print(json.dumps({"streams": len(chosen), "documents": {k: len(v) for k, v in docs.items()},
                      "dropped_questions": why, "balance": stats, "config": asdict(StreamConfig())}, indent=2))


if __name__ == "__main__":
    main()

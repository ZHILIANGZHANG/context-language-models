"""Compose real QA documents into page streams for KV-management SFT data (docs/real_data_sft.md).

A *stream* is one training episode: real paragraphs arrive page by page; questions about them are
asked at different times; the model answers each with its question id. Text, questions and answers
are the datasets' own; only the arrangement is ours.

* Wiki-style sources (HotpotQA, 2WikiMultihopQA, MuSiQue): the paragraphs of 2-8 questions (mixed
  sources allowed) share one stream, so the other questions' paragraphs are real distractors and a
  stream needs several recalls at different times. Order: a full shuffle or loose blocks.
* Qasper: one paper in reading order; its questions are asked before the evidence ("first": read
  with the question in mind) or after it ("last": the evidence may have been paged out).
* A page holds 1-3 consecutive paragraphs, so page lengths follow the real text (long tail).
* The budget is an absolute KV slot count drawn per stream from 15-80% of the stream tokens; ~18%
  of streams get no budget pressure at all (they should contain no kv_out). The environment
  declares ``page_cap`` = the longest page of the stream, shown as "next page <= P".

Nothing here needs a model; ``count`` is a token counter (the real Qwen3 tokenizer for data).

    python -m self_paging.data.qa_streams --tokenizer $M --out streams.jsonl --n 3000 \\
        --hotpot hotpot_train_v1.1.json --2wiki train.json --musique musique_ans_v1.0_train.jsonl \\
        --qasper qasper-train-v0.3.json --longbench-zip LongBench/data.zip
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass

from .qa_sources import LOADERS, Exclusion, QADoc, clean, norm

WIKI = ("hotpot", "2wiki", "musique")


@dataclass
class StreamConfig:
    items: tuple[int, int] = (2, 8)            # questions merged into one wiki-style stream
    paras_per_page: tuple[int, int] = (1, 3)
    p_shuffle: float = 0.6                      # full shuffle vs loose per-document blocks
    p_first: float = 0.5                        # question asked before its evidence
    p_no_pressure: float = 0.18
    budget_frac: tuple[float, float] = (0.15, 0.8)
    max_tokens: int = 24000                     # stream text; the rest of the 32K log is structure
    structure_per_page: int = 40                # header, trailer, assistant turn (estimate for budgets)
    system_tokens: int = 700
    p_qasper: float = 0.3                       # share of Qasper streams


def page_text(paras: list[tuple[str, str]]) -> str:
    return "\n\n".join(f"{t}: {x}" if t else x for t, x in paras)


def _group(rng, n_paras: int, cfg: StreamConfig) -> list[list[int]]:
    out, i = [], 0
    while i < n_paras:
        k = rng.randint(*cfg.paras_per_page)
        out.append(list(range(i, min(n_paras, i + k))))
        i += k
    return out


def compose_wiki(rng, docs: list[QADoc], cfg: StreamConfig) -> dict:
    """Merge the paragraphs of several single-question documents into one stream."""
    paras, owner, seen = [], [], {}
    for d, doc in enumerate(docs):
        for j, (t, x) in enumerate(doc.paragraphs):
            key = norm(t + " " + x)
            if key not in seen:  # the same Wikipedia paragraph used by two questions appears once
                seen[key] = len(paras)
                paras.append((t, x))
                owner.append([])
            owner[seen[key]].append((d, j))
    order = list(range(len(paras)))
    if rng.random() < cfg.p_shuffle:
        rng.shuffle(order)
    else:  # loose blocks: documents in random order, paragraphs shuffled inside and lightly mixed
        blocks = {}
        for i in order:
            blocks.setdefault(owner[i][0][0], []).append(i)
        keys = list(blocks)
        rng.shuffle(keys)
        order = []
        for k in keys:
            b = blocks[k]
            rng.shuffle(b)
            order += b
        for _ in range(len(order) // 4):
            a, b = rng.randrange(len(order)), rng.randrange(len(order))
            order[a], order[b] = order[b], order[a]
    paras = [paras[i] for i in order]
    owner = [owner[i] for i in order]
    where = {dj: p for p, os_ in enumerate(owner) for dj in os_}  # (doc, para) -> stream paragraph
    groups = _group(rng, len(paras), cfg)
    page_of = {p: g for g, members in enumerate(groups) for p in members}
    questions = []
    for d, doc in enumerate(docs):
        for q in doc.questions:
            questions.append(dict(qid=q.qid, question=q.question, answers=q.answers, source=doc.source,
                                  evidence_pages=sorted({page_of[where[(d, e)]] for e in q.evidence})))
    return dict(kind="wiki", sources=sorted({d.source for d in docs}), docs=[d.doc_id for d in docs],
                pages=[dict(text=page_text([paras[p] for p in g]), titles=[paras[p][0] for p in g]) for g in groups],
                questions=questions)


def compose_qasper(rng, doc: QADoc, cfg: StreamConfig) -> dict:
    groups = _group(rng, len(doc.paragraphs), cfg)
    page_of = {p: g for g, members in enumerate(groups) for p in members}
    questions = [dict(qid=q.qid, question=q.question, answers=q.answers, source="qasper",
                      evidence_pages=sorted({page_of[e] for e in q.evidence})) for q in doc.questions]
    return dict(kind="qasper", sources=["qasper"], docs=[doc.doc_id], title=doc.meta.get("title"),
                pages=[dict(text=page_text([doc.paragraphs[p] for p in g]),
                            titles=[doc.paragraphs[p][0] for p in g]) for g in groups],
                questions=questions)


def finalize(rng, s: dict, count, cfg: StreamConfig) -> dict | None:
    """Token counts, length cap (drop questions whose evidence was cut), question timing, budget."""
    lengths, total = [], 0
    for p in s["pages"]:
        n = count(p["text"])
        if total + n > cfg.max_tokens:
            break
        lengths.append(n)
        total += n
    s["pages"] = s["pages"][:len(lengths)]
    s["questions"] = [q for q in s["questions"] if q["evidence_pages"] and max(q["evidence_pages"]) < len(lengths)]
    if not s["questions"] or len(lengths) < 3:
        return None
    n = len(lengths)
    for q in s["questions"]:
        first, last = min(q["evidence_pages"]), max(q["evidence_pages"])
        if rng.random() < cfg.p_first and first > 0:
            q["mode"] = "first"
            q["ask_after"] = rng.randint(-1, first - 1)       # -1: before the first page
        else:
            q["mode"] = "last"
            q["ask_after"] = rng.randint(last, n - 1)          # after all of its evidence
    s["questions"].sort(key=lambda q: (q["ask_after"], q["qid"]))
    s["page_tokens"] = lengths
    s["page_cap"] = max(lengths)
    s["stream_tokens"] = total
    need = total + cfg.system_tokens + cfg.structure_per_page * n
    floor = cfg.system_tokens + 2 * s["page_cap"] + 8 * cfg.structure_per_page
    if rng.random() < cfg.p_no_pressure:
        s["budget"], s["pressure"] = None, False
    else:
        b = int(rng.uniform(*cfg.budget_frac) * need)
        s["budget"] = max(b, floor)
        s["pressure"] = s["budget"] < need
        if not s["pressure"]:
            s["budget"] = None
    return s


def build(docs_by_source: dict[str, list[QADoc]], n: int, count, cfg: StreamConfig, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    wiki = [d for s in WIKI for d in docs_by_source.get(s, [])]
    qasper = list(docs_by_source.get("qasper", []))
    rng.shuffle(wiki)
    rng.shuffle(qasper)
    out, wi = [], 0
    while len(out) < n and (wi < len(wiki) or qasper):
        if qasper and (rng.random() < cfg.p_qasper or wi >= len(wiki)):
            s = compose_qasper(rng, qasper.pop(), cfg)
        else:
            k = rng.randint(*cfg.items)
            if wi + k > len(wiki):
                if not qasper:
                    break
                continue
            s = compose_wiki(rng, wiki[wi:wi + k], cfg)
            wi += k
        s = finalize(rng, s, count, cfg)
        if s is not None:
            s["stream_id"] = f"s{seed}-{len(out):06d}"
            out.append(s)
    return out


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
    streams = build(docs, a.n, lambda t: len(tok.encode(t)), StreamConfig(), a.seed)
    with open(a.out, "w") as f:
        for s in streams:
            f.write(json.dumps(s) + "\n")
    print(json.dumps({"streams": len(streams), "documents": {k: len(v) for k, v in docs.items()},
                      "dropped_questions": why, "config": asdict(StreamConfig())}, indent=2))


if __name__ == "__main__":
    main()

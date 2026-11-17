"""Model-free checks of QA page streams (docs/real_data_sft.md §checks). The model-based necessity
check (answer must fail without the evidence pages) is ``eval/qa_necessity.py`` and runs on GPU.

Per question:
* ``answer_elsewhere``: the answer string also appears on a page that is not marked as evidence, so
  another page could answer it and "which page to recall" is ambiguous (dropped);
* ``answer_in_evidence``: the answer string appears in the evidence text (False for abstractive
  answers; kept, reported);
* ``recall_needed``: at the moment the question can be answered, the evidence cannot all sit in the
  most recent window the budget keeps ("last" mode), or cannot all be kept while reading ("first"
  mode: evidence + structure exceed the budget). If False, recency alone answers: these are the
  "no kv_in" negatives, fine in moderation, but they must not be the majority;
* ``bm25_rank``: worst rank of an evidence page among the pages delivered so far, scoring the page
  *text* against the question. An upper bound on what model-written labels can reach; the label
  realizability check redoes this on labels.

``report`` aggregates them with the stream statistics that reveal fixed patterns: page lengths,
budget / stream ratio, pressure onset, questions per stream, evidence position and age.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter

from .qa_sources import norm

_TOK = re.compile(r"[a-z0-9]+")
STOP = set("the a an of in on at to for and or is are was were be by with as from that this which who what "
           "when where how did does do it its their his her they he she has have had".split())


def terms(text: str) -> list[str]:
    return [w for w in _TOK.findall(text.lower()) if w not in STOP]


def bm25_scores(query: str, docs: list[str], k1: float = 1.2, b: float = 0.75) -> list[float]:
    toks = [terms(d) for d in docs]
    n = len(docs)
    avg = sum(map(len, toks)) / max(1, n)
    df = Counter(w for t in toks for w in set(t))
    q = terms(query)
    out = []
    for t in toks:
        tf = Counter(t)
        s = 0.0
        for w in q:
            if w in tf:
                idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
                s += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(t) / max(1, avg)))
        out.append(s)
    return out


def recent_window(page_tokens: list[int], upto: int, room: int) -> set[int]:
    """Pages ending at ``upto`` (inclusive) that fit, newest first, in ``room`` tokens."""
    out, used = set(), 0
    for p in range(upto, -1, -1):
        if used + page_tokens[p] > room:
            break
        out.add(p)
        used += page_tokens[p]
    return out


def check_question(s: dict, q: dict, system_tokens: int = 700, structure_per_page: int = 40) -> dict:
    pages, lengths = s["pages"], s["page_tokens"]
    ev = q["evidence_pages"]
    answers = [norm(a) for a in q["answers"] if norm(a)]
    ev_text = " " + " ".join(norm(pages[p]["text"]) for p in ev) + " "
    others = [p for p in range(len(pages)) if p not in ev]
    elsewhere = any(any(f" {a} " in f" {norm(pages[p]['text'])} " for a in answers) for p in others)
    in_ev = any(f" {a} " in ev_text for a in answers)
    ready = max(ev) if q["mode"] == "first" else q["ask_after"]  # when it can be answered
    budget = s.get("budget")
    if budget is None:
        recall = False
    elif q["mode"] == "last":
        room = budget - system_tokens - structure_per_page * (ready + 1)
        recall = not set(ev) <= recent_window(lengths, ready, max(0, room))
    else:
        recall = sum(lengths[p] for p in ev) + system_tokens + structure_per_page * (ready + 1) + lengths[ready] > budget
    seen = list(range(ready + 1))
    sc = bm25_scores(q["question"], [pages[p]["text"] for p in seen])
    order = sorted(seen, key=lambda p: -sc[p])
    rank = max(order.index(p) for p in ev) + 1
    return dict(answer_elsewhere=elsewhere, answer_in_evidence=in_ev, recall_needed=recall, bm25_rank=rank,
                n_seen=len(seen), evidence_age=ready - min(ev), ready=ready)


def annotate(s: dict) -> dict:
    s = dict(s)
    s["questions"] = [dict(q, checks=check_question(s, q)) for q in s["questions"]]
    return s


def _q(xs, qs=(0.1, 0.5, 0.9)):
    xs = sorted(xs)
    return [xs[min(len(xs) - 1, int(q * len(xs)))] for q in qs] if xs else []


def report(streams: list[dict]) -> dict:
    qs = [(s, q) for s in streams for q in s["questions"]]
    c = Counter()
    for s, q in qs:
        k = q["checks"]
        c["questions"] += 1
        c[f"mode/{q['mode']}"] += 1
        c["answer_elsewhere"] += k["answer_elsewhere"]
        c["answer_in_evidence"] += k["answer_in_evidence"]
        c["recall_needed"] += k["recall_needed"]
        c[f"recall_needed/{q['mode']}"] += k["recall_needed"]
        c["bm25_top1"] += k["bm25_rank"] <= len(q["evidence_pages"])
        c["bm25_top5"] += k["bm25_rank"] <= 5
    pressure = [s for s in streams if s.get("budget")]
    onset = []
    for s in pressure:  # first page at which the stream no longer fits the budget
        used = 700
        for i, n in enumerate(s["page_tokens"]):
            used += n + 40
            if used > s["budget"]:
                onset.append(i)
                break
    n_q = c["questions"] or 1
    return {
        "streams": len(streams),
        "by_kind": dict(Counter(s["kind"] for s in streams)),
        "sources_per_stream": dict(Counter(len(s["sources"]) for s in streams)),
        "questions": c["questions"],
        "questions_per_stream": dict(sorted(Counter(len(s["questions"]) for s in streams).items())),
        "share": {k: round(c[k] / n_q, 3) for k in ("mode/first", "mode/last", "answer_elsewhere",
                                                     "answer_in_evidence", "recall_needed", "bm25_top1", "bm25_top5")},
        "recall_needed_by_mode": {m: round(c[f"recall_needed/{m}"] / max(1, c[f"mode/{m}"]), 3) for m in ("first", "last")},
        "page_tokens_q10_50_90": _q([n for s in streams for n in s["page_tokens"]]),
        "page_cap_q10_50_90": _q([s["page_cap"] for s in streams]),
        "pages_per_stream_q10_50_90": _q([len(s["pages"]) for s in streams]),
        "stream_tokens_q10_50_90": _q([s["stream_tokens"] for s in streams]),
        "no_pressure_share": round(1 - len(pressure) / max(1, len(streams)), 3),
        "budget_q10_50_90": _q([s["budget"] for s in pressure]),
        "budget_over_stream_q10_50_90": _q([round(s["budget"] / s["stream_tokens"], 2) for s in pressure]),
        "pressure_onset_page_q10_50_90": _q(onset),
        "evidence_pages_per_question": dict(sorted(Counter(len(q["evidence_pages"]) for _, q in qs).items())),
        "evidence_relative_position_q10_50_90": _q([round(min(q["evidence_pages"]) / max(1, len(s["pages"]) - 1), 2)
                                                    for s, q in qs]),
        "evidence_age_pages_q10_50_90": _q([q["checks"]["evidence_age"] for _, q in qs]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--streams", required=True)
    ap.add_argument("--out", help="annotated streams (adds checks to every question)")
    a = ap.parse_args()
    streams = [annotate(json.loads(line)) for line in open(a.streams) if line.strip()]
    if a.out:
        with open(a.out, "w") as f:
            f.writelines(json.dumps(s) + "\n" for s in streams)
    print(json.dumps(report(streams), indent=2))


if __name__ == "__main__":
    main()

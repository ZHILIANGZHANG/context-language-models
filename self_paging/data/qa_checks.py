"""Checks of QA page streams (docs/real_data_sft.md §checks). The model-based necessity check (the
answer must fail without the evidence pages) is ``eval/qa_necessity.py`` and runs on GPU.

Per question, model-free:
* ``answer_elsewhere``: the answer string also appears on a non-evidence page (mostly coincidences
  of short answers; the necessity check decides);
* ``answer_in_evidence``: the answer string appears in the evidence text (False for abstractive);
* ``bm25_rank``: worst rank of an evidence page among the pages delivered so far, by page text;
and from the teacher plan (``qa_policy.simulate``, attached by ``qa_streams``): ``recall`` (the
teacher pages something in for it), ``steps`` (page-in steps; multi-hop needs several) and
``realizable`` (every step's target was findable from what is visible).

``report`` also measures whether *surface features* predict a page-in: mutual information between
``recall`` and the mode, source, length bucket, page style, ask position and questions per stream.
It should be near zero: if a surface feature predicts the action, the model will learn the
feature instead of the cue.
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


def ready_page(q: dict) -> int:
    return q.get("due_after", max(q["evidence_pages"]) if q["mode"] == "first" else q["ask_after"])


def check_question(s: dict, q: dict) -> dict:
    pages = s["pages"]
    ev = q["evidence_pages"]
    answers = [norm(a) for a in q["answers"] if norm(a)]
    ev_text = " " + " ".join(norm(pages[p]["text"]) for p in ev) + " "
    others = [p for p in range(len(pages)) if p not in ev]
    elsewhere = any(any(f" {a} " in f" {norm(pages[p]['text'])} " for a in answers) for p in others)
    ready = ready_page(q)
    seen = list(range(ready + 1))
    sc = bm25_scores(q["question"], [pages[p]["text"] for p in seen])
    order = sorted(seen, key=lambda p: -sc[p])
    return dict(answer_elsewhere=elsewhere, answer_in_evidence=any(f" {a} " in ev_text for a in answers),
                bm25_rank=max(order.index(p) for p in ev) + 1, evidence_age=ready - min(ev), ready=ready)


def annotate(s: dict) -> dict:
    s = dict(s)
    s["questions"] = [dict(q, checks=check_question(s, q)) for q in s["questions"]]
    return s


def _q(xs, qs=(0.1, 0.5, 0.9)):
    xs = sorted(xs)
    return [xs[min(len(xs) - 1, int(q * len(xs)))] for q in qs] if xs else []


def mutual_information(xs: list, ys: list) -> float:
    """I(X; Y) in bits for discrete sequences."""
    n = len(xs)
    if not n:
        return 0.0
    pxy, px, py = Counter(zip(xs, ys)), Counter(xs), Counter(ys)
    return sum(c / n * math.log2((c / n) / (px[x] / n * py[y] / n)) for (x, y), c in pxy.items())


def surface_features(s: dict, q: dict) -> dict:
    from .qa_streams import len_bucket

    n = len(s["pages"])
    return {"mode": q["mode"], "source": q["source"], "length": len_bucket(s["stream_tokens"]),
            "style": s.get("style", "para"), "ask_quartile": min(3, 4 * max(0, q["ask_after"]) // max(1, n)),
            "due_quartile": min(3, 4 * max(0, ready_page(q)) // max(1, n)),
            "questions_in_stream": min(len(s["questions"]), 8), "pressure": bool(s.get("budget"))}


def surface_predictability(streams: list[dict], folds: int = 5, keys=None) -> dict:
    """How well the best lookup table over surface-feature combinations predicts a page-in, out of
    fold (streams split into folds). Compare with the majority-class rate: the closer, the less a
    model can learn from surface features. ``pressure`` is excluded by default: it is shown on the
    gauge and is a legitimate cue."""
    keys = keys or ("mode", "source", "length", "style", "ask_quartile", "due_quartile", "questions_in_stream")
    rows = [(i % folds, tuple(surface_features(s, q)[k] for k in keys), bool(q.get("recall")))
            for i, s in enumerate(streams) if s.get("budget") for q in s["questions"]]
    if not rows:
        return {}
    hit = 0
    for f in range(folds):
        table: dict = {}
        for g, x, y in rows:
            if g != f:
                table.setdefault(x, Counter())[y] += 1
        prior = Counter(y for g, _, y in rows if g != f).most_common(1)[0][0]
        for g, x, y in rows:
            if g == f:
                hit += (table[x].most_common(1)[0][0] if x in table else prior) == y
    base = max(sum(y for *_, y in rows), sum(not y for *_, y in rows)) / len(rows)
    return {"questions_under_pressure": len(rows), "majority_rate": round(base, 3),
            "surface_lookup_accuracy": round(hit / len(rows), 3)}


def report(streams: list[dict]) -> dict:
    qs = [(s, q) for s in streams for q in s["questions"]]
    nq = max(1, len(qs))
    rec = [bool(q.get("recall")) for _, q in qs]
    feats = [surface_features(s, q) for s, q in qs]
    h = mutual_information(rec, rec) or 1e-9  # entropy of recall
    mi = {k: round(mutual_information([f[k] for f in feats], rec), 4) for k in feats[0]} if feats else {}
    by = {}
    for k in ("mode", "source", "length", "style"):
        groups: dict = {}
        for f, r in zip(feats, rec):
            groups.setdefault(f[k], []).append(r)
        by[k] = {g: round(sum(v) / len(v), 3) for g, v in sorted(groups.items())}
    pressure = [s for s in streams if s.get("budget")]
    plans = [s["plan"] for s in streams if "plan" in s]
    k = [q["checks"] for _, q in qs if "checks" in q]
    return {
        "streams": len(streams),
        "by_kind": dict(Counter(s["kind"] for s in streams)),
        "by_style": dict(Counter(s.get("style", "para") for s in streams)),
        "sources_per_stream": dict(Counter(len(s["sources"]) for s in streams)),
        "questions": len(qs),
        "questions_per_stream": dict(sorted(Counter(len(s["questions"]) for s in streams).items())),
        "recall_share": round(sum(rec) / nq, 3),
        "recall_share_by": by,
        "recall_entropy_bits": round(h, 4),
        "surface_mutual_information_bits": mi,
        "surface_predictability": surface_predictability(streams),
        "pagein_steps": dict(sorted(Counter(q.get("steps", 0) for _, q in qs if q.get("recall")).items())),
        "realizable_share_of_recalls": round(sum(q.get("realizable", True) for _, q in qs if q.get("recall"))
                                             / max(1, sum(rec)), 3),
        "kv_out_per_stream_pressure_q10_50_90": _q([s["plan"]["kv_out"] for s in pressure if "plan" in s]),
        "kv_out_in_no_pressure_streams": sum(s["plan"]["kv_out"] for s in streams if not s.get("budget") and "plan" in s),
        "kv_in_per_stream_q10_50_90": _q([p["kv_in"] for p in plans]),
        "answer_elsewhere": round(sum(c["answer_elsewhere"] for c in k) / max(1, len(k)), 3),
        "answer_in_evidence": round(sum(c["answer_in_evidence"] for c in k) / max(1, len(k)), 3),
        "bm25_top5": round(sum(c["bm25_rank"] <= 5 for c in k) / max(1, len(k)), 3),
        "page_tokens_q10_50_90": _q([n for s in streams for n in s["page_tokens"]]),
        "page_cap_q10_50_90": _q([s["page_cap"] for s in streams]),
        "stream_tokens_q10_50_90": _q([s["stream_tokens"] for s in streams]),
        "stream_tokens_by_bucket": dict(Counter(surface_features(s, s["questions"][0])["length"] for s in streams)),
        "no_pressure_share": round(1 - len(pressure) / max(1, len(streams)), 3),
        "budget_q10_50_90": _q([s["budget"] for s in pressure]),
        "budget_over_stream_q10_50_90": _q([round(s["budget"] / s["stream_tokens"], 2) for s in pressure]),
        "eval_like_budget_share": round(sum(s["budget"] in (2048, 3072, 4096, 6144, 8192) for s in pressure)
                                        / max(1, len(pressure)), 3),
        "evidence_pages_per_question": dict(sorted(Counter(len(q["evidence_pages"]) for _, q in qs).items())),
        "evidence_age_pages_q10_50_90": _q([c["evidence_age"] for c in k]),
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

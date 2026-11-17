"""The observable-only teacher for QA page streams, at the plan level (pages and token counts, no
text generation yet). docs/real_data_sft.md §decision program.

Every decision uses only what the student will see, except one thing the datasets give us: which
pages are evidence. That use is checked for realizability (``realizable``) instead of trusted.

* **When to page out**: before a page arrives (or before a page-in) the resident cost plus the
  incoming page does not fit the budget. Cost = system prompt + resident pages with their headers
  + one label stub per paged-out page + one assistant turn. The student sees it on the gauge
  (``free F | next page <= P``).
* **What to page out**: first pages not relevant to any open question, then pages relevant to open
  questions. A page counts as relevant only if the student could tell: it is evidence of an open
  question and BM25 over the text seen so far ranks it in the top ``top_m`` for the question plus
  the already-seen evidence text (multi-hop: the bridge entity comes from the first hop). Inside a
  tier the choice is uninformed (random, and marked ``informed=False`` so the page-id tokens get no
  loss); there is no "always the oldest" rule.
* **When to page in, and which pages**: a question is answered when it is due (``due_after``:
  right after asking for "last" questions; for "first" questions when the evidence is complete or
  later, when the environment asks for the answer), with all its evidence resident. If some is paged out, page it in step by step: at each step the pages that rank in the
  top ``top_m`` among the paged-out pages for (question + evidence text already resident) come in
  together; the next step uses what was just read. A step whose target does not rank there is
  ``realizable=False`` (the student would fetch the wrong page); such questions are flagged.
  Page-out pages are ranked by their text here, as a stand-in for the model-written labels; the
  label check redoes this with the real labels.
* **No page-in** when the evidence is still resident, and **no page-out** without pressure.

``simulate`` returns the event plan and, per question, ``recall`` (any kv_in), ``steps`` and
``realizable``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .qa_checks import bm25_scores


@dataclass
class PolicyConfig:
    system_tokens: int = 700
    header_tokens: int = 16       # per resident page (header + trailer)
    stub_tokens: int = 24         # per paged-out page (model-written label line)
    turn_tokens: int = 80         # one assistant turn + gauge (transient)
    top_m: int = 2


class Infeasible(Exception):
    """The budget cannot hold even the protected pages plus the incoming one."""


def _rank_ok(query: str, candidates: list[int], texts: list[str], target: int, top_m: int) -> bool:
    if len(candidates) <= top_m:
        return target in candidates
    sc = bm25_scores(query, [texts[p] for p in candidates])
    order = sorted(range(len(candidates)), key=lambda i: -sc[i])
    return target in {candidates[i] for i in order[:top_m]}


def simulate(s: dict, cfg: PolicyConfig | None = None, seed: int = 0, residency: bool = False) -> dict:
    cfg = cfg or PolicyConfig()
    rng = random.Random(f"{s.get('stream_id', '')}-{seed}")
    texts = [p["text"] for p in s["pages"]]
    L = s["page_tokens"]
    B = s.get("budget")
    qs = {q["qid"]: q for q in s["questions"]}
    asks: dict[int, list[str]] = {}
    for q in s["questions"]:
        asks.setdefault(q["ask_after"], []).append(q["qid"])
    resident: list[int] = []
    out: set[int] = set()
    timeline: list[set[int]] = []  # resident pages after page t was read (before questions due at t)
    peak = [0]
    open_q: list[str] = []
    events: list[dict] = []
    outcome = {qid: {"recall": False, "steps": 0, "realizable": True} for qid in qs}

    def cost(extra: int = 0) -> int:
        return (cfg.system_tokens + sum(L[p] + cfg.header_tokens for p in resident)
                + cfg.stub_tokens * len(out) + cfg.turn_tokens + extra)

    memo: dict = {}

    def relevant(p: int, seen: int) -> bool:
        key = (p, seen, tuple(open_q))
        if key in memo:
            return memo[key]
        memo[key] = _relevant(p, seen)
        return memo[key]

    def _relevant(p: int, seen: int) -> bool:
        for qid in open_q:
            q = qs[qid]
            if p in q["evidence_pages"]:
                other = " ".join(texts[e] for e in q["evidence_pages"] if e != p and e <= seen)
                if _rank_ok(q["question"] + " " + other, list(range(seen + 1)), texts, p, cfg.top_m):
                    return True
        return False

    def make_room(need: int, protect: set[int], seen: int) -> None:
        if B is None:
            return
        while cost(need) > B:
            cands = [p for p in resident if p not in protect]
            if not cands:
                raise Infeasible
            tier0 = [p for p in cands if not relevant(p, seen)]
            pool = tier0 or cands
            victim = rng.choice(pool)
            resident.remove(victim)
            out.add(victim)
            events.append(dict(op="kv_out", pages=[victim], informed=False,
                               tier="irrelevant" if tier0 else "relevant", free=B - cost(), need=need))

    def try_answer(qid: str, seen: int) -> None:
        q = qs[qid]
        ev = q["evidence_pages"]
        step = 0
        while True:
            missing = [p for p in ev if p in out]
            if not missing:
                break
            step += 1
            known = " ".join(texts[p] for p in ev if p in resident)
            query = q["question"] + " " + known
            cands = sorted(out)
            batch = [p for p in missing if _rank_ok(query, cands, texts, p, cfg.top_m)]
            if not batch:  # nothing realizable from what is visible: fetch the next missing page anyway, flagged
                outcome[qid]["realizable"] = False
                batch = [missing[0]]
            make_room(sum(L[p] + cfg.header_tokens - cfg.stub_tokens for p in batch), set(ev) & set(resident), seen)
            for p in batch:
                out.discard(p)
                resident.append(p)
            peak[0] = max(peak[0], cost())
            events.append(dict(op="kv_in", pages=batch, qid=qid, step=step))
        outcome[qid]["recall"] = step > 0
        outcome[qid]["steps"] = step
        events.append(dict(op="answer", qid=qid))
        open_q.remove(qid)

    def ask(t: int) -> None:
        for qid in asks.get(t, []):
            open_q.append(qid)
            events.append(dict(op="ask", qid=qid))

    ask(-1)
    for t in range(len(texts)):
        make_room(L[t] + cfg.header_tokens, set(), t - 1)
        resident.append(t)
        peak[0] = max(peak[0], cost())
        events.append(dict(op="read", page=t))
        if residency:
            timeline.append(set(resident))
        ask(t)
        for qid in list(open_q):
            if qs[qid].get("due_after", max(qs[qid]["evidence_pages"])) <= t:  # due now (the environment says so)
                try_answer(qid, t)
    if open_q:
        raise ValueError(f"unanswered questions {open_q}")
    return dict(events=events, outcome=outcome, peak=peak[0], **({"timeline": timeline} if residency else {}),
                kv_out=sum(e["op"] == "kv_out" for e in events),
                kv_in=sum(e["op"] == "kv_in" for e in events))

"""Quality and diversity report for SFT data (docs/data_and_training.md §2.8–§2.9).

Page-out / page-in events are reconstructed from the resident-set versions of each record
(system housekeeping of gauges and acknowledgements is ignored). Prints, per budget × mode:
episodes, page-outs and page-ins per episode, share of episodes with a page-in, pages paged in
more than once, median turns a page stays out; then the action-combination diversity, episode
shapes and recovery examples.

    python -m self_paging.data.inspect_sft --data data/t1_sft.jsonl --tokenizer /path/Qwen3-8B
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from statistics import median

from ..train.common import load_tokenizer, read_jsonl

MOVABLE = {"tool", "assistant", "note"}
SYM = {"next_events": "n", "store": "s", "kv_out": "o", "kv_in": "i", "note": "w", "kv_drop": "d"}


def page_events(rec: dict) -> list[tuple[int, str, int, str]]:
    """(turn, "out"|"in", pid, kind) for every model-initiated page move, in time order."""
    kind = {p["pid"]: p["kind"] for p in rec["pages"]}
    start = {p["pid"]: p["start"] for p in rec["pages"]}
    snaps = [set(s) for s in rec["snapshots"]]
    first: dict[int, int] = {}
    for i, v in enumerate(rec["seg_of"]):
        first.setdefault(v, i)
    nxt = len(rec["seg_of"])
    for v in range(len(snaps) - 1, -1, -1):  # versions without tokens share the next boundary
        if v in first:
            nxt = first[v]
        else:
            first[v] = nxt
    a_starts = sorted(p["start"] for p in rec["pages"] if p["kind"] == "assistant")

    def turn(pos):
        return sum(1 for s in a_starts if s < pos) - 1

    ev = []
    for v in range(1, len(snaps)):
        t = turn(first[v])
        for p in snaps[v - 1] - snaps[v]:
            if kind[p] in MOVABLE:
                ev.append((t, "out", p, kind[p]))
        for p in snaps[v] - snaps[v - 1]:
            if kind[p] in MOVABLE and start[p] < first[v]:  # pages opened in v are not page-ins
                ev.append((t, "in", p, kind[p]))
    return sorted(ev, key=lambda e: (e[0], 0 if e[1] == "out" else 1))


def turn_actions(rec: dict, tok) -> list[str]:
    out = []
    for p in rec["pages"]:
        if p["kind"] != "assistant":
            continue
        body = tok.decode(rec["tokens"][p["start"]:p["start"] + p["length"]])
        names = re.findall(r'\{"name": "(\w+)", "arguments"', body)
        out.append("".join(SYM.get(x, "?") for x in names) or "-")
    return out


def report(recs: list[dict], tok) -> dict:
    cells = defaultdict(list)
    for r in recs:
        cells[(r["meta"].get("ratio"), r["meta"].get("mode"))].append(r)
    table = {}
    back = Counter()
    stay_out = []
    for key in sorted(cells, key=lambda k: (-(k[0] or 0), str(k[1]))):
        rs = cells[key]
        outs = ins = with_in = multi = 0
        for r in rs:
            ev = page_events(r)
            o = sum(1 for e in ev if e[1] == "out")
            i = sum(1 for e in ev if e[1] == "in")
            outs, ins, with_in = outs + o, ins + i, with_in + bool(i)
            per = defaultdict(list)
            for e in ev:
                per[e[2]].append(e)
            last = {}
            for t, kind, p, _ in ev:
                if kind == "out":
                    last[p] = t
                elif p in last:
                    stay_out.append(t - last[p])
            for es in per.values():
                n_in = sum(1 for e in es if e[1] == "in")
                if any(e[1] == "out" for e in es):
                    back[n_in] += 1
                multi += n_in >= 2
        n = len(rs)
        table[f"{key[0]}/{key[1]}"] = {"n": n, "out_per_ep": round(outs / n, 1), "in_per_ep": round(ins / n, 2),
                                       "ep_with_in": f"{with_in / n:.0%}", "pages_in_2plus": multi}
    combos, grams = Counter(), Counter()
    for r in recs:
        seq = turn_actions(r, tok)
        combos.update(seq)
        grams.update(zip(seq, seq[1:], seq[2:]))
    err = [r for r in recs if r["meta"].get("error_pids")]
    return {
        "cells": table,
        "paged_out_pages_by_times_back_in": dict(sorted(back.items())),
        "turns_out_before_back_in": ({"median": median(stay_out), "max": max(stay_out), "n": len(stay_out)}
                                     if stay_out else {}),
        "per_turn_action_combos": dict(combos.most_common()),
        "distinct_3turn_action_ngrams": len(grams),
        "decisions_per_episode": dict(sorted(Counter(r["meta"].get("t1_config", {}).get("n_decisions")
                                                     for r in recs).items(), key=lambda x: str(x[0]))),
        "turns_per_episode": {"min": min(r["meta"]["turns"] for r in recs),
                              "median": median(r["meta"]["turns"] for r in recs),
                              "max": max(r["meta"]["turns"] for r in recs)},
        "recovery": {"episodes": len(err), "mistake_turns_no_loss": sum(len(r["meta"]["error_pids"]) for r in err)},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tokenizer", default="toy")
    a = ap.parse_args()
    print(json.dumps(report(read_jsonl(a.data), load_tokenizer(a.tokenizer)), indent=1))


if __name__ == "__main__":
    main()

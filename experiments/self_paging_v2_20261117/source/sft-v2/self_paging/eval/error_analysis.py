"""Why did a decision go wrong? (docs/rl_design.md §1)

For every wrong T1 decision in evaluation records (``t1_eval --records-out``), check the text the
model could see in the turns between the decision page and the next page request:

  info_missing                 no turn had enough resident environment text for the rule reader
  info_present_reasoning_err   the last turn had enough text, the answer was still wrong
  info_present_then_lost       enough text at some point, gone by the last turn

and, for info_missing, which needed pages were absent (quarantine notice / correction or scan /
store-dispatch records). Model-written notes and earlier assistant turns are NOT counted as
information (decoding them needs the tokenizer), so ``info_missing`` is an upper bound; the
decision-kind breakdown does not depend on that.

    python -m self_paging.eval.error_analysis --records 'raw/*.jsonl.gz' [--n-decisions 4]
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
from collections import Counter, defaultdict

from ..envs.t1_warehouse import T1Config, generate, rule_read


def _open(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def analyse(records, cfg: T1Config) -> dict:
    out = defaultdict(Counter)
    for r in records:
        key = f"{r['method']}@{r['ratio']}"
        ep = generate(r["seed"], cfg)
        if ep.pages != r.get("environment_pages", ep.pages):
            out[key]["config_mismatch"] += 1
            continue
        snaps = [set(s) for s in r["snapshots"]]
        epid = {int(k): v for k, v in r["env_page"].items()}
        start = {p["pid"]: p["start"] for p in r["pages"]}
        a_pages = sorted(p["start"] for p in r["pages"] if p["kind"] == "assistant")
        order = sorted(epid)
        answers = {d["page"]: d.get("predicted") for d in r.get("decisions", [])}
        for D in ep.decisions:
            out[key]["decisions"] += 1
            out[key][f"kind={D.kind}"] += 1
            pred = answers.get(D.page)
            if pred == D.answer:
                out[key]["right"] += 1
                continue
            out[key]["wrong"] += 1
            out[key][f"wrong_kind={D.kind}"] += 1
            if D.page not in epid:
                out[key]["never_reached"] += 1
                continue
            lo = start[epid[D.page]]
            nxt = [start[epid[x]] for x in order if x > D.page]
            hi = nxt[0] if nxt else len(r["tokens"])
            turns = [s for s in a_pages if lo < s < hi] or [lo]
            ok = []
            for s in turns:
                vis = [ep.pages[x] for x in order if epid[x] in snaps[r["seg_of"][s]]]
                ok.append(bool(vis) and rule_read(vis) == D.answer)
            cat = ("info_missing" if not any(ok) else
                   "info_present_reasoning_err" if ok[-1] else "info_present_then_lost")
            out[key][cat] += 1
            if pred is None:
                out[key]["no_answer"] += 1
            if cat == "info_missing":
                last = snaps[r["seg_of"][turns[-1]]]
                for e in ep.need[D.page]:
                    if e in epid and epid[e] not in last:
                        t = ep.pages[e]
                        tag = ("quarantine_notice" if "QUARANTINED" in t else
                               "correction_or_scan" if ("never completed" in t or "Inventory scan" in t) else
                               "store_dispatch_records")
                        out[key][f"missing:{tag}"] += 1
    return {k: dict(v) for k, v in sorted(out.items())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True, help="glob of t1_eval --records-out files (.jsonl or .jsonl.gz)")
    ap.add_argument("--n-decisions", type=int, default=4, help="T1Config used by the evaluation")
    a = ap.parse_args()
    recs = [json.loads(line) for f in sorted(glob.glob(a.records)) for line in _open(f) if line.strip()]
    print(json.dumps(analyse(recs, T1Config(n_decisions=a.n_decisions)), indent=1))


if __name__ == "__main__":
    main()

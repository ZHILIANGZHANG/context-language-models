"""Does T1 really test memory, and what does the model's paging respond to? (2026-11-17 diagnosis,
docs/sft_v2_diagnosis.md)

``shortcut``: accuracy of a recency reader that sees only the last k pages and assumes shelves it
has not seen are OCCUPIED (shelves fill from the bottom up). An LLM can do this; the strict rule
reader cannot (it treats unseen shelves as empty), so rule-reader "need" overstates what a model
needs. If the shortcut reaches most decisions within the window a budget keeps, the configuration
does not test memory: StreamingLLM/H2O will score like Full.

``trajectories``: for T1 evaluation trajectories of the ``model`` arm (``faithful_eval`` /
``control_eval`` ``*.trajectory.json[.gz]``), at each store decision:
  * whether the resident environment pages are sufficient for the rule reader;
  * resident share of the environment pages;
and for every model ``kv_out``, the resident fill (resident / budget) when it was issued, versus
turns without ``kv_out`` (eviction that does not track pressure is a learned habit, not paging).

    python -m self_paging.eval.t1_audit shortcut --seeds 100000 100300 [--n-decisions 4]
    python -m self_paging.eval.t1_audit trajectories --glob 'raw/faithful-baselines/**/*-model-*.trajectory.json.gz'
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import re
from collections import Counter, defaultdict
from statistics import median

from ..envs.t1_warehouse import T1Config, generate, rule_read

KINDS = ("explicit", "implicit", "delayed")


def recency_reader(texts: list[str]) -> int:
    """Lowest shelf seen EMPTY (not quarantined) in ``texts``; else one above the highest shelf seen
    occupied. Unseen shelves count as occupied."""
    status: dict[int, str] = {}
    quarantined: set[int] = set()
    for t in texts:
        for line in t.splitlines():
            if (a := re.search(r"ACTION: STORE \S+ -> S(\d+)", line)):
                status[int(a.group(1))] = "occ"
            if (d := re.search(r"S(\d+) is now EMPTY", line)):
                status[int(d.group(1))] = "empty"
            if (q := re.search(r"shelf S(\d+) QUARANTINED", line)):
                quarantined.add(int(q.group(1)))
            if (c := re.search(r"never completed;.*S(\d+) is EMPTY", line)):
                status[int(c.group(1))] = "empty"
            if "Inventory scan" in line:
                for s, st in re.findall(r"S(\d+): (EMPTY|OCCUPIED)", line):
                    status[int(s)] = "empty" if st == "EMPTY" else "occ"
    empty = [s for s, v in status.items() if v == "empty" and s not in quarantined]
    if empty:
        return min(empty)
    return max([s for s, v in status.items() if v == "occ"] or [0]) + 1


def shortcut(seeds, cfg: T1Config, windows=(2, 3, 5, 8)) -> dict:
    hit, n = Counter(), Counter()
    for seed in seeds:
        ep = generate(seed, cfg)
        for d in ep.decisions:
            n[d.kind] += 1
            for k in windows:
                hit[(d.kind, k)] += recency_reader(ep.pages[max(0, d.page + 1 - k):d.page + 1]) == d.answer
    return {kind: {f"last{k}": round(hit[(kind, k)] / n[kind], 3) for k in windows} | {"n": n[kind]}
            for kind in KINDS if n[kind]}


def _load(path):
    return json.load(gzip.open(path, "rt") if path.endswith(".gz") else open(path))


def trajectories(paths, cfg: T1Config) -> dict:
    dec = Counter()
    resident_share = defaultdict(list)
    fill_out, fill_other = defaultdict(list), defaultdict(list)
    for path in paths:
        rec = _load(path)
        seed = rec.get("seed") or int(path.split("/")[-1].split("-")[0])
        ratio = rec.get("ratio") or path.split("-model-")[-1].split(".traj")[0]
        ep = generate(int(seed), cfg)
        pid2env = {v: int(k) for k, v in rec["env_page"].items()}
        snaps = [set(s) for s in rec["snapshots"]]
        length = {p["pid"]: p["length"] for p in rec["pages"]}
        meta = rec["meta"]
        for d in ep.decisions:
            call = next((c for c in meta.get("env_calls", []) if c["name"] == "store"
                         and c["decision_page"] == d.page), None)
            if call is None:
                dec[(d.kind, "no_answer")] += 1
                continue
            res = sorted(pid2env[p] for p in snaps[call["version"]] if p in pid2env)
            ok = int(call["args"].get("shelf", -1)) == d.answer
            suff = rule_read([ep.pages[e] for e in res]) == d.answer
            dec[(d.kind, ("sufficient" if suff else "insufficient") + ("/right" if ok else "/wrong"))] += 1
            resident_share[str(ratio)].append(len(res) / (d.page + 1))
        # resident fill when each turn started (version of the first env call / kv entry of that turn)
        budget = rec.get("budget")
        if budget:
            out_turns = {e["turn"] for e in meta.get("kv_log", []) if e["op"] == "kv_out"}
            for c in meta.get("env_calls", []):
                fill = sum(length[p] for p in snaps[c["version"]]) / budget
                (fill_out if c["turn"] in out_turns else fill_other)[str(ratio)].append(fill)
    return {
        "decisions": {f"{k[0]}:{k[1]}": v for k, v in sorted(dec.items())},
        "resident_env_share": {r: round(sum(v) / len(v), 3) for r, v in sorted(resident_share.items())},
        "fill_median_turns_with_kv_out": {r: round(median(v), 3) for r, v in sorted(fill_out.items()) if v},
        "fill_median_other_turns": {r: round(median(v), 3) for r, v in sorted(fill_other.items()) if v},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("shortcut")
    a.add_argument("--seeds", type=int, nargs=2, default=[100000, 100300])
    a.add_argument("--n-decisions", type=int, default=4)
    b = sub.add_parser("trajectories")
    b.add_argument("--glob", required=True)
    b.add_argument("--n-decisions", type=int, default=4)
    args = ap.parse_args()
    cfg = T1Config(n_decisions=args.n_decisions)
    if args.cmd == "shortcut":
        out = shortcut(range(*args.seeds), cfg)
    else:
        out = trajectories(sorted(glob.glob(args.glob, recursive=True)), cfg)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

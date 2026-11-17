"""Decision-kind strata for the RL prompt pool (docs/rl_design.md §3).

SFT v1 errors concentrated on delayed decisions (a quarantine notice that takes effect many pages
later: 90/120 errors). The deployed ``rl_preparation.select_training`` balances budget ratio and
decision count only, so delayed decisions keep their natural share (~36% of decisions on the
training config sampler) and most of the gradient goes to kinds the model already solves.

This module is model-free and needs no tokenizer:

* ``decision_profile``: per decision its kind and the distance from its key event (the quarantine
  notice, the correction or the stock scan) to the decision, in events and in pages;
* ``annotate``: adds ``n_delayed``, ``n_long_delayed``, ``max_gap_pages``, ``stratum`` and the
  per-decision profile to a pool row (regenerates the world from ``seed`` + ``t1_config``);
* ``select_stratified``: the same budget quotas and decision-count balance as ``select_training``,
  plus a delayed-decision share (default ≥ 40%) and a long-gap share (default ≥ 35% of rows with a
  delayed decision whose key event is ≥ 4 pages back);
* ``select_profile_stratified``: the 64-prompt GPU profile set, balanced over (ratio, stratum).

Command line: re-select from the full CPU preparation output (``cpu-profile.jsonl``, every condition)
and annotate the evaluation splits, so evaluation can report accuracy per decision kind:

    python -m self_paging.data.rl_strata --profile $N/cpu-profile.jsonl --out-dir $N/strata
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict

from ..envs.t1_warehouse import T1Config, generate

LONG_GAP_PAGES = 4   # ≈ upper quartile of delayed key-to-decision distance on the training sampler
STRATA = ("long_delayed", "delayed", "no_delayed")
_DECISION_STEP = re.compile(r"\[step (\d+)\][^\n]*\n    DECISION REQUIRED")


def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def row_config(row: dict) -> T1Config:
    return T1Config(**{k: tuple(v) if isinstance(v, list) else v for k, v in row["t1_config"].items()
                       if k in T1Config.__dataclass_fields__})


def decision_profile(ep) -> list[dict]:
    """Kind and key-event distance of every decision of an episode."""
    out = []
    for d in ep.decisions:
        marker = f"[step {d.key_step}] "
        key_page = next(i for i, p in enumerate(ep.pages) if marker in p)
        step = int(_DECISION_STEP.findall(ep.pages[d.page])[-1])
        out.append({"kind": d.kind, "gap_steps": step - d.key_step, "gap_pages": d.page - key_page})
    return out


def stratum_of(profile: list[dict], long_gap: int = LONG_GAP_PAGES) -> str:
    delayed = [p for p in profile if p["kind"] == "delayed"]
    if any(p["gap_pages"] >= long_gap for p in delayed):
        return "long_delayed"
    return "delayed" if delayed else "no_delayed"


def annotate(row: dict, long_gap: int = LONG_GAP_PAGES, cache: dict | None = None) -> dict:
    """Return ``row`` with stratum fields (the row is not modified). ``cache`` maps world keys to
    profiles, so the four budget ratios of a world are generated once."""
    key = (row["seed"], _digest(row["t1_config"]))
    prof = cache.get(key) if cache is not None else None
    if prof is None:
        prof = decision_profile(generate(row["seed"], row_config(row)))
        if cache is not None:
            cache[key] = prof
    kinds = Counter(p["kind"] for p in prof)
    return {**row, "decision_profile": prof, "n_delayed": kinds["delayed"],
            "n_explicit": kinds["explicit"], "n_implicit": kinds["implicit"],
            "n_long_delayed": sum(p["kind"] == "delayed" and p["gap_pages"] >= long_gap for p in prof),
            "max_gap_pages": max((p["gap_pages"] for p in prof), default=0),
            "stratum": stratum_of(prof, long_gap)}


def shares(rows: list[dict]) -> dict:
    """Delayed share of decisions, long-gap share of rows, stratum and decision-count coverage."""
    dec = sum(r["n_decisions"] for r in rows)
    return {"rows": len(rows),
            "delayed_decision_share": round(sum(r["n_delayed"] for r in rows) / dec, 4) if dec else 0.0,
            "long_delayed_row_share": round(sum(r["stratum"] == "long_delayed" for r in rows) / len(rows), 4)
            if rows else 0.0,
            "strata": dict(Counter(r["stratum"] for r in rows)),
            "n_decisions": {str(k): v for k, v in sorted(Counter(r["n_decisions"] for r in rows).items())}}


def select_stratified(rows: list[dict], target: int = 5000,
                      quotas=((1.0, 0.1), (0.6, 0.3), (0.4, 0.3), (0.25, 0.3)),
                      delayed_share: float = 0.4, long_share: float = 0.35, tag: str = "select-v2"):
    """Budget quotas and decision-count balance as in ``rl_preparation.select_training``; inside each
    budget, a running greedy keeps the delayed share of decisions ≥ ``delayed_share`` and the share of
    long-gap rows ≥ ``long_share``. Rows need ``annotate`` fields. Returns (selected, report)."""
    selected, report = [], {"targets": {"delayed_decision_share": delayed_share,
                                        "long_delayed_row_share": long_share}, "by_ratio": {}, "gaps": {}}
    for ratio, proportion in quotas:
        quota = round(target * proportion)
        buckets: dict[int, list[dict]] = defaultdict(list)
        for r in rows:
            if r["split"] == "train" and r["ratio"] == ratio and r["train_eligible"]:
                buckets[r["n_decisions"]].append(r)
        for vs in buckets.values():
            vs.sort(key=lambda r: _digest([tag, r["prompt_id"]]))
        eligible = [r for vs in buckets.values() for r in vs]
        got: list[dict] = []
        stats = {"rows": 0, "long": 0, "delayed": 0, "decisions": 0}
        while len(got) < quota and any(buckets.values()):
            for key in sorted(buckets):
                if not buckets[key] or len(got) >= quota:
                    continue
                i = _pick_row(buckets[key], stats, delayed_share, long_share)
                r = buckets[key].pop(i)
                got.append(r)
                stats["rows"] += 1
                stats["long"] += r["stratum"] == "long_delayed"
                stats["delayed"] += r["n_delayed"]
                stats["decisions"] += r["n_decisions"]
        selected.extend(got)
        sel, pool = shares(got), shares(eligible)
        report["gaps"][f"{ratio:g}"] = quota - len(got)
        report["by_ratio"][f"{ratio:g}"] = {
            "selected": sel, "eligible": pool,
            "delayed_target_met": sel["delayed_decision_share"] >= delayed_share - 1e-9,
            "long_target_met": sel["long_delayed_row_share"] >= long_share - 1e-9}
    report["selected"] = shares(selected)
    return selected, report


def _pick_row(rows: list[dict], stats: dict, delayed_share: float, long_share: float) -> int:
    """Index of the next row of one decision-count bucket (rows are in a fixed hash order): a long-gap
    row while that share is short, else a row whose own delayed share is above the target while the
    running share is short (the most-delayed row if none is), else
    the next row in hash order (the natural mix)."""
    if stats["long"] < long_share * (stats["rows"] + 1):
        for i, r in enumerate(rows):
            if r["stratum"] == "long_delayed":
                return i
    if stats["delayed"] < delayed_share * (stats["decisions"] + rows[0]["n_decisions"]):
        above = next((i for i, r in enumerate(rows) if r["n_delayed"] > delayed_share * r["n_decisions"]), None)
        if above is not None:
            return above
        best = max(r["n_delayed"] for r in rows)
        return next(i for i, r in enumerate(rows) if r["n_delayed"] == best)
    return 0


def select_profile_stratified(rows: list[dict], count: int = 64, tag: str = "profile-v2") -> list[dict]:
    """Round-robin over (ratio, stratum), then decision count, so the GPU profile sees every stratum
    under every budget."""
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        buckets[(r["ratio"], r["stratum"])].append(r)
    for vs in buckets.values():
        vs.sort(key=lambda r: _digest([tag, r["prompt_id"]]))
        # interleave decision counts inside a bucket
        by_n: dict[int, list[dict]] = defaultdict(list)
        for r in vs:
            by_n[r["n_decisions"]].append(r)
        vs[:] = [r for group in _interleave([by_n[k] for k in sorted(by_n)]) for r in group]
    out: list[dict] = []
    while len(out) < count and any(buckets.values()):
        for k in sorted(buckets):
            if buckets[k] and len(out) < count:
                out.append(buckets[k].pop(0))
    if len(out) != count:
        raise ValueError(f"only {len(out)} rows for a {count}-prompt profile")
    return out


def _interleave(lists):
    lists = [list(x) for x in lists]
    while any(lists):
        yield [x.pop(0) for x in lists if x]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--profile", required=True, help="cpu-profile.jsonl from the CPU preparation (all conditions)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--target", type=int, default=5000)
    ap.add_argument("--delayed-share", type=float, default=0.4)
    ap.add_argument("--long-share", type=float, default=0.35)
    ap.add_argument("--profile-count", type=int, default=64)
    a = ap.parse_args()
    import os

    os.makedirs(a.out_dir, exist_ok=True)
    cache: dict = {}
    with open(a.profile) as f:
        rows = [annotate(json.loads(line), cache=cache) for line in f if line.strip()]
    selected, report = select_stratified(rows, a.target, delayed_share=a.delayed_share, long_share=a.long_share)
    sha = {}

    def dump(name, rs):
        path = os.path.join(a.out_dir, name)
        with open(path, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rs)
        sha[name] = hashlib.sha256(open(path, "rb").read()).hexdigest()

    dump("rl_train_pool_stratified.jsonl", selected)
    dump("profile64_stratified.jsonl", select_profile_stratified(selected, a.profile_count))
    for split in ("dev", "test", "ood"):
        dump(f"{split}_all_conditions_annotated.jsonl", [r for r in rows if r["split"] == split])
    report["eval_splits"] = {s: shares([r for r in rows if r["split"] == s]) for s in ("dev", "test", "ood")}
    report["sha256"] = sha
    report["source"] = os.path.abspath(a.profile)
    with open(os.path.join(a.out_dir, "strata-report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: report[k] for k in ("gaps", "selected")}, indent=2))


if __name__ == "__main__":
    main()

"""Write probe prompts and a separate answer key; score model responses.

    python -m state_study.probes.build_items build --out DIR --kinds explicit:raw,state ...
    python -m state_study.probes.build_items score --out DIR

``build`` writes ``DIR/items/<id>.txt`` (the prompt only) and ``DIR/key.jsonl`` (answers kept
apart from the prompts). ``score`` reads ``DIR/responses.jsonl`` -- one ``{"id", "response"}``
object per line -- and prints accuracy per kind x condition, plus how often a wrong answer is
exactly the shelf an agent would pick if it ignored the key event.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from .render import CONDITIONS, applicable, render
from .warehouse import generate

# Conditions whose prompt is the previous step's prompt plus appended text (prefix-cache safe).
CACHE_SAFE = {"raw", "tomb_tail"}


def build(out: Path, plan: dict[str, list[str]], seeds: list[int], gap: int, warmup: int) -> None:
    (out / "items").mkdir(parents=True, exist_ok=True)
    with (out / "key.jsonl").open("w") as key:
        for kind, conds in plan.items():
            for seed in seeds:
                sc = generate(kind, seed=seed, gap=gap, warmup=warmup)
                for cond in conds:
                    if cond not in CONDITIONS or not applicable(sc, cond):
                        raise SystemExit(f"condition {cond!r} not applicable to {kind}")
                    item_id = f"{kind}-s{seed}-{cond}"
                    prompt = render(sc, cond)
                    (out / "items" / f"{item_id}.txt").write_text(prompt)
                    key.write(json.dumps({
                        "id": item_id, "kind": kind, "seed": seed, "condition": cond,
                        "gap": gap, "answer": sc.answer, "deaf_answer": sc.deaf_answer,
                        "est_tokens": len(prompt) // 4, "cache_safe": cond in CACHE_SAFE,
                    }) + "\n")


def score(out: Path) -> dict:
    key = {r["id"]: r for r in map(json.loads, (out / "key.jsonl").read_text().splitlines())}
    cells: dict[tuple, list] = defaultdict(list)
    for line in (out / "responses.jsonl").read_text().splitlines():
        r = json.loads(line)
        k = key[r["id"]]
        m = re.search(r'"shelf"\s*:\s*"S(\d+)"', r["response"])
        got = int(m.group(1)) if m else None
        cells[(k["kind"], k["condition"])].append(
            {"correct": got == k["answer"], "deaf": got == k["deaf_answer"], "parsed": got is not None})
    table = {}
    for (kind, cond), rows in sorted(cells.items()):
        n = len(rows)
        table[f"{kind}/{cond}"] = {
            "n": n,
            "correct": sum(r["correct"] for r in rows),
            "deaf_errors": sum(r["deaf"] for r in rows),
            "unparsed": sum(not r["parsed"] for r in rows),
        }
    return table


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--plan", nargs="+", required=True,
                   help="kind:cond1,cond2 ...  e.g. explicit:raw,tomb_tail,state")
    b.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    b.add_argument("--gap", type=int, default=40)
    b.add_argument("--warmup", type=int, default=60)
    s = sub.add_parser("score")
    s.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "build":
        plan = {p.split(":")[0]: p.split(":")[1].split(",") for p in args.plan}
        build(args.out, plan, args.seeds, args.gap, args.warmup)
    else:
        for cell, v in score(args.out).items():
            print(f"{cell:28s} {v['correct']}/{v['n']} correct, "
                  f"{v['deaf_errors']} deaf-errors, {v['unparsed']} unparsed")


if __name__ == "__main__":
    main()

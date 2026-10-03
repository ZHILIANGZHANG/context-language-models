import json, re, sys, glob
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from vdepth import generate, render

D = Path(sys.argv[1])
bm = json.loads((D / "blind_map.json").read_text())
cells = defaultdict(list)
for f in sorted(glob.glob(str(D / "responses" / "*.jsonl"))):
    for line in open(f):
        r = json.loads(line)
        meta = bm[r["blind"]]
        sc = generate(meta["seed"], meta["f"], **meta["kw"])
        p = render(sc, "raw")
        got = re.findall(r'"shelf"\s*:\s*"S(\d+)"', r["response"])
        got = int(got[-1]) if got else None
        lee = min(int(x) for x in re.findall(r"S(\d+) is now EMPTY", p))
        mre = int(re.findall(r"S(\d+) is now EMPTY", p)[-1])
        if got == sc.answer: lab = "correct"
        elif got is None: lab = "unparsed"
        elif got < sc.answer: lab = "superseded_low"   # a lower shelf that is occupied now (it was EMPTY in an old version)
        elif got == mre: lab = "most_recent_empty"
        elif sc.shelves.get(got) is None: lab = "missed_lower"
        else: lab = "occupied_high"
        cells[(r["model"], meta["cell"])].append((meta["seed"], lab))
order = ["raw_f0", "raw_f1", "raw_f3", "raw_f6", "raw_f12", "tomb_f12", "state_f12", "raw_f0_noirr", "raw_f3_tele2x"]
for model in sorted({m for m, _ in cells}):
    print(f"== {model}")
    for c in order:
        rows = cells.get((model, c))
        if not rows: continue
        n = len(rows); k = sum(l == "correct" for _, l in rows)
        errs = defaultdict(int)
        for _, l in rows:
            if l != "correct": errs[l] += 1
        print(f"   {c:14s} {k}/{n} correct   errors: {dict(errs) or '-'}")

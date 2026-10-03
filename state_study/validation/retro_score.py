import json, re, sys, glob
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from retro import generate
D = Path(sys.argv[1]); bm = json.loads((D / "blind_map.json").read_text())
cells = defaultdict(list)
for f in sorted(glob.glob(str(D / "responses" / "*.jsonl"))):
    for line in open(f):
        r = json.loads(line); m = bm[r["blind"]]
        sc = generate(m["seed"], m["form"], m["thoughts"])
        got = re.findall(r'"shelf"\s*:\s*"S(\d+)"', r["response"]); got = int(got[-1]) if got else None
        lab = ("correct" if got == sc["answer"] else "unparsed" if got is None else
               "stale(ignored the update)" if got == sc["deaf"] else
               "occupied" if sc["status"].get(got) else "other")
        cells[(r["model"], m["cell"])].append(lab)
order = ["fwd_none_raw", "fwd_thought_raw", "retro_none_raw", "retro_thought_raw", "retro_thought_tomb",
         "retro_thought_state", "retro_thought_logstate"]
for model in sorted({k[0] for k in cells}):
    print("==", model)
    for c in order:
        rows = cells.get((model, c))
        if rows:
            err = defaultdict(int)
            for l in rows:
                if l != "correct": err[l] += 1
            print(f"   {c:24s} {rows.count('correct')}/{len(rows)}  errors: {dict(err) or '-'}")

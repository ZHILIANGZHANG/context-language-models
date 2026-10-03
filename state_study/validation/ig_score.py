"""Score step B of the integration-gate pilot: python ig_score.py OUT
correct = the next pallet goes to the shelf the correction emptied; deaf = the shelf an agent that
ignored the correction would pick."""
import collections, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import ig

out = Path(sys.argv[1])
bm = json.loads((out / "blind_map_B.json").read_text())
res = collections.defaultdict(list)
for b, it in sorted(bm.items(), key=lambda x: (x[1]["cell"], x[1]["seed"])):
    f = out / "outB" / f"{b}.txt"
    if not f.exists():
        continue
    sc = ig.generate(it["seed"])
    s = ig.store_shelf(f.read_text())
    res[it["cell"]].append("ok" if s == sc["answer"] else ("deaf" if s == sc["deaf"] else f"other:{s}"))
for cell, r in sorted(res.items()):
    c = collections.Counter(r)
    print(f"{cell:24s} n={len(r)}  correct={c['ok']}/{len(r)}  deaf={c['deaf']}  other={[x for x in r if x.startswith('other')]}")

"""Render blinded items for the version-depth pilot. The answer key is not written next to the items;
it is regenerated at scoring time from (seed, f, cond, variant)."""
import hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from vdepth import generate, render

OUT = Path(sys.argv[1]); SALT = sys.argv[2]
CELLS = [  # (cell, f, cond, kwargs)
    ("raw_f0", 0, "raw", {}), ("raw_f1", 1, "raw", {}), ("raw_f3", 3, "raw", {}),
    ("raw_f6", 6, "raw", {}), ("raw_f12", 12, "raw", {}),
    ("tomb_f12", 12, "tomb", {}), ("state_f12", 12, "state", {}),
    ("raw_f0_noirr", 0, "raw", {"irrelevant": False}),
    ("raw_f3_tele2x", 3, "raw", {"tele_mult": 2.0}),
]
(OUT / "items").mkdir(parents=True, exist_ok=True)
blind = {}
for cell, f, cond, kw in CELLS:
    for seed in range(8):
        iid = f"{cell}-s{seed}"
        b = hashlib.sha1((SALT + iid).encode()).hexdigest()[:10]
        p = render(generate(seed, f, **kw), cond)
        (OUT / "items" / f"{b}.txt").write_text(p)
        blind[b] = {"id": iid, "cell": cell, "f": f, "cond": cond, "kw": kw, "seed": seed, "tokens": len(p) // 4}
(OUT / "blind_map.json").write_text(json.dumps(blind, indent=1))
for cell, *_ in CELLS:
    t = [v["tokens"] for v in blind.values() if v["cell"] == cell]
    print(f"{cell:14s} items={len(t)} tokens~{sum(t)//len(t)}")

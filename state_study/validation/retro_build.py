import hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from retro import generate, render
OUT = Path(sys.argv[1]); SALT = sys.argv[2]
CELLS = [("fwd_none_raw", "forward", False, "raw"), ("fwd_thought_raw", "forward", True, "raw"),
         ("retro_none_raw", "retro", False, "raw"), ("retro_thought_raw", "retro", True, "raw"),
         ("retro_thought_tomb", "retro", True, "tomb"), ("retro_thought_state", "retro", True, "state"),
         ("retro_thought_logstate", "retro", True, "log_state")]
(OUT / "items").mkdir(parents=True, exist_ok=True)
bm = {}
for cell, form, th, cond in CELLS:
    for seed in range(8):
        iid = f"{cell}-s{seed}"; b = hashlib.sha1((SALT + iid).encode()).hexdigest()[:10]
        (OUT / "items" / f"{b}.txt").write_text(render(generate(seed, form, th), cond))
        bm[b] = {"id": iid, "cell": cell, "form": form, "thoughts": th, "cond": cond, "seed": seed}
(OUT / "blind_map.json").write_text(json.dumps(bm, indent=1))
print(len(bm), "items")

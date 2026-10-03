import hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from echo import generate, render
OUT = Path(sys.argv[1]); SALT = sys.argv[2]
CELLS = [("retro_k0_raw", "retro", 0, True, "raw"), ("retro_k3_thought_raw", "retro", 3, True, "raw"),
         ("retro_k8_thought_raw", "retro", 8, True, "raw"), ("retro_k8_actions_raw", "retro", 8, False, "raw"),
         ("retro_k8_thought_tomb", "retro", 8, True, "tomb"), ("retro_k8_thought_logstate", "retro", 8, True, "log_state"),
         ("retro_k8_thought_state", "retro", 8, True, "state"), ("fwd_k8_thought_raw", "forward", 8, True, "raw")]
(OUT / "items").mkdir(parents=True, exist_ok=True)
bm = {}
for cell, form, k, th, cond in CELLS:
    for seed in range(8):
        iid = f"{cell}-s{seed}"; b = hashlib.sha1((SALT + iid).encode()).hexdigest()[:10]
        (OUT / "items" / f"{b}.txt").write_text(render(generate(seed, form, k, th), cond))
        bm[b] = {"id": iid, "cell": cell, "form": form, "k": k, "thoughts": th, "cond": cond, "seed": seed}
(OUT / "blind_map.json").write_text(json.dumps(bm, indent=1))
print(len(bm))

"""Integration-gate pilot (two-call closed loop, see ig.py).

    python ig_build.py a OUT     # step A prompts for all cells -> OUT/A, OUT/blind_map.json
    python ig_build.py b OUT     # step B prompts from OUT/outA replies -> OUT/B, OUT/blind_map_B.json

Blind ids are sha1(salt + item id)[:10]; the key is regenerated from the seed at scoring time."""
import hashlib, json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import ig

SALT_A, SALT_B = "igsalt", "igB-7f3d"
# (cell, representation, notice covered by the spec, integrate sentence, field-filter sentence)
CELLS = [("H_covered", "H", True, False, False), ("H_uncovered", "H", False, False, False),
         ("H_uncovered_integrate", "H", False, True, False), ("S_covered", "S", True, False, False),
         ("S_uncovered", "S", False, False, False),
         ("Hf_uncovered", "H", False, False, True), ("Sf_uncovered", "S", False, False, True),
         ("Hf_uncovered_integrate", "H", False, True, True), ("Hf_covered", "H", True, False, True)]


def build_a(out: Path):
    (out / "A").mkdir(parents=True, exist_ok=True)
    bm = {}
    for cell, rep, cov, integ, filt in CELLS:
        for seed in range(8):
            iid = f"{cell}-s{seed}"
            b = hashlib.sha1((SALT_A + iid).encode()).hexdigest()[:10]
            (out / "A" / f"{b}.txt").write_text(ig.prompt_A(ig.generate(seed), rep, cov, integ, filt=filt))
            bm[b] = {"id": iid, "cell": cell, "rep": rep, "covered": cov, "integrate": integ, "filt": filt, "seed": seed}
    (out / "blind_map.json").write_text(json.dumps(bm, indent=1))
    print(len(bm), "step-A items")


def norm_a(text: str, rep: str) -> str:
    """Fit a step-A reply into the transcript: drop an echoed observation and a leading
    'Reasoning & Action:' label, and put the reply on one line like the earlier steps."""
    if rep != "H":
        return text
    t = text.strip()
    k = t.find("Reasoning:")
    if k > 0:
        t = t[k:]
    t = re.sub(r"^Reasoning & Action:\s*", "", t)
    return re.sub(r"\s+", " ", t).strip()


def build_b(out: Path):
    bm = json.loads((out / "blind_map.json").read_text())
    (out / "B").mkdir(exist_ok=True)
    bmB = {}
    for a_id, it in sorted(bm.items(), key=lambda x: (x[1]["cell"], x[1]["seed"])):
        reply = out / "outA" / f"{a_id}.txt"
        if not reply.exists():
            continue
        b_id = hashlib.sha1((SALT_B + a_id).encode()).hexdigest()[:10]
        p = ig.prompt_B(ig.generate(it["seed"]), it["rep"], it["covered"], norm_a(reply.read_text(), it["rep"]),
                        it["integrate"], it.get("filt", False))
        (out / "B" / f"{b_id}.txt").write_text(p)
        bmB[b_id] = dict(it, a_id=a_id)
    (out / "blind_map_B.json").write_text(json.dumps(bmB, indent=1))
    print(len(bmB), "step-B items")


if __name__ == "__main__":
    {"a": build_a, "b": build_b}[sys.argv[1]](Path(sys.argv[2]))

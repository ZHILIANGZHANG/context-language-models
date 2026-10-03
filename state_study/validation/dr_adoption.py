"""After an ACTION REJECTED observation that names shelf s, does the model's next state patch fix s?"""
import glob, json, re, sys
from pathlib import Path
from collections import Counter
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "third_party" / "references" / "delayed-relevance" / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from dr_persistence import load, belief, reality, kind
from dr.envs.warehouse import Warehouse
from dr.runner import NO_OP
from dr.runtimes.skillstate import SkillStateRuntime, _merge_into
from dr.types import Action

def analyse(pattern):
    c = Counter(); origins = Counter(); examples = []
    for f in sorted(glob.glob(pattern)):
        head, steps = load(f)
        env = Warehouse(horizon=len(steps), seed=head["seed"], apendice_b=head.get("apendice_b", False),
                        sin_telemetria=head.get("sin_telemetria", False), ruido=head.get("ruido", 0))
        env.reset(); fields = set(env.schema_fields()); state = {}
        prev_disc = set()
        for p in steps:
            obs = env.observe().text
            b, r = belief(state), reality(env)
            disc_before = {s for s in r if kind(b.get(s), r[s])}
            m = re.search(r"ACTION REJECTED: shelf (\d+)", obs)
            rejected = int(m.group(1)) if m else None
            for resp in p["raw"]["respuestas"]:
                parsed = SkillStateRuntime._parse(resp)
                if parsed is not None and all(k in fields for k in parsed[0]):
                    _merge_into(state, parsed[0], deep=True); break
            b_after = belief(state)
            if rejected is not None:
                was_wrong = kind(b.get(rejected), r[rejected]) is not None
                now_wrong = kind(b_after.get(rejected), r[rejected]) is not None
                c["rejections"] += 1
                c["rejections_on_wrong_belief"] += was_wrong
                c["fixed_by_next_patch"] += was_wrong and not now_wrong
                if was_wrong and now_wrong and len(examples) < 3:
                    examples.append((f.split('/')[-1], p["step"], obs.split("\n")[0][:110], p["ejecutado"]))
            act = Action.parse(p["ejecutado"]) if p.get("ejecutado") else None
            env.apply(act if act is not None else NO_OP)
            b2, r2 = belief(state), reality(env)
            new = {s for s in r2 if kind(b2.get(s), r2[s])} - disc_before
            for s in new:
                origins["action_correct_patch_wrong" if p.get("correct") else "action_wrong"] += 1
    return c, origins, examples

for label, pat in [("Haiku T=200 SKILL.state", "adj_T200_claude-haiku-4-5_skillstate_s*.jsonl"),
                   ("Sonnet L2 SKILL.state", "l2v2_claude-sonnet-5_skillstate_*.jsonl")]:
    c, o, ex = analyse(f"{sys.argv[1]}/results/{pat}")
    print(label, dict(c), "| how wrong beliefs start:", dict(o))
    for e in ex:
        print("   unfixed after rejection:", e)

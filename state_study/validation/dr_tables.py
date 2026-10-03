"""Tables from the delayed-relevance traces, recomputed from the per-step records.

    python dr_tables.py T        # T = results/ directory extracted from the delayed-relevance traces (see README)

1. L2 (retroactive correction, k=10): first decision that depends on the correction, per model and
   runtime, and how often the ReAct reply at the notice step calls the event undefined/out of scope.
2. L1 v3 (quarantine notice 40 steps before the decision it affects): first decision per model,
   variant (plain / reminder / oracle slot / free-form hatch slot) and runtime.
3. L1 hatch runs: did the state patch at the notice step write the quarantine down (heuristic regex),
   and was the decision right?"""
import ast, collections, glob, json, re, sys
from pathlib import Path

T = Path(sys.argv[1])
UNDEF = re.compile(r"undefined|not (?:a )?(?:defined|listed|recogni|covered|one of)|unknown event|"
                   r"not in the procedure|no procedure", re.I)


def load(f):
    rows = [json.loads(l) for l in open(f) if l.strip()]
    return [r for r in rows if r.get("kind") != "run_header"]


def replies(r):
    raw = r["raw"]
    if isinstance(raw, str):
        raw = ast.literal_eval(raw)
    return raw["respuestas"]


print("== L2: first correction-dependent decision correct / episodes  (notice called out of scope)")
for model in ("claude-haiku-4-5", "claude-sonnet-5", "gemini-3-flash-preview"):
    for rt in ("react", "skillstate"):
        n = ok = und = 0
        for f in sorted(glob.glob(str(T / f"l2v2_{model}_{rt}_s*_r*_k10_mt8192.jsonl"))):
            st = load(f)
            c = next(r for r in st if "correction_notice" in (r.get("observation") or ""))
            dep = [r for r in st if r.get("dependiente")]
            n += 1
            ok += bool(dep and dep[0]["correct"])
            und += bool(UNDEF.search(replies(c)[-1]))
        print(f"   {model:24s} {rt:10s} {ok:2d}/{n}   ({und} out of scope)")

print("\n== L1 v3 (k=40): first decision correct / episodes")
cells = collections.defaultdict(lambda: [0, 0])
for f in glob.glob(str(T / "l1v3_T50_*_estricto_v3_*_k40_s*_r*.jsonl")):
    m = re.match(r"l1v3_T50_(.+?)_(?:(reminder|oracle|hatch)_)?mt8192(?:_tb0)?_estricto_v3_(react|skillstate)_k40",
                 Path(f).name)
    model, var, rt = m.group(1), m.group(2) or "plain", m.group(3)
    d = [r for r in load(f) if r.get("es_el_paso")]
    if d:
        cells[(model, var, rt)][0] += bool(d[0]["correct"])
        cells[(model, var, rt)][1] += 1
for k in sorted(cells):
    print(f"   {k[0]:24s} {k[1]:9s} {k[2]:10s} {cells[k][0]:2d}/{cells[k][1]}")

print("\n== L1 hatch runs: (wrote the quarantine down at the notice step, decision correct) -> episodes")
NOTE_KEY = re.compile(r'"(?:notes|hatch|other|constraints|quarantine\w*|restrict\w*|blocked\w*|unavailable\w*)"', re.I)
for model in ("claude-haiku-4-5", "claude-sonnet-5"):
    for var in ("hatch_", "oracle_", ""):
        tab = collections.Counter()
        for f in sorted(glob.glob(str(T / f"l1v3_T50_{model}_{var}mt8192*_estricto_v3_skillstate_k40_s*_r*.jsonl"))):
            st = load(f)
            n = next(r for r in st if "facility_notice" in r["observation"])
            d = next(r for r in st if r.get("es_el_paso"))
            txt = " ".join(replies(n))
            wrote = bool(re.search(r"quarantin|do_not_store|maintenance", txt, re.I) and NOTE_KEY.search(txt))
            tab[(wrote, bool(d["correct"]))] += 1
        print(f"   {model:24s} {var or 'plain_':8s} {dict(tab)}")

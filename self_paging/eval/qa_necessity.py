"""Model-based necessity check for QA page streams (docs/real_data_sft.md): a question enters the SFT
data only if the evidence pages are really needed. Runs Qwen3-8B (the base model the student starts
from) greedily on three views of the stream at the moment the question can be answered:

* ``closed``: the question alone;
* ``without``: every page delivered so far except the evidence pages, then the question;
* ``with``: every page delivered so far, then the question (the Full view).

Kept (``necessary``) when F1(with) >= ``--keep`` and F1(without) < ``--drop`` and
F1(closed) < ``--drop``: the answer is reachable with the evidence and not without it, so a correct
kv_in is required whenever the evidence is not resident. Answers are scored with SQuAD-style token
F1 against all gold answers.

    python -m self_paging.eval.qa_necessity --model $M --tokenizer $M --streams streams.jsonl \\
        --out necessity.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter

import torch

from ..core.protocol import IM_END, IM_START, NO_THINK

INSTRUCTION = ("Answer the question using the passages above. Reply with the answer only, as a short "
               "phrase, with no explanation.")


def _norm(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1(pred: str, golds: list[str]) -> float:
    best = 0.0
    p = _norm(pred).split()
    for g in golds:
        t = _norm(g).split()
        common = Counter(p) & Counter(t)
        same = sum(common.values())
        if not p or not t:
            best = max(best, float(p == t))
            continue
        if same:
            prec, rec = same / len(p), same / len(t)
            best = max(best, 2 * prec * rec / (prec + rec))
    return best


def prompt(pages: list[str], question: str) -> str:
    body = "\n\n".join(f"[Passage {i + 1}]\n{t}" for i, t in enumerate(pages))
    user = (body + "\n\n" if body else "") + f"Question: {question}\n{INSTRUCTION}"
    return f"{IM_START}user\n{user}{IM_END}\n{IM_START}assistant\n{NO_THINK}"


def views(s: dict, q: dict) -> dict[str, list[str]]:
    ready = q.get("checks", {}).get("ready")
    if ready is None:
        ready = max(q["evidence_pages"]) if q["mode"] == "first" else q["ask_after"]
    seen = list(range(ready + 1))
    ev = set(q["evidence_pages"])
    texts = [p["text"] for p in s["pages"]]
    return {"closed": [], "without": [texts[p] for p in seen if p not in ev], "with": [texts[p] for p in seen]}


@torch.no_grad()
def answer(model, tok, text: str, max_new: int = 32) -> str:
    ids = tok.encode(text)
    device = next(model.parameters()).device
    out = model.generate(torch.tensor([ids], device=device), attention_mask=torch.ones(1, len(ids), device=device),
                         max_new_tokens=max_new, do_sample=False, pad_token_id=tok.im_end)
    gen = out[0, len(ids):].tolist()
    if tok.im_end in gen:
        gen = gen[:gen.index(tok.im_end)]
    return tok.decode(gen).strip()


def check(model, tok, s: dict, keep: float = 0.5, drop: float = 0.3, max_new: int = 32) -> list[dict]:
    rows = []
    for q in s["questions"]:
        preds = {k: answer(model, tok, prompt(v, q["question"]), max_new) for k, v in views(s, q).items()}
        scores = {k: f1(p, q["answers"]) for k, p in preds.items()}
        rows.append(dict(stream_id=s["stream_id"], qid=q["qid"], source=q["source"], mode=q["mode"],
                         preds=preds, f1=scores,
                         necessary=scores["with"] >= keep and scores["without"] < drop and scores["closed"] < drop))
    return rows


def main() -> None:
    from ..train.common import load_model, load_tokenizer

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--streams", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep", type=float, default=0.5)
    ap.add_argument("--drop", type=float, default=0.3)
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--shard", type=int, nargs=2, default=[0, 1], metavar=("I", "N"), help="process streams i::n")
    a = ap.parse_args()
    tok = load_tokenizer(a.tokenizer)
    model = load_model(a.model, "sdpa", device="cuda" if torch.cuda.is_available() else "cpu").eval()
    done = set()
    try:
        done = {(r["stream_id"], r["qid"]) for r in map(json.loads, open(a.out))}
    except FileNotFoundError:
        pass
    c = Counter()
    with open(a.streams) as f, open(a.out, "a") as out:
        for i, line in enumerate(f):
            if i % a.shard[1] != a.shard[0]:
                continue
            s = json.loads(line)
            if all((s["stream_id"], q["qid"]) in done for q in s["questions"]):
                continue
            for r in check(model, tok, s, a.keep, a.drop, a.max_new):
                out.write(json.dumps(r) + "\n")
                c["questions"] += 1
                c["necessary"] += r["necessary"]
                c[f"necessary/{r['source']}"] += r["necessary"]
                c[f"n/{r['source']}"] += 1
            out.flush()
    print(json.dumps(dict(c), indent=2))


if __name__ == "__main__":
    main()

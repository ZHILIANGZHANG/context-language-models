"""Controlled E2 replay: identical visible IDs AND RoPE positions across OUT/REWRITE.

This is a teacher-forced context probe, not an agent success-rate benchmark. The
warehouse case comes from the same T1 generator. Neutral telemetry is inserted
after the correction to vary residue age without changing the correct state.
REWRITE recomputes all surviving KV at original positions: compacting positions
would confound residue with RoPE. No sampled assistant reasoning is inserted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re

import torch

from ..core.protocol import IM_START, IM_END, assistant_prefix
from ..envs.t1_warehouse import T1Config, generate, rule_read
from ..hf.logstore import PagedKV
from ..train.common import append_jsonl, load_model, load_tokenizer
from state_study.probes.warehouse import RULE


def make_case(seed, tok, dose=1024):
    cfg = T1Config(n_decisions=1, kinds=("explicit",), gap=(1, 1), events_per_page=(1, 1))
    ep = generate(seed, cfg)
    if len(ep.decisions) != 1:
        raise ValueError("Expected exactly one explicit decision")
    decision = ep.decisions[0]
    pages = ep.pages[:decision.page + 1]
    b = next(i for i, text in enumerate(pages) if "Audit correction" in text)
    step = int(re.search(r"storage recorded at step (\d+)", pages[b]).group(1))
    a = next(i for i, text in enumerate(pages) if text.startswith(f"[step {step}]"))
    assert a < b < decision.page
    correct = decision.answer
    stale = rule_read([p for i, p in enumerate(pages) if i != b])
    assert stale != correct
    assert rule_read([p for i, p in enumerate(pages) if i != a]) == correct
    # One user context with fixed page delimiters. Labels and all non-a tokens
    # are retained even when a is removed. Tokenization happens only once.
    prefix = tok.encode(f"{IM_START}system\n{RULE}{IM_END}\n{IM_START}user\n"
                        "Read the log below. At the decision, output only the shelf number.\n")
    chunks = [prefix] + [tok.encode(f"\n[warehouse page {i}]\n{p}\n") for i, p in enumerate(pages)]
    chunks[-1].extend(tok.encode(f"{IM_END}\n"))
    a, b = a + 1, b + 1
    # Dose is the exact gap from the end of old-record page a to the start
    # of the decision page, excluding the decision itself and answer prefix.
    # Neutral telemetry never stores or dispatches an item.
    base_dose = sum(map(len, chunks[a + 1:-1]))
    if base_dose > dose:
        raise ValueError("Requested dose shorter than the fixed correction/decision context")
    filler = tok.encode("Telemetry: dock temperature stable, conveyor online.\n")
    count = dose - base_dose
    if count:
        chunks.insert(b + 1, (filler * (count // len(filler) + 1))[:count])
    chunks.append(tok.encode(assistant_prefix(False)))
    starts, n = [], 0
    for ids in chunks:
        starts.append(n)
        n += len(ids)
    if n > 32768:
        raise ValueError("E2 exceeded the registered 32K global-position limit")
    assert starts[-2] - starts[a] - len(chunks[a]) == dose
    return dict(seed=seed, chunks=chunks, starts=starts, a=a, b=b,
                decision=len(chunks) - 2, correct=correct, stale=stale,
                dose=dose, dose_definition="end_of_a_to_start_of_decision", prompt_tokens=n)


class PositionedModel(torch.nn.Module):
    """Probe-only wrapper. PagedKV metadata is compact, but model positions aren't."""
    def __init__(self, model):
        super().__init__()
        self.base, self.config, self.start = model, model.config, 0

    def forward(self, **kw):
        n = kw["input_ids"].shape[1]
        kw["position_ids"] = torch.arange(self.start, self.start + n,
                                          device=kw["input_ids"].device)[None]
        self.start += n
        return self.base(**kw)


@torch.no_grad()
def replay(model, case, condition):
    if condition not in ("FULL", "OUT-LATE", "OUT-EARLY", "REWRITE"):
        raise ValueError(condition)
    wrapper = PositionedModel(model)
    kv = PagedKV(wrapper)
    pids = {}
    visible, positions = [], []
    for i, ids in enumerate(case["chunks"]):
        if condition == "REWRITE" and i == case["a"]:
            continue
        if condition == "OUT-LATE" and i == case["decision"]:
            kv.page_out([pids[case["a"]]])
        wrapper.start = case["starts"][i]
        pid = kv.open("tool")
        pids[i] = pid
        logits = kv.feed(pid, ids)[-1]
        if condition == "OUT-EARLY" and i == case["b"]:
            kv.page_out([pids[case["a"]]])
        if condition == "FULL" or i != case["a"]:
            visible.extend(ids)
            positions.extend(range(case["starts"][i], case["starts"][i] + len(ids)))
    return kv, wrapper, logits, visible, positions


@torch.no_grad()
def measure(model, tok, case, condition):
    kv, wrapper, logits, ids, positions = replay(model, case, condition)
    # Compare whole answer-sequence logprobs, not the first digit alone.
    scores = {}
    for name in ("correct", "stale"):
        answer_ids = tok.encode(str(case[name]))
        # Independent replay keeps candidate continuation KV from leaking into
        # the next candidate's probability calculation.
        if name == "stale":
            kv, wrapper, logits, _, _ = replay(model, case, condition)
        score = 0.
        p = kv.open("assistant")
        for token in answer_ids:
            score += float(torch.log_softmax(logits.float(), -1)[token])
            logits = kv.feed(p, [token])[-1]
        scores[name + "_logprob"] = score
    kv, wrapper, logits, _, _ = replay(model, case, condition)
    p = kv.open("assistant")
    output = []
    for _ in range(16):
        token = int(logits.argmax())
        if token == tok.im_end:
            break
        output.append(token)
        logits = kv.feed(p, [token])[-1]
    raw = tok.decode(output).strip()
    match = re.fullmatch(r"(?:S)?(\d+)[.!]?", raw)
    pred = int(match.group(1)) if match else None
    return dict(seed=case["seed"], dose=case["dose"], condition=condition,
                correct=case["correct"], stale=case["stale"], prediction=pred, raw_output=raw,
                accuracy=pred == case["correct"], stale_answer=pred == case["stale"],
                prompt_tokens=case["prompt_tokens"], probe="controlled_teacher_forced_context",
                dose_definition=case["dose_definition"],
                positions="original_global_positions_including_REWRITE",
                visible_sha256=hashlib.sha256(json.dumps([ids, positions]).encode()).hexdigest(), **scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", nargs=2, type=int, default=[100000, 100020])
    ap.add_argument("--doses", nargs="+", type=int, default=[1024, 4096, 16384])
    a = ap.parse_args()
    tok = load_tokenizer(a.model)
    model = load_model(a.model, "sdpa", device="cuda").eval()
    for seed in range(*a.seeds):
        for dose in a.doses:
            case = make_case(seed, tok, dose)
            rows = [measure(model, tok, case, c) for c in ("FULL", "OUT-LATE", "OUT-EARLY", "REWRITE")]
            assert len({r["visible_sha256"] for r in rows[1:]}) == 1
            for row in rows:
                append_jsonl(a.out, row)


if __name__ == "__main__":
    main()

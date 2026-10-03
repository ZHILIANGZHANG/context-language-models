"""Rule-reader tests for the 2026-10-03 validation probes (state_study/validation).

Each probe must be answerable from its text alone: a reader that only parses the rendered prompt
recovers the key in every condition. Otherwise a model's error could not be blamed on the model.
"""
import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validation"))

import echo  # noqa: E402
import ig  # noqa: E402
import retro  # noqa: E402
import vdepth  # noqa: E402


def vdepth_reader(prompt):
    if "CURRENT STATE:" in prompt:
        st = json.loads(re.search(r"CURRENT STATE:\n(\{.*?\n\})", prompt, re.S).group(1))
        return min(int(k[1:]) for k, v in st["shelves"].items() if v == "EMPTY")
    body = prompt.split("=== CONTEXT ===")[1].split("=== END CONTEXT ===")[0]
    status = {}
    for line in body.splitlines():
        if m := re.search(r"ACTION: STORE \S+ -> S(\d+)", line):
            status[int(m.group(1))] = "O"
        if m := re.search(r"S(\d+) is now EMPTY", line):
            status[int(m.group(1))] = "E"
        if m := re.search(r"S(\d+) is OCCUPIED again", line):
            status[int(m.group(1))] = "O"
        if m := re.search(r"S(\d+) verified, holds", line):
            status[int(m.group(1))] = "O"
    return min(s for s, v in status.items() if v == "E")


def retro_reader(p):
    if "CURRENT STATE:" in p and "HISTORY LOG" not in p:
        st = json.loads(re.search(r"CURRENT STATE:\n(\{.*?\n\})", p, re.S).group(1))
        return min(int(k[1:]) for k, v in st["shelves"].items() if v == "EMPTY")
    body = p.split("=== CONTEXT ===")[1].split("=== END CONTEXT ===")[0].split("CURRENT STATE:")[0]
    occ = set()
    for line in body.splitlines():
        if m := re.search(r"ACTION: STORE \S+ -> S(\d+)", line):
            occ.add(int(m.group(1)))
        if m := re.search(r"S(\d+) is now EMPTY", line):
            occ.discard(int(m.group(1)))
        if m := re.search(r"never completed;.*S(\d+) is EMPTY", line):
            occ.discard(int(m.group(1)))
    return next(i for i in range(1, 300) if i not in occ)


@pytest.mark.parametrize("f", [0, 1, 3, 6, 12])
def test_vdepth_rule_reader(f):
    for seed in range(20):
        sc = vdepth.generate(seed, f)
        for cond in ("raw", "tomb", "latest", "state"):
            assert vdepth_reader(vdepth.render(sc, cond)) == sc.answer, (f, seed, cond)


def test_vdepth_paired_across_depth():
    # same seed -> same answer and same event count whatever the number of superseded versions
    for seed in range(10):
        assert len({vdepth.generate(seed, f).answer for f in (0, 3, 12)}) == 1
        assert len({vdepth.generate(seed, f).meta["n_events"] for f in (0, 3, 12)}) == 1


@pytest.mark.parametrize("form", ["forward", "retro"])
@pytest.mark.parametrize("thoughts", [False, True])
def test_retro_rule_reader(form, thoughts):
    for seed in range(15):
        sc = retro.generate(seed, form, thoughts)
        for cond in ("raw", "tomb", "state", "log_state"):
            assert retro_reader(retro.render(sc, cond)) == sc["answer"], (form, thoughts, seed, cond)


@pytest.mark.parametrize("form", ["forward", "retro"])
@pytest.mark.parametrize("k", [0, 3, 8])
def test_echo_rule_reader(form, k):
    for seed in range(10):
        for thoughts in (False, True):
            sc = echo.generate(seed, form, k, thoughts)
            for cond in ("raw", "tomb", "state", "log_state"):
                assert retro_reader(echo.render(sc, cond)) == sc["answer"], (form, k, thoughts, seed, cond)


def ig_history_reader(prompt):
    """Replay the transcript's Store/Ship actions and correction notices; return the lowest empty shelf."""
    occ = set()
    for line in prompt.splitlines():
        for s in re.findall(r'Store\(\{"shelf": (\d+)', line):
            occ.add(int(s))
        for s in re.findall(r'Ship\(\{"shelf": (\d+)', line):
            occ.discard(int(s))
        if m := re.search(r"correction_notice .*\| shelf=(\d+) \|.*shelf_is_now=empty", line):
            occ.discard(int(m.group(1)))
    return min(i for i in range(100) if i not in occ)


def test_ig_answer_is_the_corrected_shelf_and_differs_from_the_deaf_answer():
    for seed in range(40):
        sc = ig.generate(seed)
        assert sc["answer"] == sc["vs"], seed
        assert sc["deaf"] != sc["answer"], seed


@pytest.mark.parametrize("filt", [False, True])
def test_ig_rule_reader_history_and_state(filt):
    for seed in range(40):
        sc = ig.generate(seed)
        h = ig.prompt_B(sc, "H", False, "Reasoning: noted. Action: Wait({})", filt=filt)
        assert ig_history_reader(h.split("Latest Observation:")[0]) == sc["answer"], seed
        patch = json.dumps({"state_patch": {"shelf_contents": {str(sc["vs"]): None}}, "action": "Wait({})"})
        s = ig.prompt_B(sc, "S", False, patch, filt=filt)
        st = json.loads(re.search(r"Current state:\n(\{.*?\n\})", s, re.S).group(1))
        held = {int(k) for k in st["shelf_contents"]}
        assert min(i for i in range(100) if i not in held) == sc["answer"], seed


def test_ig_store_shelf_parses_both_arms():
    assert ig.store_shelf('Action: Store({"shelf": 7, "sku": "SKU-A", "units": 1, "lot": "L-1"})') == 7
    assert ig.store_shelf('{"action": "Store({\\"shelf\\": 12, \\"sku\\": \\"SKU-A\\"})"}') == 12

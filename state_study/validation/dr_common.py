"""Shared helpers for replaying delayed-relevance traces.

The delayed-relevance project has no license: its code is imported at run time from the submodule
(never copied) and its traces are read from a local extract of its runs/table1-gemini-3-flash-preview-vertex
branch (never committed here). Prompts mirror its ReActRuntime.act: system "Instructions:\\n{spec}" followed by the history
block, the latest observation and the request for 'Action: <cmd>'."""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True  # do not leave __pycache__ inside the submodule

REPO = Path(__file__).resolve().parents[2]
DR_SRC = REPO / "third_party" / "references" / "delayed-relevance" / "src"


def import_dr(src: str | None = None):
    sys.path.insert(0, str(src or DR_SRC))
    from dr.envs.warehouse import Warehouse
    from dr.types import Action
    return Warehouse, Action


def load_steps(traces: Path, name: str) -> list[dict]:
    rows = [json.loads(l) for l in (traces / name).read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r for r in rows if r.get("kind") != "run_header"]


def reply(row: dict) -> str:
    raw = row["raw"]
    if isinstance(raw, str):  # some trace versions store the dict's repr
        raw = ast.literal_eval(raw)
    return raw["respuestas"][-1]


def react_prompt(spec: str, lines: list[str], step: int, obs: str) -> str:
    return (f"Instructions:\n{spec}\n\nHistory:\n" + "".join(l + "\n" for l in lines)
            + f"Latest Observation: [step {step}] {obs}\n"
            "Generate your next reasoning and action (format 'Action: <cmd>'):")


def history_lines(steps: list[dict]) -> list[str]:
    out = []
    for r in steps:
        out += [f"Observation: [step {r['step']}] {r['observation']}", f"Reasoning & Action: {reply(r)}"]
    return out

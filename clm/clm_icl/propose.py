# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Proposer: ask an LLM for candidate skills, then validate and write them.

The proposer is any chat model reachable through litellm (an OpenAI-compatible server
with ``api_base``, or a hosted model). It may be a stronger model than the agent
(assisted evolution) or the agent's own model (self-evolution). Tests and custom setups
can pass any ``Proposer`` callable instead.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

PROMPT_PATH = Path(__file__).with_name("proposer_prompt.md")

# (system, user) -> response text
Proposer = Callable[[str, str], str]


def litellm_proposer(
    model: str,
    *,
    api_base: str | None = None,
    temperature: float | None = 0.4,
    max_tokens: int = 16000,
    timeout_s: float = 900,
) -> Proposer:
    import litellm

    def call(system: str, user: str) -> str:
        kw = dict(model=model, max_tokens=max_tokens, timeout=timeout_s,
                  messages=[{"role": "system", "content": system},
                            {"role": "user", "content": user}])
        if api_base:
            kw["api_base"] = api_base
            if model.startswith("openai/") and not os.environ.get("OPENAI_API_KEY"):
                kw["api_key"] = "EMPTY"  # local OpenAI-compatible servers need no key
        if temperature is not None:
            kw["temperature"] = temperature
        resp = litellm.completion(**kw)
        return resp.choices[0].message.content or ""

    return call


def system_prompt(n_candidates: int, path: Path = PROMPT_PATH) -> str:
    return path.read_text().replace("{N}", str(n_candidates))


def user_message(current_skill: str, note: str, task_description: str) -> str:
    parts = ["## TASKS", task_description.strip() or "(no task description provided)", "",
             "## CURRENT SKILL (SKILL.md; the text you are improving on)",
             current_skill.strip() or "(none: the agent currently runs without a skill)", "",
             "## NOTE ON THE CURRENT SKILL'S ROLLOUTS (evidence)", note, "",
             "## YOUR TASK",
             "Follow the loop. Emit the ANALYSIS block, then the SKILL blocks per the OUTPUT "
             "CONTRACT. Nothing else."]
    return "\n".join(parts)


# The slug may be followed by three or more '>' (the template itself renders ">>>>").
SKILL_RE = re.compile(r"<<<SKILL name=([^\n>]+?)>{3,}\s*\n(.*?)\n<<<END>>>", re.S)
ANALYSIS_RE = re.compile(r"<<<ANALYSIS>{3,}\s*\n(.*?)\n<<<END>>>", re.S)
# Evidence citation: an episode id (E3) or a step / message index.
CITE_RE = re.compile(r"\bE\d+\b|\bstep\s*\d+|\bmsg\s*\d+", re.I)


@dataclass
class Proposal:
    analysis: str = ""
    skills: list[tuple[str, str]] = field(default_factory=list)  # (slug, SKILL.md text)
    rejected: list[str] = field(default_factory=list)


def parse_proposal(content: str, max_skills: int | None = None) -> Proposal:
    """Parse and validate a proposer response.

    Reasoning traces (``<think>...</think>``) are removed first, since they can contain
    draft blocks. A skill is kept if it has YAML frontmatter with ``name`` and
    ``description``, a non-empty body, and cites evidence in its description or the
    start of its body. Malformed skills are dropped and listed in ``rejected``; they
    are never repaired.
    """
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.S)
    content = re.sub(r"^.*?</think>", "", content, count=1, flags=re.S)
    p = Proposal()
    m = ANALYSIS_RE.search(content)
    p.analysis = m.group(1).strip() if m else ""
    if not p.analysis:
        p.rejected.append("ANALYSIS block missing or empty")
    seen: set[str] = set()
    for name, block in SKILL_RE.findall(content):
        name, b = name.strip(), block.strip()
        fm = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", b, re.S)
        if not fm:
            p.rejected.append(f"skill {name}: no YAML frontmatter"); continue
        front, body = fm.group(1), fm.group(2).strip()
        if "name:" not in front or "description:" not in front:
            p.rejected.append(f"skill {name}: frontmatter missing name/description"); continue
        if not body:
            p.rejected.append(f"skill {name}: empty body"); continue
        desc = next((ln for ln in front.splitlines() if ln.strip().startswith("description:")), "")
        if not CITE_RE.search(desc + " " + body[:400]):
            p.rejected.append(f"skill {name}: cites no episode or step"); continue
        slug = re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-")[:48] or "skill"
        while slug in seen:
            slug += "-b"
        seen.add(slug)
        p.skills.append((slug, b + "\n"))
    if not SKILL_RE.search(content):
        p.rejected.append(f"no parseable <<<SKILL ...>>> blocks ({len(content)} chars)")
    if max_skills is not None and len(p.skills) > max_skills:
        p.rejected += [f"skill {s}: over the {max_skills}-candidate limit"
                       for s, _ in p.skills[max_skills:]]
        p.skills = p.skills[:max_skills]
    return p

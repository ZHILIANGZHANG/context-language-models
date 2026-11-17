"""T1b: retrieval-friendly streaming entity records, with exact evidence sets.

Values are random codes, not inferable from entity names. Each entity appears
once, so the sufficient set is monotone and provides a clean control for T1's
state updates/corrections. Future questions and evidence IDs are never rendered.
"""
from __future__ import annotations

from dataclasses import dataclass
import random
import re

from .t2_bcp import _fn

SYSTEM = ("Read entity records page by page using next_records. When a page ends with a lookup "
          "question, call lookup with that entity's exact value before reading more. "
          "Use the offered memory tools to manage your paged history; do not guess unseen values.")
TOOLS = [_fn("next_records", "Read the next page of entity records.", {}, []),
         _fn("lookup", "Submit the exact value for the pending entity.",
             {"value": {"type": "string"}}, ["value"])]


@dataclass(frozen=True)
class LookupConfig:
    n_pages: int = 20
    records_per_page: int = 8
    n_decisions: int = 4


@dataclass(frozen=True)
class LookupDecision:
    page: int
    entity: str
    answer: str
    evidence_page: int
    kind: str = "lookup"


@dataclass
class LookupEpisode:
    seed: int
    pages: list[str]
    decisions: list[LookupDecision]
    need: dict[int, set[int]]

    def decision_at(self, page):
        return next((d for d in self.decisions if d.page == page), None)

    def next_use(self, page, after):
        return next((d.page for d in self.decisions if d.page > after and page in self.need[d.page]), None)


def rule_lookup(texts, entity):
    matches = []
    pattern = re.compile(r"^ENTITY " + re.escape(entity) + r": ([A-Z0-9-]+)$", re.M)
    for text in texts:
        matches.extend(pattern.findall(text))
    if len(set(matches)) > 1:
        raise ValueError("Conflicting entity records")
    return matches[0] if matches else None


def generate(seed, cfg=None):
    cfg = cfg or LookupConfig()
    if not (cfg.n_pages > cfg.n_decisions >= 1 and cfg.records_per_page >= 1):
        raise ValueError("Need positive record count and more pages than decisions")
    rng = random.Random(f"t1b-{seed}")
    pages, records, decisions, need = [], [], [], {}
    decision_pages = {(i + 1) * (cfg.n_pages - 1) // cfg.n_decisions for i in range(cfg.n_decisions)}
    for page in range(cfg.n_pages):
        lines = []
        for j in range(cfg.records_per_page):
            entity = f"entity-{seed}-{page}-{j}"
            value = ''.join(rng.choices('ABCDEFGHJKLMNPQRSTUVWXYZ23456789', k=12))
            lines.append(f"ENTITY {entity}: {value}")
            records.append((page, entity, value))
        if page in decision_pages:
            # Query an earlier page; no current-page shortcut by construction.
            evidence, entity, value = rng.choice([r for r in records if r[0] < page])
            lines.append(f"LOOKUP REQUIRED: what is the value of {entity}? Call lookup.")
            decisions.append(LookupDecision(page, entity, value, evidence))
            need[page] = {evidence, page}
        pages.append('\n'.join(lines))
    return LookupEpisode(seed, pages, decisions, need)


class LookupEnv:
    def __init__(self, seed, cfg=None):
        self.ep = generate(seed, cfg)
        self.system, self.tools = SYSTEM, TOOLS
        self.task = "Process the entity log until it ends. Call next_records to begin."
        self.cursor, self.pending = 0, None
        self.answers = {d.page: None for d in self.ep.decisions}

    @property
    def done(self):
        return self.cursor == len(self.ep.pages) and self.pending is None

    def step(self, name, args):
        if name == 'next_records':
            if self.pending is not None:
                return 'ERROR: answer the pending lookup first.', {}, False
            if self.cursor == len(self.ep.pages):
                return 'END OF LOG.', {}, True
            i = self.cursor
            self.cursor += 1
            self.pending = self.ep.decision_at(i)
            return self.ep.pages[i], {'env_page': i}, self.done
        if name == 'lookup':
            if self.pending is None:
                return 'ERROR: no lookup is pending.', {}, self.done
            value = args.get('value')
            if not isinstance(value, str) or not value.strip():
                return 'ERROR: value must be a nonempty string.', {}, False
            self.answers[self.pending.page] = value.strip()
            self.pending = None
            return 'Recorded.', {}, self.done
        return f'ERROR: unknown tool {name}.', {}, self.done

    def score(self):
        return sum(self.answers[d.page] == d.answer for d in self.ep.decisions) / len(self.ep.decisions)

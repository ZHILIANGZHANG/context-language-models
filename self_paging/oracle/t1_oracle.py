"""Oracle policy for T1 (docs/data_and_training.md §1.3, §2.2, §2.6).

Online, size-aware Belady: before pulling the next page it evicts what keeps the next page within
budget, choosing pages that are never needed again first, then the page needed farthest in the
future (assistant turns and old acknowledgements go before environment pages). Before answering a
decision it pages back in whatever the rule reader needs. In ``notes`` mode it writes a state note
(which ``rule_read`` understands) instead of keeping raw pages.

Every eviction is labelled *observable* (justified by what has been seen: the page carries no fact
the current state needs and no quarantine notice) or *future-informed* (only Belady's look-ahead
justifies it). The counts go into the episode meta for the realizability filter.
"""

from __future__ import annotations

import random

from ..core.protocol import render_tool_call
from ..envs.t1_warehouse import minimal_sufficient, rule_read, state_note


class T1Oracle:
    def __init__(self, mode: str = "paging", seed: int = 0, gen_reserve: int | None = None) -> None:
        if mode not in ("paging", "notes", "mixed"):
            raise ValueError(mode)
        self.mode, self.rng, self.gen_reserve = mode, random.Random(seed), gen_reserve
        self._sizes: dict[int, int] = {}
        self._tries: dict[int, int] = {}

    def _reserve(self, st) -> int:
        """Room for this turn's text and the next one (measured on the longest templates)."""
        if self.gen_reserve is None:
            longest = (self._derivation_text(999) + " Storing I-999999 on S999.\n"
                       + render_tool_call("store", {"shelf": 999}) + "\n"
                       + "Memory is tight; paging out " + ",".join(["9999"] * 8) + ".\n"
                       + render_tool_call("kv_out", {"pages": ",".join(["9999"] * 8)}) + "\n"
                       + render_tool_call("next_events", {}))
            self.gen_reserve = 2 * len(st.tok.encode(longest)) + 16
        return self.gen_reserve

    # -- helpers ---------------------------------------------------------------------------
    def _size(self, st, e: int) -> int:
        if e not in self._sizes:
            self._sizes[e] = len(st.tok.encode(st.env.ep.pages[e]))
        return self._sizes[e]

    @staticmethod
    def _visible_texts(st) -> list[str]:
        t, ep = st.log.table, st.env.ep
        by_pid = {pid: ep.pages[e] for e, pid in st.env_page.items()}
        by_pid.update(st.notes)
        return [by_pid[p] for p in t.in_order(t.resident()) if p in by_pid]

    def _observable(self, st, e: int) -> bool:
        ep = st.env.ep
        delivered = ep.pages[: st.env.cursor]
        now = rule_read(delivered)
        needed_now = minimal_sufficient(delivered, now, set())
        return e not in needed_now and "QUARANTINED" not in ep.pages[e]

    def _victims(self, st, need_free: int, protect: set[int]) -> list[int]:
        """Pids to evict so that ``need_free`` more tokens fit."""
        t, ep = st.log.table, st.env.ep
        free = st.cfg.budget - st.log.resident_tokens() - st.cfg.struct_reserve - self._reserve(st)
        if free >= need_free:
            return []
        env_of = {pid: e for e, pid in st.env_page.items()}
        cands = [p for p in t.resident() if not t.pages[p].pinned and p != t.current and p not in protect
                 and t.pages[p].kind not in ("gauge", "ack")]
        after = st.env.cursor - 1

        def key(p):
            if p not in env_of:  # assistant turns, acks, notes: oldest first
                return (0 if t.pages[p].kind != "note" else 1, 0, t.pages[p].start)
            nu = ep.next_use(env_of[p], after)
            return (2, -(10**9) if nu is None else -nu, -t.pages[p].length)

        out = []
        for p in sorted(cands, key=key):
            if free >= need_free:
                break
            out.append(p)
            free += t.pages[p].length
            if p in env_of:
                obs = self._observable(st, env_of[p])
                k = "observable_evictions" if obs else "future_informed_evictions"
                st.meta[k] = st.meta.get(k, 0) + 1
        return out

    # -- policy ----------------------------------------------------------------------------
    def __call__(self, st) -> str:
        env, ep = st.env, st.env.ep
        t = st.log.table
        if env.pending is not None:
            d = env.pending
            visible = self._visible_texts(st)
            tries = self._tries.get(d.page, 0)
            if rule_read(visible) != d.answer and tries < 2:
                self._tries[d.page] = tries + 1
                missing = [st.env_page[e] for e in sorted(ep.need[d.page])
                           if e in st.env_page and t.pages[st.env_page[e]].where == "host"]
                if missing:
                    size = sum(t.pages[p].length for p in missing)
                    victims = self._victims(st, size, protect={st.env_page[e] for e in ep.need[d.page]
                                                               if e in st.env_page})
                    parts = [f"Deciding the shelf for {d.item} needs pages "
                             f"{','.join(map(str, missing))}, which are out of memory; paging them back in."]
                    if victims:
                        parts.append(render_tool_call("kv_out", {"pages": ",".join(map(str, victims))}))
                    parts.append(render_tool_call("kv_in", {"pages": ",".join(map(str, missing))}))
                    return "\n".join(parts)
            if rule_read(visible) != d.answer:
                st.meta["oracle_unsupported"] = st.meta.get("oracle_unsupported", 0) + 1
            return (f"{self._derivation_text(d.answer)} Storing {d.item} on S{d.answer}.\n"
                    + render_tool_call("store", {"shelf": d.answer}))
        if env.cursor >= len(ep.pages):
            return "The log is finished.\n" + render_tool_call("next_events", {})
        nxt = self._size(st, env.cursor) + st.cfg.struct_reserve
        parts = []
        if st.cfg.budget is not None:
            victims = self._victims(st, nxt, protect=set())
            env_victims = [p for p in victims if t.pages[p].kind == "tool"]
            use_note = st.cfg.notes and env_victims and (
                self.mode == "notes" or (self.mode == "mixed" and self.rng.random() < 0.5))
            if use_note:
                note = state_note(self._visible_texts(st))
                parts.append("Summarising the current shelf state in a note before freeing memory.")
                parts.append(render_tool_call("note", {"text": note}))
                old_notes = [p for p in st.notes if t.pages[p].where == "gpu"]
                victims = sorted(set(victims) | set(env_victims) | set(old_notes))
            if victims:
                parts.append(f"Memory is tight; paging out {','.join(map(str, victims))}.")
                parts.append(render_tool_call("kv_out", {"pages": ",".join(map(str, victims))}))
        parts.append(render_tool_call("next_events", {}))
        return "\n".join(parts)

    @staticmethod
    def _derivation_text(a: int) -> str:
        return (f"Checking shelves from S1 upward against the visible records: every shelf below S{a} "
                f"is occupied or quarantined, and S{a} is empty and not quarantined.")

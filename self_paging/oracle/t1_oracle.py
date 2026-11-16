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

from ..agent.pagers import _overhead
from ..core.protocol import render_tool_call
from ..envs.t1_warehouse import (derivation, digest_shelves, minimal_sufficient, page_digest, page_role,
                                 rule_read, state_note)


class T1Oracle:
    def __init__(self, mode: str = "paging", seed: int = 0, gen_reserve: int | None = None,
                 error_rate: float = 0.0) -> None:
        # paging: evict on demand (Belady); notes: write a state note, then evict;
        # eager: also evict, every turn, the pages the current state does not need (reversible, so
        #        later decisions page them back in: this is what produces kv_in examples); mixed.
        if mode not in ("paging", "notes", "mixed", "eager"):
            raise ValueError(mode)
        self.mode, self.rng, self.gen_reserve = mode, random.Random(seed), gen_reserve
        self._sizes: dict[int, int] = {}
        self._tries: dict[int, int] = {}
        # mixed: per-episode propensities, so episodes differ in how much they note / evict early
        self.p_note = self.rng.uniform(0.1, 0.7) if mode == "mixed" else 0.0
        self.p_eager = self.rng.uniform(0.1, 0.9) if mode == "mixed" else 0.0
        # recovery examples: with this probability per turn, evict a page the next decision needs
        # (an unforced error). Those turns are listed in meta["error_pids"]; the data builder
        # removes them from the loss, so the model learns the recovery (kv_in), not the mistake.
        self.error_rate = error_rate

    def _reserve(self, st) -> int:
        """Room for this turn's text and the next one (measured on the longest templates)."""
        if self.gen_reserve is None:
            longest = ("Shelf check from the visible records: " + ", ".join(["S99 quarantined"] * 24)
                       + ", S99 empty and not quarantined. Lowest free shelf: S99. Storing I-999999 on S99.\n"
                       + render_tool_call("store", {"shelf": 999}) + "\n"
                       + "Memory is tight; paging out " + ", ".join(["9999 (not needed for the current shelf state)"] * 8) + ".\n"
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
    def _visible_texts(st, exclude=(), include=()) -> list[str]:
        """Text the rule reader would see: resident environment pages and notes, in log order."""
        t, ep = st.log.table, st.env.ep
        by_pid = {pid: ep.pages[e] for e, pid in st.env_page.items()}
        by_pid.update(st.notes)
        pids = (t.resident() | set(include)) - set(exclude)
        return [by_pid[p] for p in t.in_order(pids) if p in by_pid]

    def _observable(self, st, e: int) -> bool:
        ep = st.env.ep
        delivered = ep.pages[: st.env.cursor]
        now = rule_read(delivered)
        needed_now = minimal_sufficient(delivered, now, set())
        return e not in needed_now and "QUARANTINED" not in ep.pages[e]

    def _victims(self, st, need_free: int, protect: set[int]) -> list[int]:
        """Pids to evict so that ``need_free`` more tokens fit."""
        t, ep = st.log.table, st.env.ep
        free = st.cfg.budget - st.log.resident_tokens() - _overhead(st) - self._reserve(st)
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

        # Prefer evictions that keep the current state readable: removing a page must not change
        # what the rule reader concludes now (e.g. never drop a dispatch line while its store line
        # stays resident). Belady order decides among those; state-breaking ones go last.
        truth = rule_read(ep.pages[: st.env.cursor]) if st.env.cursor else None
        out: list[int] = []
        while free < need_free:
            left = [p for p in cands if p not in out]
            if not left:
                break

            def full_key(p):
                breaks = 0
                if truth is not None and (p in env_of or p in st.notes):
                    breaks = int(rule_read(self._visible_texts(st, exclude=out + [p])) != truth)
                k = key(p)
                return (k[0], breaks) + k[1:]

            p = min(left, key=full_key)
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
                need = {st.env_page[e] for e in ep.need[d.page] if e in st.env_page}
                missing = sorted(p for p in need if t.pages[p].where == "host")
                misleading: list[int] = []
                if rule_read(self._visible_texts(st, include=missing)) != d.answer:
                    # extra resident pages mislead the reader (a record whose follow-up is out):
                    # shrink the visible environment text to the minimal sufficient set
                    misleading = sorted(p for p in t.resident() if (p in st.notes or p in
                                        {pid for pid in st.env_page.values()}) and p not in need)
                for p in missing:  # can the model tell from the header that this page is the one?
                    if not self._cue(st, p, d.answer):
                        st.meta["unrealizable_kv_in"] = st.meta.get("unrealizable_kv_in", 0) + 1
                if missing or misleading:
                    size = sum(t.pages[p].length for p in missing) - sum(t.pages[p].length for p in misleading)
                    victims = self._victims(st, max(0, size), protect=need | set(misleading))
                    out = sorted(set(victims) | set(misleading))
                    parts = []
                    if missing:
                        parts.append(f"Deciding the shelf for {d.item} needs pages {','.join(map(str, missing))}, "
                                     "which are out of memory; paging them back in.")
                    if misleading:
                        parts.append(f"Pages {','.join(map(str, misleading))} hold records whose follow-ups are "
                                     "out of memory and would mislead the shelf check; paging them out.")
                    if victims:
                        parts.append("Making room: paging out " + self._reasons(st, victims) + ".")
                    if out:
                        parts.append(render_tool_call("kv_out", {"pages": ",".join(map(str, out))}))
                    if missing:
                        parts.append(render_tool_call("kv_in", {"pages": ",".join(map(str, missing))}))
                    return "\n".join(parts)
            if rule_read(visible) != d.answer:
                st.meta["oracle_unsupported"] = st.meta.get("oracle_unsupported", 0) + 1
            text = derivation(visible) if rule_read(visible) == d.answer else f"Lowest free shelf: S{d.answer}."
            return (f"{text} Storing {d.item} on S{d.answer}.\n"
                    + render_tool_call("store", {"shelf": d.answer}))
        if env.cursor >= len(ep.pages):
            return "The log is finished.\n" + render_tool_call("next_events", {})
        nxt = self._size(st, env.cursor)
        parts = []
        if st.cfg.budget is not None:
            victims = self._victims(st, nxt, protect=set())
            if self.mode == "eager" or (self.mode == "mixed" and self.rng.random() < self.p_eager):
                victims = sorted(set(victims) | set(self._eager(st)))
            mistake = self._mistake(st) if self.error_rate and self.rng.random() < self.error_rate else None
            if mistake is not None and mistake not in victims:
                victims = sorted(victims + [mistake])
                st.meta.setdefault("error_pids", []).append(t.current)
            env_victims = [p for p in victims if t.pages[p].kind == "tool"]
            use_note = st.cfg.notes and env_victims and (
                self.mode == "notes" or (self.mode == "mixed" and self.rng.random() < self.p_note))
            if use_note:
                note = state_note(self._visible_texts(st))
                parts.append("Summarising the current shelf state in a note before freeing memory.")
                parts.append(render_tool_call("note", {"text": note}))
                old_notes = [p for p in st.notes if t.pages[p].where == "gpu"]
                victims = sorted(set(victims) | set(env_victims) | set(old_notes))
            if victims:
                parts.append(("Freeing memory" if self.mode == "eager" else "Memory is tight")
                             + "; paging out " + self._reasons(st, victims) + ".")
                parts.append(render_tool_call("kv_out", {"pages": ",".join(map(str, victims))}))
                if self.mode == "mixed" and self.rng.random() < 0.3:
                    return "\n".join(parts)  # memory housekeeping as its own turn (action diversity)
        parts.append(render_tool_call("next_events", {}))
        return "\n".join(parts)

    def _reasons(self, st, victims: list[int]) -> str:
        t, ep = st.log.table, st.env.ep
        env_of = {pid: e for e, pid in st.env_page.items()}
        now = rule_read(ep.pages[: st.env.cursor]) if st.env.cursor else None
        out = []
        for p in victims:
            if p in env_of:
                why = page_role(ep.pages[env_of[p]], now)
            elif t.pages[p].kind == "note":
                why = "older note, superseded"
            else:
                why = "my earlier turn"
            out.append(f"{p} ({why})")
        return ", ".join(out)

    @staticmethod
    def _cue(st, pid: int, answer: int) -> bool:
        """A page-in is realizable only if the page's header digest points to it: it names a
        quarantine, or a shelf at or below the answer (docs/rl_design.md §2.2)."""
        if not getattr(getattr(st.env, "cfg", None), "digest", False):
            return False
        env_of = {v: k for k, v in st.env_page.items()}
        if pid not in env_of:
            return True  # notes and own turns: their text names what they hold
        dg = page_digest(st.env.ep.pages[env_of[pid]])
        return "quarantine" in dg or any(s <= answer for s in digest_shelves(dg))

    def _mistake(self, st) -> int | None:
        """A resident page that the next decision needs (evicting it forces a later kv_in)."""
        ep, t = st.env.ep, st.log.table
        upcoming = [d for d in ep.decisions if d.page >= st.env.cursor]
        if not upcoming:
            return None
        need = ep.need[upcoming[0].page]
        cands = [pid for e, pid in st.env_page.items() if e in need and t.pages[pid].where == "gpu"
                 and pid != t.current]
        return self.rng.choice(cands) if cands else None

    def _eager(self, st) -> list[int]:
        """Resident environment pages the current state does not need and whose removal keeps the
        current reading intact. Judged from what has been seen, so these evictions are observable."""
        t, ep = st.log.table, st.env.ep
        if not st.env.cursor:
            return []
        delivered = ep.pages[: st.env.cursor]
        now = rule_read(delivered)
        needed = minimal_sufficient(delivered, now, set())
        out = []
        for e, pid in sorted(st.env_page.items()):
            if t.pages[pid].where != "gpu" or e in needed or pid == t.current:
                continue
            if rule_read(self._visible_texts(st, exclude=out + [pid])) == now:
                out.append(pid)
                st.meta["observable_evictions"] = st.meta.get("observable_evictions", 0) + 1
        return out

    @staticmethod
    def _derivation_text(a: int) -> str:
        return (f"Checking shelves from S1 upward against the visible records: every shelf below S{a} "
                f"is occupied or quarantined, and S{a} is empty and not quarantined.")

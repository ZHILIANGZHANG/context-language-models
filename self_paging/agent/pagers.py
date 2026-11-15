"""System-side paging baselines (docs/experiment_plan.md §3.4): the *system* manages the resident
set and the model only does the task (KV tools are not offered).

A pager is called before every assistant turn and before every tool block with the episode state
and the number of tokens about to be written; it may page out / page in. ``Recent`` and ``Belady``
(T1 only: it reads the environment's future needs) are implemented. QuestPage and AttnEvict need
model internals (per-page key summaries, attention) and come with the SGLang/HF rollout backend.
"""

from __future__ import annotations


class RecentPager:
    """Keep pinned pages and the most recent pages that fit (StreamingLLM at page level)."""

    def __call__(self, st, incoming: int) -> None:
        log, cfg = st.log, st.cfg
        if cfg.budget is None:
            return
        while cfg.budget - log.resident_tokens() < incoming + cfg.struct_reserve + cfg.min_gen_room:
            victims = [p for p in log.table.in_order(log.table.resident())
                       if not log.table.pages[p].pinned and p != log.table.current]
            if not victims:
                return
            log.page_out([victims[0]])


class BeladyPager:
    """T1 upper bound: evict pages needed farthest in the future; page in what the pending decision
    needs. Uses ``env.ep`` (future information), so it is an oracle, not a deployable policy."""

    def __call__(self, st, incoming: int) -> None:
        log, cfg, env = st.log, st.cfg, st.env
        if cfg.budget is None:
            return
        t, ep = log.table, env.ep
        env_of = {pid: e for e, pid in st.env_page.items()}
        protect = set()
        if env.pending is not None:
            protect = {st.env_page[e] for e in ep.need[env.pending.page] if e in st.env_page}
            missing = [p for p in protect if t.pages[p].where == "host"]
            if missing:
                need = sum(t.pages[p].length for p in missing)
                self._evict(st, need + incoming, protect, env_of)
                try:
                    log.page_in(missing)
                except Exception:  # noqa: BLE001 - infeasible budget: answer with what is resident
                    pass
                return
        self._evict(st, incoming, protect, env_of)

    @staticmethod
    def _evict(st, need, protect, env_of):
        log, cfg, ep = st.log, st.cfg, st.env.ep
        t = log.table
        after = st.env.cursor - 1

        def key(p):
            if p not in env_of:
                return (0, 0, t.pages[p].start)
            nu = ep.next_use(env_of[p], after)
            return (1, -(10**9) if nu is None else -nu, -t.pages[p].length)

        while cfg.budget - log.resident_tokens() < need + cfg.struct_reserve + cfg.min_gen_room:
            cands = [p for p in t.resident() if not t.pages[p].pinned and p != t.current and p not in protect]
            if not cands:
                return
            log.page_out([sorted(cands, key=key)[0]])

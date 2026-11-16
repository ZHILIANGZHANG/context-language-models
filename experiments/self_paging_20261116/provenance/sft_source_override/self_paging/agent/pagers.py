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


class QuestPagePager:
    """Reversible key-mean page retrieval as specified by the local runbook.

    This is an approximate system control, not the upstream token-level Quest
    implementation. It uses no task answer, evidence ID or future observation.
    """
    def __call__(self, st, incoming):
        if st.cfg.budget is None:
            return
        log,table=st.log,st.log.table
        scores=log.keymean_scores()
        fixed={p for p in table.resident() if table.pages[p].pinned or p==table.current}
        room=st.cfg.budget-sum(table.pages[p].length for p in fixed)-incoming-st.cfg.struct_reserve-st.cfg.min_gen_room
        selected=set(fixed)
        candidates=[p for p in scores if p not in fixed and table.pages[p].where!='dropped']
        for p in sorted(candidates,key=lambda p:(-scores[p],table.pages[p].start)):
            if table.pages[p].length<=room:
                selected.add(p);room-=table.pages[p].length
        victims=table.resident()-selected
        if victims:log.page_out(sorted(victims))
        restore=[p for p in selected if table.pages[p].where=='host']
        if restore:log.page_in(sorted(restore))


class AttnEvictPager:
    """Irreversibly evict pages with least actual recent attention mass."""
    def __call__(self, st, incoming):
        if st.cfg.budget is None:
            return
        log,table=st.log,st.log.table
        scores=log.attention_scores()
        while log.resident_tokens()+incoming+st.cfg.struct_reserve+st.cfg.min_gen_room>st.cfg.budget:
            candidates=[p for p in table.resident() if not table.pages[p].pinned and p!=table.current]
            if not candidates:break
            victim=min(candidates,key=lambda p:(scores.get(p,0.0),table.pages[p].start))
            log.drop([victim])

"""Run one episode: environment + policy + token log (or ``PagedKV``) under a KV budget.

The same runner produces SFT data (scripted oracle policy on a model-free ``TokenLog``), rollouts
for RL (``ModelPolicy`` on ``PagedKV``), and evaluation. Its output record is everything training
needs: token ids exactly as fed (token-in, token-out), page and resident-set version per token,
loss / control masks, behaviour log-probs, reward and episode statistics.

Turn structure (docs/experiment_plan.md §2):
  assistant page  <|im_start|>assistant\\n[<think>..</think>]...tool calls...<|im_end|>\\n
  tool block      KV tool calls run first, then environment calls; every response is
                  pinned header + body page + pinned trailer; one gauge page; pinned close.
Over-budget tool output is truncated (the system never decides what to evict), except when not
even a minimal assistant turn fits: then the oldest unpinned page is evicted and counted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..core.pages import PagingError
from ..core.protocol import (KV_TOOLS, Tok, assistant_prefix, block_close, block_open, escape, gauge,
                             kv_tool_schemas, parse_pages, parse_tool_calls, render_system,
                             render_user, response_header, response_trailer)

TRUNC = "\n[truncated {n} tokens: over the KV budget]"


@dataclass
class EpisodeConfig:
    budget: int | None = None
    max_turns: int = 60
    max_new_tokens: int = 512
    thinking: bool = False
    reversible: bool = True
    notes: bool = True
    max_invalid: int = 5
    min_gen_room: int = 48
    struct_reserve: int = 64  # tokens kept free for trailers, gauge and block close
    kv_tools: bool = True     # False: the system pages (see agent/pagers.py), the model only does the task


@dataclass
class EpisodeState:
    """What a policy may look at (the oracle also reads ``env`` internals)."""

    env: object
    log: object
    tok: Tok
    cfg: EpisodeConfig
    turn: int = 0
    env_page: dict[int, int] = field(default_factory=dict)  # env page index -> log pid
    notes: dict[int, str] = field(default_factory=dict)      # note pid -> text
    meta: dict = field(default_factory=dict)


def _fill_to(log, pid, ids):
    return log.feed(pid, ids)


def run_episode(env, log, policy, tok: Tok, cfg: EpisodeConfig, pager=None) -> dict:
    st = EpisodeState(env, log, tok, cfg)
    st.meta.update(invalid=0, truncated_tokens=0, forced_evictions=0, kv_errors=0,
                   kv_out=0, kv_in=0, notes=0, turns=0)
    tools = list(env.tools) + (kv_tool_schemas(cfg.reversible, cfg.notes) if cfg.kv_tools else [])
    loss_mask: list[int] = []
    ctrl_mask: list[int] = []
    logprobs: list[float] = []

    def feed(pid, text_or_ids, *, gen=False):
        ids = tok.encode(text_or_ids) if isinstance(text_or_ids, str) else list(text_or_ids)
        out = log.feed(pid, ids)
        loss_mask.extend([1 if gen else 0] * len(ids))
        ctrl_mask.extend([0] * len(ids))
        logprobs.extend([0.0] * len(ids))
        return out

    def pinned(kind, text):
        return feed(log.open(kind, pinned=True), text)

    pinned("system", render_system(env.system, tools))
    last = pinned("task", render_user(env.task))

    done = False
    for turn in range(cfg.max_turns):
        st.turn = turn
        st.meta["turns"] = turn + 1
        if pager is not None:
            pager(st, 0)
        # make sure a minimal assistant turn fits; otherwise the system evicts (and it is counted)
        if cfg.budget is not None:
            while cfg.budget - log.resident_tokens() < cfg.min_gen_room + cfg.struct_reserve:
                victims = [p for p in log.table.in_order(log.table.resident())
                           if not log.table.pages[p].pinned and p != log.table.current]
                if not victims:
                    break
                log.page_out([victims[0]])
                st.meta["forced_evictions"] += 1
                last = None
        a = log.open("assistant")
        prefix = feed(a, assistant_prefix(cfg.thinking))
        last = prefix if prefix is not None else last
        room = cfg.max_new_tokens
        if cfg.budget is not None:
            room = max(1, min(room, cfg.budget - log.resident_tokens() - cfg.struct_reserve))
        start = log.log_len
        ids, lps = policy.generate(log, a, None if last is None else last[-1], room, st)
        n = len(ids)
        loss_mask.extend([1] * n)
        ctrl_mask.extend([0] * n)
        logprobs.extend(lps if lps is not None else [0.0] * n)
        if not ids or ids[-1] != tok.im_end:
            feed(a, [tok.im_end])
        feed(a, "\n")
        text = tok.decode(ids)
        calls, spans, bad = parse_tool_calls(text)
        st.meta["invalid"] += bad + (0 if calls else 1)
        _mark_ctrl(ctrl_mask, start, ids, text, calls, spans, tok)

        responses = []  # (label, body, kind, env_page)
        for c in [c for c in calls if c["name"] in KV_TOOLS]:
            responses.append(_kv_call(c, st))
        for c in [c for c in calls if c["name"] not in KV_TOOLS]:
            if c["name"] not in {t["function"]["name"] for t in env.tools}:
                responses.append(("tool=" + c["name"], f"ERROR: unknown tool {c['name']}.", "ack", None))
                st.meta["invalid"] += 1
                continue
            body, meta, d = env.step(c["name"], c["arguments"])
            done = done or d
            kind = "tool" if "env_page" in meta else "ack"
            responses.append(("tool=" + c["name"], body, kind, meta.get("env_page")))
        if not responses:
            responses.append(("kv", "ERROR: no valid tool call. Call a tool.", "ack", None))

        if pager is not None:
            pager(st, sum(len(tok.encode(b)) for _, b, k, _ in responses if k != "ack"))
        last = _write_block(st, responses, feed, pinned)
        if done or getattr(env, "done", False) or st.meta["invalid"] >= cfg.max_invalid:
            break

    st.meta["over_budget"] = int(cfg.budget is not None and log.peak_resident > cfg.budget)
    rec = log.record()
    rec.update(loss_mask=loss_mask, ctrl_mask=ctrl_mask, logprobs=logprobs,
               reward=float(env.score()), budget=cfg.budget, peak_resident=log.peak_resident,
               env_page={str(k): v for k, v in st.env_page.items()}, meta=st.meta)
    return rec


def _kv_call(c, st: EpisodeState):
    log, name, args = st.log, c["name"], c["arguments"]
    if name == "note":
        text = str(args.get("text", ""))[:2000]
        st.meta["notes"] += 1
        return ("note", escape(text) or "(empty note)", "note", None)
    try:
        pids = parse_pages(args.get("pages", ""))
        if name == "kv_out":
            log.page_out(pids)
            st.meta["kv_out"] += 1
            return ("kv", f"paged out: {','.join(map(str, pids))}", "ack", None)
        if name == "kv_in":
            log.page_in(pids)
            st.meta["kv_in"] += 1
            return ("kv", f"paged in: {','.join(map(str, pids))}", "ack", None)
        if name == "kv_drop":
            log.drop(pids)
            return ("kv", f"dropped: {','.join(map(str, pids))}", "ack", None)
    except (PagingError, ValueError) as e:
        st.meta["kv_errors"] += 1
        return ("kv", f"ERROR: {e}", "ack", None)
    return ("kv", f"ERROR: unknown KV tool {name}", "ack", None)


def _write_block(st: EpisodeState, responses, feed, pinned):
    log, tok, cfg = st.log, st.tok, st.cfg
    # system housekeeping (not model actions): last turn's acknowledgements and gauge go out
    old = [p for p in log.table.resident() if log.table.pages[p].kind in ("gauge", "ack")]
    if old:
        log.page_out(old)
    pinned("header", block_open())
    for label, body, kind, env_page in responses:
        if kind == "ack":  # short replies are one unpinned page, wrapper included, no directory entry
            feed(log.open("ack"), f"\n<tool_response>\n{body}\n</tool_response>")
            continue
        h = log.next_pid
        pinned("header", response_header(h + 1, label))
        pid = log.open(kind)
        ids = tok.encode(escape(body) if kind == "tool" else body) or tok.encode(" ")
        if cfg.budget is not None:
            avail = cfg.budget - log.resident_tokens() - cfg.struct_reserve
            if len(ids) > avail:
                marker = tok.encode(TRUNC.format(n=len(ids) - max(avail, 0)))
                keep = max(0, avail - len(marker))
                st.meta["truncated_tokens"] += len(ids) - keep
                ids = ids[:keep] + marker
        feed(pid, ids)
        if env_page is not None:
            st.env_page[env_page] = pid
        if kind == "note":
            st.notes[pid] = body
        pinned("header", response_trailer())
    t = log.table
    gpu = [(p, t.pages[p].kind) for p in t.in_order(t.resident()) if not t.pages[p].pinned
           and t.pages[p].kind not in ("gauge", "ack")]
    host = [(p, t.pages[p].kind) for p in t.in_order(p for p, pg in t.pages.items() if pg.where == "host")
            if t.pages[p].kind not in ("gauge", "ack")]
    feed(log.open("gauge"), gauge(log.resident_tokens(), cfg.budget, gpu, host))
    return pinned("header", block_close())


def _mark_ctrl(ctrl_mask, start, ids, text, calls, spans, tok):
    """Mark the tokens of KV tool calls (for the efficiency channel of the advantage)."""
    kv_spans = [s for c, s in zip(calls, spans) if c["name"] in KV_TOOLS]
    if not kv_spans or not ids:
        return
    ends, acc = [], 0
    for i in range(len(ids)):
        acc = len(tok.decode(ids[: i + 1]))
        ends.append(acc)
    begin = 0
    for i, end in enumerate(ends):
        if any(a < end and begin < b for a, b in kv_spans):
            ctrl_mask[start + i] = 1
        begin = end


# -- policies --------------------------------------------------------------------------------
class ModelPolicy:
    """Sample from the model through the paged log. Records behaviour log-probs."""

    def __init__(self, tok: Tok, temperature: float = 1.0, top_p: float = 1.0, top_k: int = 0,
                 generator: torch.Generator | None = None) -> None:
        self.tok, self.t, self.top_p, self.top_k, self.gen = tok, temperature, top_p, top_k, generator

    def generate(self, log, pid, last_logits, max_new, st):
        ids, lps = [], []
        logits = last_logits
        for _ in range(max_new):
            lp = torch.log_softmax(logits.float() / self.t, -1)
            probs = lp.exp()
            if self.top_k:
                kth = torch.topk(probs, self.top_k).values[-1]
                probs = torch.where(probs >= kth, probs, torch.zeros_like(probs))
            if self.top_p < 1.0:
                sp, si = torch.sort(probs, descending=True)
                cut = torch.cumsum(sp, 0) - sp > self.top_p
                sp[cut] = 0
                probs = torch.zeros_like(probs).scatter(0, si, sp)
            t = int(torch.multinomial(probs / probs.sum(), 1, generator=self.gen))
            ids.append(t)
            lps.append(float(lp[t]))
            logits = log.feed(pid, [t])[-1]
            if t == self.tok.im_end:
                break
        return ids, lps


class ScriptedPolicy:
    """Force a text chosen by ``fn(state)`` (oracle, templates, hinted teacher output).
    Records the model's log-probs of the forced tokens when the log has a model."""

    def __init__(self, tok: Tok, fn) -> None:
        self.tok, self.fn = tok, fn

    def generate(self, log, pid, last_logits, max_new, st):
        # no truncation: a scripted turn that does not fit is recorded as over_budget instead
        ids = self.tok.encode(self.fn(st)) + [self.tok.im_end]
        logits = log.feed(pid, ids)
        if logits is None or last_logits is None:
            return ids, None
        prev = torch.cat([last_logits[None], logits[:-1]], 0).float()
        lps = torch.log_softmax(prev, -1).gather(1, torch.tensor(ids)[:, None].to(prev.device))[:, 0]
        return ids, lps.tolist()

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
import re

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
    max_log_tokens: int | None = None  # absolute position limit, independent of resident budget
    answer_reserve: int = 1536         # room for the limit notice and one final answer
    overflow: str = "truncate"        # lru is a separately labelled hybrid-system ablation


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
    last_access: dict[int, int] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


def _fill_to(log, pid, ids):
    return log.feed(pid, ids)


def run_episode(env, log, policy, tok: Tok, cfg: EpisodeConfig, pager=None) -> dict:
    if cfg.overflow not in ("truncate", "lru"):
        raise ValueError("Unknown overflow policy")
    st = EpisodeState(env, log, tok, cfg)
    st.meta.update(invalid=0, truncated_tokens=0, forced_evictions=0, kv_errors=0,
                   kv_out=0, kv_in=0, notes=0, turns=0)
    tools = list(env.tools) + (kv_tool_schemas(cfg.reversible, cfg.notes) if cfg.kv_tools else [])
    loss_mask: list[int] = []
    ctrl_mask: list[int] = []
    logprobs: list[float] = []

    def feed(pid, text_or_ids, *, gen=False):
        ids = tok.encode(text_or_ids) if isinstance(text_or_ids, str) else list(text_or_ids)
        if cfg.max_log_tokens is not None and log.log_len + len(ids) > cfg.max_log_tokens:
            raise ValueError("Absolute log limit would be exceeded")
        out = log.feed(pid, ids)
        st.last_access[pid] = log.log_len
        loss_mask.extend([1 if gen else 0] * len(ids))
        ctrl_mask.extend([0] * len(ids))
        logprobs.extend([0.0] * len(ids))
        return out

    def pinned(kind, text):
        return feed(log.open(kind, pinned=True), text)

    pinned("system", render_system(env.system, tools))
    last = pinned("task", render_user(env.task))
    if cfg.max_log_tokens is not None:
        if cfg.answer_reserve < cfg.max_new_tokens + 256:
            raise ValueError("answer_reserve must include generation plus 256 structural tokens")
        if log.log_len + cfg.answer_reserve > cfg.max_log_tokens:
            raise ValueError("Initial prompt leaves no room for a final answer")

    done = False
    for turn in range(cfg.max_turns):
        st.turn = turn
        st.meta["turns"] = turn + 1
        final_turn = bool(st.meta.get("log_limit_notice"))
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
                st.meta.setdefault("kv_log", []).append(
                    {"turn": turn, "turn_pid": None, "op": "forced_out", "pages": [victims[0]], "ok": True})
                last = None
        turn_version = log.table.version
        a = log.open("assistant")
        prefix = feed(a, assistant_prefix(cfg.thinking))
        last = prefix if prefix is not None else last
        room = cfg.max_new_tokens
        if cfg.budget is not None:
            room = max(1, min(room, cfg.budget - log.resident_tokens() - cfg.struct_reserve))
        if cfg.max_log_tokens is not None:
            room = min(room, cfg.max_log_tokens - log.log_len - 8)
            if room <= 0:
                st.meta["log_limit_reached"] = 1
                break
        start = log.log_len
        ids, lps = policy.generate(log, a, None if last is None else last[-1], room, st)
        n = len(ids)
        loss_mask.extend([1] * n)
        ctrl_mask.extend([0] * n)
        logprobs.extend(lps if lps is not None else [0.0] * n)
        if not ids or ids[-1] != tok.im_end:
            feed(log.table.current, [tok.im_end])
        feed(log.table.current, "\n")
        text = tok.decode(ids)
        parse_text = text
        if cfg.thinking:
            # Never execute a quoted/planned tool call inside reasoning. Keep
            # character offsets intact for the control-token loss mask.
            parse_text = re.sub(r"<think>.*?(?:</think>|$)", lambda m: " " * len(m[0]), text, flags=re.S)
        calls, spans, bad = parse_tool_calls(parse_text)
        st.meta["invalid"] += bad + (0 if calls else 1)
        _mark_ctrl(ctrl_mask, start, ids, text, calls, spans, tok)

        responses = []  # (label, body, kind, env_page)
        for c in [c for c in calls if c["name"] in KV_TOOLS]:
            if not cfg.kv_tools:
                # tools that were not offered are not executed (strict system-paging baselines)
                responses.append(("tool=" + c["name"], f"ERROR: unknown tool {c['name']}.", "ack", None))
                st.meta["invalid"] += 1
                st.meta["unoffered_kv_calls"] = st.meta.get("unoffered_kv_calls", 0) + 1
                continue
            responses.append(_kv_call(c, st, turn_pid=a, turn=turn))
        for c in [c for c in calls if c["name"] not in KV_TOOLS]:
            if final_turn and c["name"] != "answer":
                st.meta["invalid"] += 1
                continue  # the limit notice allows one answer, never another retrieval
            if c["name"] not in {t["function"]["name"] for t in env.tools}:
                responses.append(("tool=" + c["name"], f"ERROR: unknown tool {c['name']}.", "ack", None))
                st.meta["invalid"] += 1
                continue
            pending = getattr(env, "pending", None)
            body, meta, d = env.step(c["name"], c["arguments"])
            st.meta.setdefault("env_calls", []).append(
                {"turn": turn, "turn_pid": a, "version": turn_version, "name": c["name"],
                 "args": c["arguments"], "decision_page": getattr(pending, "page", None)})
            done = done or d
            kind = "tool" if "env_page" in meta else "ack"
            label = "tool=" + c["name"] + (f" | {meta['digest']}" if meta.get("digest") else "")
            responses.append((label, body, kind, meta.get("env_page")))
        if not responses:
            responses.append(("kv", "ERROR: no valid tool call. Call a tool.", "ack", None))

        if final_turn:
            st.meta["log_limit_reached"] = 1
            break  # no unnecessary acknowledgement after the final capped answer
        if pager is not None:
            pager(st, sum(len(tok.encode(b)) for _, b, k, _ in responses if k != "ack"))
        last = _write_block(st, responses, feed, pinned)
        if done or getattr(env, "done", False) or st.meta["invalid"] >= cfg.max_invalid or st.meta.get("structural_exhausted", 0):
            break

    st.meta["over_budget"] = int(cfg.budget is not None and log.peak_resident > cfg.budget)
    rec = log.record()
    rec.update(loss_mask=loss_mask, ctrl_mask=ctrl_mask, logprobs=logprobs,
               reward=float(env.score()), budget=cfg.budget, peak_resident=log.peak_resident,
               env_page={str(k): v for k, v in st.env_page.items()}, meta=st.meta)
    return rec


def _kv_call(c, st: EpisodeState, turn_pid=None, turn=None):
    log, name, args = st.log, c["name"], c["arguments"]
    entry = {"turn": turn, "turn_pid": turn_pid, "op": name, "pages": [], "ok": False}
    st.meta.setdefault("kv_log", []).append(entry)  # used for counterfactual credit (train/advantage.py)
    if name == "note":
        text = str(args.get("text", ""))[:2000]
        st.meta["notes"] += 1
        entry["ok"] = True
        return ("note", escape(text) or "(empty note)", "note", None)
    try:
        pids = parse_pages(args.get("pages", ""))
        entry["pages"] = pids
        if name == "kv_out":
            log.page_out(pids)
            entry["ok"] = True
            st.meta["kv_out"] += 1
            return ("kv", f"paged out: {','.join(map(str, pids))}", "ack", None)
        if name == "kv_in":
            log.page_in(pids)
            entry["ok"] = True
            for pid in pids:
                st.last_access[pid] = log.log_len
            st.meta["kv_in"] += 1
            return ("kv", f"paged in: {','.join(map(str, pids))}", "ack", None)
        if name == "kv_drop":
            log.drop(pids)
            entry["ok"] = True
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
    if cfg.overflow == "lru" and cfg.budget is not None:
        # LRU at page accesses (creation/feed and explicit kv_in). Reading all
        # resident keys in one attention step does not distinguish their ages.
        # Include actual headers/directories in the fit calculation below,
        # rather than guessing a constant structural allowance.
        _lru_room(st, sum(len(tok.encode(body)) for _, body, _, _ in responses)
                  + cfg.struct_reserve + cfg.min_gen_room)
    # Plan the entire response block before feeding it. Directory size grows
    # with the number of pages, so a constant 64-token reserve is insufficient.
    # Headers/acks/gauge are never truncated; tool bodies consume the remainder.
    plans = []
    def add(kind, ids, pin=False, env=None, text=None):
        plans.append(dict(kind=kind, ids=ids, pin=pin, env=env, text=text))
    add("header", tok.encode(block_open()), True)
    for label, body, kind, env_page in responses:
        if kind == "ack":
            add("ack", tok.encode(f"\n<tool_response>\n{body}\n</tool_response>"))
            continue
        body = (escape(body) if kind == "tool" else body) or " "
        body_pid = log.next_pid + len(plans) + 1
        h, ids, tail = tok.encode_wrapped(response_header(body_pid, label), body, response_trailer())
        add("header", h, True)
        add(kind, ids, env=env_page, text=body)
        add("header", tail, True)
    t = log.table
    gpu = [(p, t.pages[p].kind) for p in t.in_order(t.resident()) if not t.pages[p].pinned
           and t.pages[p].kind not in ("gauge", "ack")]
    gpu += [(log.next_pid + i, p["kind"]) for i, p in enumerate(plans)
            if not p["pin"] and p["kind"] not in ("gauge", "ack")]
    host = [(p, t.pages[p].kind) for p in t.in_order(p for p, pg in t.pages.items() if pg.where == "host")
            if t.pages[p].kind not in ("gauge", "ack")]
    close_ids = tok.encode(block_close())
    planned_tokens = sum(len(p["ids"]) for p in plans) + len(close_ids) + 256
    if cfg.max_log_tokens is not None and log.log_len + planned_tokens > cfg.max_log_tokens - cfg.answer_reserve:
        # Do not partly expose a document and pretend it was complete. Reserve
        # a final generation before the absolute position limit. The discarded
        # result is explicitly reported and not added to env_page/resident state.
        st.meta["log_limit_notice"] = 1
        st.meta["log_limit_omitted_result_tokens"] = sum(len(tok.encode(body)) for _, body, _, _ in responses)
        notice = ("The absolute conversation log limit is near. The latest tool result was omitted. "
                  "Do not call search, get_document, or next_session again. "
                  "Call answer now using only information already received; acknowledge uncertainty if needed.")
        notice_ids = tok.encode(render_user(notice))
        if cfg.budget is not None and log.resident_tokens() + len(notice_ids) > cfg.budget:
            st.meta["structural_exhausted"] = 1
            return None
        return pinned("header", notice_ids)
    baseline = log.resident_tokens()
    originals = {i: p["ids"][:] for i, p in enumerate(plans) if p["kind"] in ("tool", "note")}
    # Re-evaluate the rendered counter after truncation; integer tokenization
    # can change with the resident-token count. Shrink only, guaranteeing exit.
    while True:
        before_gauge = baseline + sum(len(p["ids"]) for p in plans)
        gauge_ids = tok.encode(gauge(before_gauge, cfg.budget, gpu, host))
        excess = 0 if cfg.budget is None else before_gauge + len(gauge_ids) + len(close_ids) - cfg.budget
        if excess <= 0:
            break
        if cfg.overflow == "lru" and _lru_room(st, before_gauge - baseline + len(gauge_ids) + len(close_ids)):
            # The directory and resident-token gauge must describe the new
            # resident set; regenerate the complete plan after system eviction.
            return _write_block(st, responses, feed, pinned)
        changed = False
        for i in reversed(list(originals)):
            current = plans[i]["ids"]
            original = originals[i]
            # Find the largest prefix plus a complete truncation marker that
            # actually fits; never silently feed an over-budget marker itself.
            limit = max(0, len(current) - excess)
            keep = min(len(original) - 1, limit)
            while keep >= 0:
                marker = tok.encode(TRUNC.format(n=len(original) - keep))
                if keep + len(marker) <= limit:
                    plans[i]["ids"] = original[:keep] + marker
                    excess -= len(current) - len(plans[i]["ids"])
                    changed = True
                    break
                keep -= 1
            if excess <= 0:
                break
        if not changed:
            st.meta["structural_exhausted"] = st.meta.get("structural_exhausted", 0) + 1
            return None  # metadata itself cannot fit: end, never exceed budget
    for i, p in enumerate(plans):
        pid = log.open(p["kind"], pinned=p["pin"])
        feed(pid, p["ids"])
        if p["env"] is not None:
            st.env_page[p["env"]] = pid
        if p["kind"] == "note":
            st.notes[pid] = tok.decode(p["ids"])
        if i in originals and p["ids"] != originals[i]:
            # Count omitted original tokens, not marker length.
            prefix = 0
            for x, y in zip(originals[i], p["ids"]):
                if x != y: break
                prefix += 1
            st.meta["truncated_tokens"] += len(originals[i]) - prefix
            st.meta["truncated_pages"] = st.meta.get("truncated_pages", 0) + 1
    feed(log.open("gauge"), gauge_ids)
    return pinned("header", close_ids)


def block_overhead(st, n_responses: int = 1, label: str = "tool=next_events") -> int:
    """Tokens a tool-response block costs besides its bodies: open/close, per-response header and
    trailer, and the gauge, which lists every page and therefore grows with the directory. Pagers
    and the oracle must reserve this, not a constant (docs/rl_design.md §7.1)."""
    tok, log = st.tok, st.log
    t = log.table
    gpu = [(p, t.pages[p].kind) for p in t.in_order(t.resident()) if not t.pages[p].pinned
           and t.pages[p].kind not in ("gauge", "ack")]
    host = [(p, t.pages[p].kind) for p in t.in_order(p for p, pg in t.pages.items() if pg.where == "host")
            if t.pages[p].kind not in ("gauge", "ack")]
    gpu += [(log.next_pid + i, "tool") for i in range(n_responses)]
    per = len(tok.encode(response_header(log.next_pid + 1, label + " | " + "x" * 80))) + len(tok.encode(response_trailer()))
    return (len(tok.encode(block_open())) + len(tok.encode(block_close())) + n_responses * per
            + len(tok.encode(gauge(st.cfg.budget or 0, st.cfg.budget, gpu, host))) + 16)


def _lru_room(st, incoming):
    """Evict oldest page accesses without changing immutable/pinned content."""
    if st.cfg.budget is None:
        return False
    log, table = st.log, st.log.table
    changed = False
    while log.resident_tokens() + incoming > st.cfg.budget:
        candidates = [p for p in table.resident() if not table.pages[p].pinned and p != table.current]
        if not candidates:
            break
        victim = min(candidates, key=lambda p: (st.last_access.get(p, table.pages[p].end), table.pages[p].start))
        log.page_out([victim])
        st.meta['lru_evictions'] = st.meta.get('lru_evictions', 0) + 1
        changed = True
    return changed


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
            if self.t == 0:
                lp = torch.log_softmax(logits.float(), -1)
                t = int(logits.argmax())
                ids.append(t)
                lps.append(float(lp[t]))
                logits = _feed_generated(log, pid, [t], st)[-1]
                pid = log.table.current
                if t == self.tok.im_end:
                    break
                continue
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
            probs = probs / probs.sum()
            t = int(torch.multinomial(probs, 1, generator=self.gen))
            ids.append(t)
            lps.append(float(probs[t].log()))
            logits = _feed_generated(log, pid, [t], st)[-1]
            pid = log.table.current
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
        if getattr(log, "last_logits_only", False):
            raise ValueError("Teacher-forced scoring requires all token logits")
        ids = self.tok.encode(self.fn(st)) + [self.tok.im_end]
        logits = _feed_generated(log, pid, ids, st)
        if logits is None or last_logits is None:
            return ids, None
        prev = torch.cat([last_logits[None], logits[:-1]], 0).float()
        lps = torch.log_softmax(prev, -1).gather(1, torch.tensor(ids)[:, None].to(prev.device))[:, 0]
        return ids, lps.tolist()


def _feed_generated(log, pid, ids, st):
    """Split thought/body pages at special-token boundaries without retokenizing."""
    if not st.cfg.thinking:
        return log.feed(pid, ids)
    begin, end = st.tok.token_id("<think>"), st.tok.token_id("</think>")
    outputs, chunk = [], []
    def flush():
        if chunk:
            outputs.append(log.feed(log.table.current, list(chunk)))
            chunk.clear()
    for token in ids:
        if token == begin and log.table.pages[log.table.current].kind != "think":
            flush()
            log.open("think")
        chunk.append(token)
        if token == end and log.table.pages[log.table.current].kind == "think":
            flush()
            log.open("assistant")
    flush()
    return torch.cat(outputs,0) if outputs and outputs[0] is not None else None

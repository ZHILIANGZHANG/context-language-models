"""Qwen3 rendering and parsing for the self-paging protocol (docs/experiment_plan.md §2).

Everything is rendered incrementally, segment by segment, because the log is append-only. The
segment strings follow the Qwen3 chat template [K]; ``scripts``-side check: render a conversation
with these helpers and compare token by token with ``tokenizer.apply_chat_template`` on the server
(``self_paging/tests/test_template_server.py``, skipped unless ``QWEN3_TOKENIZER`` is set).

Layout of one tool-response block (pinned parts keep the chat structure intact when a body is out):

    <|im_start|>user                                   pinned   (block open)
    \\n<tool_response>\\n[[PAGE 12 tool=next_events]]\\n  pinned   (header; names the body page)
    <body text>                                        page 12  (can be paged out)
    \\n</tool_response>                                 pinned   (trailer)
    ... more responses ...
    \\n[KV] resident ...                                gauge    (system pages out old gauges)
    <|im_end|>\\n                                       pinned   (block close)
"""

from __future__ import annotations

import json
import re

IM_START, IM_END = "<|im_start|>", "<|im_end|>"
NO_THINK = "<think>\n\n</think>\n\n"

KV_TOOLS = ("kv_out", "kv_in", "note", "kv_drop")

_PAGES_ARG = {"type": "string", "description": "Page ids, e.g. \"3,5-8\"."}


def kv_tool_schemas(reversible: bool = True, notes: bool = True) -> list[dict]:
    tools = []
    if reversible:
        tools.append(_fn("kv_out", "Move pages out of GPU memory. They can be paged back in at any time; "
                         "while out you only see their header line.", {"pages": _PAGES_ARG}, ["pages"]))
        tools.append(_fn("kv_in", "Page previously paged-out pages back into GPU memory.",
                         {"pages": _PAGES_ARG}, ["pages"]))
    else:
        tools.append(_fn("kv_drop", "Permanently delete pages from memory. This cannot be undone.",
                         {"pages": _PAGES_ARG}, ["pages"]))
    if notes:
        tools.append(_fn("note", "Write a short note. It becomes a new page that stays resident and can "
                         "replace a whole page of raw text.", {"text": {"type": "string"}}, ["text"]))
    return tools


def _fn(name: str, desc: str, props: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required}}}


# -- rendering -------------------------------------------------------------------------------
def render_system(system: str, tools: list[dict]) -> str:
    """System turn with tool signatures, as the Qwen3 template writes it [K]."""
    lines = "\n".join(json.dumps(t, ensure_ascii=False) for t in tools)
    return (f"{IM_START}system\n{system}\n\n# Tools\n\nYou may call one or more functions to assist with "
            "the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:"
            f"\n<tools>\n{lines}\n</tools>\n\nFor each function call, return a json object with function name "
            "and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, "
            f"\"arguments\": <args-json-object>}}\n</tool_call>{IM_END}\n")


def render_user(text: str) -> str:
    return f"{IM_START}user\n{text}{IM_END}\n"


def assistant_prefix(thinking: bool) -> str:
    return f"{IM_START}assistant\n" + ("" if thinking else NO_THINK)


def block_open() -> str:
    return f"{IM_START}user"


def response_header(body_pid: int, label: str) -> str:
    return f"\n<tool_response>\n[[PAGE {body_pid} {label}]]\n"


def response_trailer() -> str:
    return "\n</tool_response>"


def block_close() -> str:
    return f"{IM_END}\n"


def render_tool_call(name: str, arguments: dict) -> str:
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False) + "\n</tool_call>"


def gauge(resident: int, budget: int | None, gpu: list[tuple[int, str]], host: list[tuple[int, str]],
          extra: str = "") -> str:
    def fmt(items):
        return ",".join(f"{p}{k[0]}" for p, k in items) or "-"
    cap = f"/{budget}" if budget else ""
    s = f"\n[KV] resident {resident}{cap} | GPU: {fmt(gpu)} | host: {fmt(host)}"
    return s + (f" | {extra}" if extra else "")


def escape(text: str) -> str:
    """Neutralise text that could forge page headers or chat structure (CLM's forged-header bug,
    docs/clm_notes.md §7.2)."""
    return text.replace("[[PAGE", "[ [PAGE").replace("<|", "< |").replace("<tool_call>", "< tool_call>") \
        .replace("</tool_response>", "< /tool_response>")


# -- parsing ---------------------------------------------------------------------------------
_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)


def parse_tool_calls(text: str) -> tuple[list[dict], list[tuple[int, int]], int]:
    """Returns (calls, char spans of each parsed call, number of malformed calls)."""
    calls, spans, bad = [], [], 0
    for m in _CALL.finditer(text):
        try:
            obj = json.loads(m.group(1))
            if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
                raise ValueError
            args = obj.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
            if not isinstance(args, dict):
                raise ValueError
            calls.append({"name": obj["name"], "arguments": args})
            spans.append(m.span())
        except (ValueError, json.JSONDecodeError):
            bad += 1
    return calls, spans, bad


def parse_pages(spec) -> list[int]:
    """"3,5-8" / [3, 5] / 3 -> sorted unique ints. Raises ValueError on garbage."""
    if isinstance(spec, int):
        return [spec]
    if isinstance(spec, list):
        return sorted({int(x) for x in spec})
    out: set[int] = set()
    for part in str(spec).replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a), int(b)
            if b < a or b - a > 10000:
                raise ValueError(part)
            out.update(range(a, b + 1))
        else:
            out.add(int(part))
    return sorted(out)


# -- tokenizer adapter -------------------------------------------------------------------------
class Tok:
    """Thin adapter so the same code runs with an HF tokenizer or the test tokenizer."""

    def __init__(self, tok) -> None:
        self.tok = tok
        self.im_end = self.token_id(IM_END)

    def token_id(self, s: str) -> int:
        tid = self.tok.convert_tokens_to_ids(s)
        if tid is None or (isinstance(tid, int) and tid < 0):
            raise KeyError(s)
        return tid

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=False)

    @property
    def vocab_size(self) -> int:
        return len(self.tok)

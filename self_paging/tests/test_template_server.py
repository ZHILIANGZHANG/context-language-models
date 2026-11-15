"""Our incremental rendering vs the official Qwen3 chat template (run on the server).

    QWEN3_TOKENIZER=/path/to/Qwen3-8B python -m pytest -q self_paging/tests/test_template_server.py

Checks (1) the rendered text of a tool-calling exchange equals ``apply_chat_template`` and
(2) tokenizing our segments one by one gives the same ids as tokenizing the whole text, which is
what the append-only log relies on. Skipped when ``QWEN3_TOKENIZER`` is not set.
"""

from __future__ import annotations

import difflib
import os

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("QWEN3_TOKENIZER"), reason="needs the real Qwen3 tokenizer")


def test_render_matches_chat_template():
    from transformers import AutoTokenizer

    from self_paging.core.protocol import (IM_END, Tok, assistant_prefix, block_close, block_open,
                                           kv_tool_schemas, render_system, render_tool_call, render_user,
                                           response_trailer)
    from self_paging.envs.t1_warehouse import SYSTEM, TASK, TOOLS

    hf = AutoTokenizer.from_pretrained(os.environ["QWEN3_TOKENIZER"])
    tok = Tok(hf)
    tools = TOOLS + kv_tool_schemas()
    call = {"name": "next_events", "arguments": {}}
    page = "[step 1] Telemetry: zone=A temp=20.1C."
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": TASK},
            {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": call}]},
            {"role": "tool", "content": page}]
    ref = hf.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=True,
                                 enable_thinking=False)
    segs = [render_system(SYSTEM, tools), render_user(TASK), assistant_prefix(False),
            render_tool_call(**call) + IM_END + "\n", block_open(), "\n<tool_response>\n", page,
            response_trailer(), block_close(), assistant_prefix(False)]
    ours = "".join(segs)
    if ours != ref:
        diff = "\n".join(difflib.unified_diff(ref.splitlines(), ours.splitlines(), "template", "ours", lineterm=""))
        pytest.fail("rendering differs from the chat template:\n" + diff)
    whole = tok.encode(ours)
    pieces = [i for s in segs for i in tok.encode(s)]
    assert pieces == whole, "segment-wise tokenization differs from whole-text tokenization"

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
    # Qwen3's historical renderer removes EMPTY thinking blocks from past
    # assistant turns. It still inserts them at the LIVE generation prefix.
    # Compare historical rendering and live prefix separately; never rewrite
    # previously cached tokens to imitate a non-prefix-stable template.
    segs = [render_system(SYSTEM, tools), render_user(TASK), assistant_prefix(True),
            render_tool_call(**call) + IM_END + "\n", block_open(), "\n<tool_response>\n", page,
            response_trailer(), block_close(), assistant_prefix(False)]
    ours = "".join(segs)
    if ours != ref:
        diff = "\n".join(difflib.unified_diff(ref.splitlines(), ours.splitlines(), "template", "ours", lineterm=""))
        pytest.fail("rendering differs from the chat template:\n" + diff)
    whole = tok.encode(ours)
    header, body, trailer = tok.encode_wrapped(segs[5], segs[6], segs[7])
    pieces = sum([tok.encode(s) for s in segs[:5]], []) + header + body + trailer
    pieces += sum([tok.encode(s) for s in segs[8:]], [])
    assert pieces == whole, "token-aligned tool pages differ from whole-text tokenization"
    live = hf.apply_chat_template(msgs[:2], tools=tools, tokenize=False,
                                 add_generation_prompt=True, enable_thinking=False)
    assert live == render_system(SYSTEM, tools) + render_user(TASK) + assistant_prefix(False)


@pytest.mark.parametrize("body", ["A.", "\n two leading spaces", "warehouse: shelf=12\n", "中文。"])
def test_new_tool_body_has_canonical_tokens(body):
    from transformers import AutoTokenizer
    from self_paging.core.protocol import Tok, response_header, response_trailer
    tok = Tok(AutoTokenizer.from_pretrained(os.environ["QWEN3_TOKENIZER"]))
    h, t = response_header(12, "tool=next_events"), response_trailer()
    parts = tok.encode_wrapped(h, body, t)
    assert sum(parts, []) == tok.encode(h + body + t)
    assert parts[1]


@pytest.mark.parametrize("seed,clean", [(100001, True), (100009, False)])
def test_oracle_conflicting_pages_and_budget_filter(seed, clean):
    from self_paging.train.common import load_tokenizer
    from self_paging.data.build_t1_sft import t1_budget, passes
    from self_paging.envs.t1_warehouse import WarehouseEnv
    from self_paging.oracle.t1_oracle import T1Oracle
    from self_paging.agent.episode import run_episode, ScriptedPolicy, EpisodeConfig
    from self_paging.core.log import TokenLog
    tok = load_tokenizer(os.environ["QWEN3_TOKENIZER"])
    env = WarehouseEnv(seed)
    budget = t1_budget(env, tok, .45)
    rec = run_episode(env, TokenLog(budget), ScriptedPolicy(tok, T1Oracle()), tok,
                      EpisodeConfig(budget=budget))
    assert rec["reward"] == 1
    if clean:
        assert rec["meta"].get("oracle_unsupported", 0) == 0
        assert passes(rec, 1.0)
    else:
        # Correct privileged answers alone must NEVER admit an unsupported
        # budget-constrained trajectory into the teacher training corpus.
        # Since the 2026-11-16 oracle fixes (state-preserving evictions, decision-time shrink) this world's
        # answers are supported by the visible text; the forced eviction alone must keep it out.
        assert rec["meta"]["forced_evictions"] > 0
        assert not passes(rec, 1.0)


def test_growing_directory_respects_total_budget():
    from self_paging.train.common import load_tokenizer
    from self_paging.data.build_t1_sft import t1_budget
    from self_paging.envs.t1_warehouse import WarehouseEnv
    from self_paging.agent.episode import run_episode, ScriptedPolicy, EpisodeConfig
    from self_paging.core.log import TokenLog
    from self_paging.core.protocol import render_tool_call
    tok = load_tokenizer(os.environ["QWEN3_TOKENIZER"])
    env = WarehouseEnv(100000)
    budget = t1_budget(env, tok, .6)
    def no_paging(st):
        return render_tool_call("store", {"shelf": 1}) if st.env.pending else render_tool_call("next_events", {})
    rec = run_episode(env, TokenLog(budget), ScriptedPolicy(tok, no_paging), tok, EpisodeConfig(budget=budget))
    assert rec["peak_resident"] <= budget
    assert rec["meta"]["over_budget"] == 0
    assert rec["meta"]["truncated_tokens"] > 0
    assert len(rec["tokens"]) == len(rec["loss_mask"]) == len(rec["logprobs"])

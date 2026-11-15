"""Privileged-hint teacher (docs/data_and_training.md §2.2).

The oracle decides the action; a real model (Qwen3-8B, or Qwen3.5-9B as backup) writes the short
natural-language reasoning in front of it, having been shown a hint it must not mention. The tool
calls in the final text are always the oracle's own, so the action stays exact; only the prose
comes from the teacher. If the prose leaks the hint or is empty, the template prose is kept and
the turn is counted in ``meta["teacher_fallbacks"]``.

The teacher sees the *visible* text only (decoded from the resident pages), so the reasoning can
be justified by what the student will see.
"""

from __future__ import annotations

import re

import torch

from ..core.protocol import IM_END, IM_START, NO_THINK

LEAK = re.compile(r"hint|oracle|privileged|提示|planned action|told to", re.I)


class HintedTeacher:
    def __init__(self, model, tok, oracle, max_new_tokens: int = 96) -> None:
        self.model, self.tok, self.oracle, self.max_new = model, tok, oracle, max_new_tokens

    def _visible_text(self, st) -> str:
        log = st.log
        res = log.table.resident()
        ids = [t for t, p in zip(log.tokens, log.page_of) if p in res]
        return self.tok.decode(ids)

    @torch.no_grad()
    def __call__(self, st) -> str:
        plan = self.oracle(st)
        cut = plan.find("<tool_call>")
        prose, calls = (plan[:cut].strip(), plan[cut:]) if cut >= 0 else (plan.strip(), "")
        hint = (f"{IM_START}user\n[Private note for you only; never mention it] Your next action will be:\n"
                f"{calls}\nUseful facts: {prose or 'none'}\nWrite one to three sentences of reasoning that "
                f"justify this action using only the log you can see. Do not write any tool call.{IM_END}\n"
                f"{IM_START}assistant\n{NO_THINK}")
        ids = self.tok.encode(self._visible_text(st) + hint)
        device = next(self.model.parameters()).device
        out = self.model.generate(torch.tensor([ids], device=device), max_new_tokens=self.max_new,
                                  do_sample=False, eos_token_id=self.tok.im_end, pad_token_id=self.tok.im_end)
        text = self.tok.decode(out[0, len(ids):].tolist()).split(IM_END)[0].strip()
        text = text.split("<tool_call>")[0].strip()
        if not text or LEAK.search(text) or len(text) > 800:
            st.meta["teacher_fallbacks"] = st.meta.get("teacher_fallbacks", 0) + 1
            text = prose
        return (text + "\n" if text else "") + calls

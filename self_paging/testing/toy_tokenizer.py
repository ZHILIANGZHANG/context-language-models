"""Character-level tokenizer with the Qwen3 special tokens, for CPU tests and smoke runs.

The real Qwen3 tokenizer cannot be downloaded in the cloud environment. This one has the same
interface subset the code uses (``encode``, ``decode``, ``convert_tokens_to_ids``, ``len``).
"""

from __future__ import annotations

import re

SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<tool_call>", "</tool_call>",
            "<tool_response>", "</tool_response>", "<think>", "</think>"]


class ToyTokenizer:
    def __init__(self) -> None:
        chars = ["\n", "\t"] + [chr(c) for c in range(32, 127)]
        self.itos = SPECIALS + chars
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        self.unk = self.stoi["?"]
        self._split = re.compile("(" + "|".join(re.escape(s) for s in SPECIALS) + ")")

    def __len__(self) -> int:
        return len(self.itos)

    def convert_tokens_to_ids(self, s: str) -> int:
        return self.stoi.get(s, -1)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        out = []
        for part in self._split.split(text):
            if not part:
                continue
            if part in self.stoi and part in SPECIALS:
                out.append(self.stoi[part])
            else:
                out.extend(self.stoi.get(ch, self.unk) for ch in part)
        return out

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        return "".join(self.itos[i] for i in ids if not (skip_special_tokens and i < len(SPECIALS)))

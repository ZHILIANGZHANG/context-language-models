# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Per-model LM request compatibility shims.

The GPT-5.4 family on some hosted endpoints 400s on ``max_tokens`` and requires
``max_completion_tokens`` (the llm client's ``USES_COMPLETION_TOKENS`` encodes the
same fact for the OpenAI-compatible client; that module is not on the harness PYTHONPATH, so
the set is mirrored here with a pattern fallback for future family members).

Usage at every completion call site that sets an output cap:

    from ..utils.lm_compat import token_cap_key
    kwargs[token_cap_key(model)] = cap
"""

from __future__ import annotations

# Mirrored from clm/llm/client.py USES_COMPLETION_TOKENS.
_USES_COMPLETION_TOKENS = {
    "gpt-5-4",
    "gpt-5-4-mini",
    "gpt-5-4-nano",
    "gpt-o4-mini",
}


def token_cap_key(model: str) -> str:
    """``"max_completion_tokens"`` for models that reject ``max_tokens``; else ``"max_tokens"``."""
    name = (model or "").split("/")[-1]
    if name in _USES_COMPLETION_TOKENS or name.startswith(("gpt-5", "gpt-o")):
        return "max_completion_tokens"
    return "max_tokens"

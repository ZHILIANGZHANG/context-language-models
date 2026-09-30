# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""KV-cache-aware inference-FLOPs metrics for ATIF-CTX trajectories.

See :mod:`.kv_cache_flops` for the formula, token-source priority (real server
``prompt_token_ids`` -> tiktoken fallback), and the attention-term caveat.
"""

from __future__ import annotations

from .kv_cache_flops import (
    DEFAULT_BLOCK_SIZE,
    N_BODY_PARAMS,
    PrefillResult,
    accumulate_prefill,
    attach_compute_cost,
    attach_flops,
    classify_model_kind,
    compute_trajectory_flops,
    estimate_usd,
    resolve_n_body,
    resolve_text_counter,
    summarize_compute_cost,
)

__all__ = [
    "DEFAULT_BLOCK_SIZE",
    "N_BODY_PARAMS",
    "PrefillResult",
    "accumulate_prefill",
    "attach_compute_cost",
    "attach_flops",
    "classify_model_kind",
    "compute_trajectory_flops",
    "estimate_usd",
    "resolve_n_body",
    "resolve_text_counter",
    "summarize_compute_cost",
]

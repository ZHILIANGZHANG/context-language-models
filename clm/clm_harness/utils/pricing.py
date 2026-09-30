# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Best-effort USD cost for models the local ``litellm`` price map doesn't know.

Some hosted model aliases are not in ``litellm.model_cost``,
so ``litellm.completion_cost`` returns ``0`` for it and every run records ``cost_usd=0``.
Here we register **public list prices** (an ESTIMATE) for such gateway names and expose
``completion_cost_usd`` which falls back to a per-token rate table whenever litellm
yields ``0``/``None`` or raises. Rates are public list prices and are ESTIMATES only —
the true hosted-endpoint cost may differ.
"""

from __future__ import annotations

import logging
from typing import Any

import litellm

logger = logging.getLogger(__name__)

# (input_per_token, output_per_token) in USD/token — public list prices (estimates).
_RATES: dict[str, tuple[float, float]] = {
    "opus": (15.0 / 1_000_000, 75.0 / 1_000_000),
    "sonnet": (3.0 / 1_000_000, 15.0 / 1_000_000),
    "haiku": (0.80 / 1_000_000, 4.0 / 1_000_000),
}

# Gateway model names litellm can't price -> family whose public rate to register.
_REGISTER: dict[str, str] = {
}

_registered = False


def _rate_for(model: str | None) -> tuple[float, float] | None:
    if not model:
        return None
    m = model.lower()
    for key, rate in _RATES.items():
        if key in m:
            return rate
    return None


def rates_for(model: str | None) -> tuple[float, float] | None:
    """Public: ``(input_per_token, output_per_token)`` USD for ``model`` (public list
    price, an ESTIMATE) by family substring, or ``None`` if unknown. Used to turn
    local prefix-match token counts into a provider-independent USD estimate."""
    return _rate_for(model)


def register_default_prices() -> None:
    """Register public list prices with litellm for gateway models it doesn't know.
    Idempotent and best-effort (never blocks a run on pricing)."""
    global _registered
    if _registered:
        return
    _registered = True
    reg: dict[str, dict[str, Any]] = {}
    for name, family in _REGISTER.items():
        in_rate, out_rate = _RATES[family]
        reg[name] = {
            "input_cost_per_token": in_rate,
            "output_cost_per_token": out_rate,
            "litellm_provider": "openai",
            "mode": "chat",
        }
    try:
        litellm.register_model(reg)
    except Exception as exc:  # noqa: BLE001 - pricing must never break a run
        logger.warning("pricing: register_model failed: %s", exc)


def completion_cost_usd(response: Any, model: str | None) -> float:
    """USD for one response. Try litellm; if it yields 0/None or raises, fall back to
    usage x public per-token rates (an ESTIMATE). Returns 0.0 when the model is unknown."""
    try:
        c = litellm.completion_cost(completion_response=response, model=model)
        if c:
            return float(c)
    except Exception:  # noqa: BLE001 - fall back to the rate table
        pass
    rate = _rate_for(model) or _rate_for(getattr(response, "model", None))
    if rate is None:
        return 0.0
    usage = getattr(response, "usage", None)
    pt = int(getattr(usage, "prompt_tokens", 0) or 0) if usage is not None else 0
    ct = int(getattr(usage, "completion_tokens", 0) or 0) if usage is not None else 0
    return pt * rate[0] + ct * rate[1]


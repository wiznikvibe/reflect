"""Approximate Gemini token cost for one design-planning turn."""

from __future__ import annotations

import os

from pydantic import BaseModel, Field

# Gemini 2.5 Flash paid-tier text rates, USD per 1,000,000 tokens.
DEFAULT_INPUT_USD_PER_1M = 0.30
DEFAULT_OUTPUT_USD_PER_1M = 2.50


class TokenUsage(BaseModel):
    """Prompt and completion counts reported by the planner."""

    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)


class UsageMetrics(BaseModel):
    """Latency and approximate spend for one chat turn."""

    latency_ms: float = Field(ge=0)
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    approx_cost_usd: float = Field(ge=0)


def estimate_cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    """Price one turn from ``GEMINI_*_USD_PER_1M_TOKENS``.

    Defaults are Gemini 2.5 Flash text rates when the variables are unset.
    Rates are read on each call.
    """

    input_rate = float(os.environ.get("GEMINI_INPUT_USD_PER_1M_TOKENS", DEFAULT_INPUT_USD_PER_1M))
    output_rate = float(
        os.environ.get("GEMINI_OUTPUT_USD_PER_1M_TOKENS", DEFAULT_OUTPUT_USD_PER_1M)
    )
    return (prompt_tokens * input_rate + completion_tokens * output_rate) / 1_000_000

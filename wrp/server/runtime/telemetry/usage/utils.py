# wrp/server/runtime/telemetry/usage/utils.py
from __future__ import annotations

from typing import Optional

from wrp.server.runtime.telemetry.payloads.types import LlmUsage, UsageCounters

# Canonical set of known numeric fields. Provider-specific extras are merged too.
FIELDS = [
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cached_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "reasoning_tokens",
    "input_characters",
    "output_characters",
    "requests",
]


def _sum_opt(a: Optional[int], b: Optional[int]) -> Optional[int]:
    if a is None and b is None:
        return None
    return (a or 0) + (b or 0)


def merge_counters(dst: UsageCounters, src: UsageCounters) -> UsageCounters:
    """Best-effort numeric merge for known and provider-specific fields."""
    for f in FIELDS:
        setattr(dst, f, _sum_opt(getattr(dst, f, None), getattr(src, f, None)))
    # provider-specific numeric extras
    for k, v in src.model_dump(exclude_none=True).items():
        if k not in FIELDS and isinstance(v, int):
            cur = getattr(dst, k, None)
            setattr(dst, k, _sum_opt(cur, v))
    return dst


def has_any_value(uc: UsageCounters | None) -> bool:
    if not uc:
        return False
    return any(getattr(uc, f) is not None for f in FIELDS)


def build_counters_from_params(
    *,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    total_tokens: int | None = None,
    cached_tokens: int | None = None,
    cache_read_tokens: int | None = None,
    cache_write_tokens: int | None = None,
    reasoning_tokens: int | None = None,
    input_characters: int | None = None,
    output_characters: int | None = None,
    requests: int | None = None,
    **extras: int,
) -> UsageCounters:
    uc = UsageCounters(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cached_tokens=cached_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        reasoning_tokens=reasoning_tokens,
        input_characters=input_characters,
        output_characters=output_characters,
        requests=requests,
        **extras,
    )
    # backfill total if trivially derivable
    if uc.total_tokens is None and uc.input_tokens is not None and uc.output_tokens is not None:
        uc.total_tokens = uc.input_tokens + uc.output_tokens
    return uc


def build_llm_usage_from_params(
    *,
    provider: str | None = None,
    model: str | None = None,
    cache_hit: bool | None = None,
    details: dict | None = None,
    counters: UsageCounters,
) -> LlmUsage:
    return LlmUsage(
        provider=provider,
        model=model,
        counters=counters,
        cache_hit=cache_hit,
        details=details,
    )


def normalize_llm_usage(usage: LlmUsage | None) -> LlmUsage | None:
    """Fill trivially derivable counters (e.g., total_tokens)."""
    if usage is None:
        return None
    c = usage.counters
    if c and c.total_tokens is None and c.input_tokens is not None and c.output_tokens is not None:
        c.total_tokens = c.input_tokens + c.output_tokens
    return usage
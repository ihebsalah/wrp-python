# wrp/server/runtime/telemetry/privacy/guards.py
from __future__ import annotations

from typing import Iterable, Optional

from .policy import TelemetryResourcePolicy
from wrp.server.runtime.store.base import Store


def _span_kind_for(
    events: Iterable[object],
    span_id: str,
) -> Optional[str]:
    """
    Best-effort: scan telemetry events to find the span_kind for a given span_id.
    Returns one of {"run","agent","llm","tool","guardrail","annotation","handoff"} or None if not found.
    """
    for ev in events:
        # TelemetryEvent has attributes: kind="span", span_id, span_kind, ...
        if getattr(ev, "kind", None) == "span" and getattr(ev, "span_id", None) == span_id:
            return getattr(ev, "span_kind", None)
    return None


async def is_private_only_span_payload(
    system_session_id: str,
    run_id: str,
    span_id: str,
    policy: TelemetryResourcePolicy,
    store: Store,
) -> bool:
    """
    Returns True iff, under the current `policy`, *all* payload kinds for this span
    (start/end or point) would be private.

    If we cannot determine span kind, returns True (fail-closed).
    """
    # Find span_kind from telemetry (best-effort).
    try:
        events = await store.load_telemetry(system_session_id, run_id, kinds={"span"})
    except Exception:
        # Fail-closed: if we can't load, treat as private-only
        return True

    span_kind = _span_kind_for(events, span_id)
    if not span_kind:
        # Best-effort fallback: try the span payload envelope (has span_kind)
        try:
            env = await store.get_span_payload(system_session_id, run_id, span_id)
            span_kind = getattr(env, "span_kind", None) if env else None
        except Exception:
            span_kind = None
    if not span_kind:
        # Still unknown -> fail-closed
        return True

    # Map span kind to payload kinds this span may expose.
    if span_kind in ("run", "agent", "llm", "tool"):
        candidate_kinds = (f"{span_kind}.start", f"{span_kind}.end")
    elif span_kind in ("guardrail", "annotation", "handoff"):
        # point-only; prefer concrete payload kind names
        candidate_kinds = ("guardrail_result",) if span_kind == "guardrail" else (span_kind,)
    else:
        # unknown kind -> fail-closed
        return True

    return all(policy.mode_for_kind(k) == "private" for k in candidate_kinds)
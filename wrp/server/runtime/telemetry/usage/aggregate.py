# wrp/server/runtime/telemetry/usage/aggregate.py
from __future__ import annotations
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from wrp.server.runtime.store.base import Store
from wrp.server.runtime.telemetry.events import (
    AgentSpanStart, AgentSpanEnd, LlmSpanEnd, TelemetryEvent,
)
from wrp.server.runtime.telemetry.payloads.types import LlmEndPayload, LlmUsage, UsageCounters
from .utils import has_any_value, merge_counters
from .types import AggregationDiagnostics


async def aggregate_agent_usage(
    store: Store,
    system_session_id: str,
    run_id: str,
    *,
    agent_span_id: str,
    agent_id: Optional[str],
    agent_end_ts: Optional[datetime] = None,
) -> Tuple[UsageCounters | None, List[LlmUsage] | None, AggregationDiagnostics]:
    """
    Flat-model aggregation (system-session aware):
      - Use the Agent span's actual time window [start_ts, end_ts] to select LLM.Ends.
      - Attribute only LLM.End events whose agent_id == this agent_id AND fall within window.
      - Produce diagnostics for missing tags, overlapping windows, etc.
    """
    diag = AggregationDiagnostics()
    if not agent_id:
        diag.missing_agent_id = True
        diag.notes.append("agent_end: missing agent_id; per-agent usage disabled")
        return None, None, diag

    events: List[TelemetryEvent] = await store.load_telemetry(system_session_id, run_id, kinds={"span"})

    # 1) Find this agent window (start ts; end ts may be current if end event not yet appended)
    start_ts: Optional[datetime] = None
    end_ts: Optional[datetime] = None
    for ev in events:
        if isinstance(ev, AgentSpanStart) and ev.span_id == agent_span_id:
            start_ts = getattr(ev, "ts", None)
            break
    # We may be called before AgentSpanEnd is recorded, so accept injected end_ts
    if agent_end_ts:
        end_ts = agent_end_ts
    else:
        for ev in events:
            if isinstance(ev, AgentSpanEnd) and ev.span_id == agent_span_id:
                end_ts = getattr(ev, "ts", None)
                break

    if not start_ts or not end_ts:
        diag.window_missing = True
        diag.notes.append("agent_end: could not resolve full agent time window")
        # Without a window, any inferences are unsafe
        return None, None, diag

    # 2) Detect ambiguous overlaps: any other Agent spans with same agent_id and overlapping window
    for ev in events:
        if isinstance(ev, AgentSpanStart) and ev.span_id != agent_span_id and getattr(ev, "agent_id", None) == agent_id:
            other_start = getattr(ev, "ts", None)
            # find end for the other
            other_end: Optional[datetime] = None
            for ev2 in events:
                if isinstance(ev2, AgentSpanEnd) and ev2.span_id == ev.span_id:
                    other_end = getattr(ev2, "ts", None)
                    break
            # Treat open-ended as "now" for overlap checks
            other_end = other_end or datetime.now(timezone.utc)
            if other_start and not (other_end <= start_ts or end_ts <= other_start):
                diag.ambiguous_agent_id = True
                diag.notes.append("agent_end: overlapping agent windows detected for same agent_id")
                return None, None, diag

    # 3) Gather LLM.End events in window; split by tag matching
    def _in_window(ts: Optional[datetime]) -> bool:
        if not isinstance(ts, datetime):
            return False
        return (start_ts <= ts <= end_ts)

    llm_all_in_window: List[LlmSpanEnd] = [e for e in events if isinstance(e, LlmSpanEnd) and _in_window(getattr(e, "ts", None))]
    diag.considered_llm_events = len(llm_all_in_window)
    diag.untagged_in_window = sum(1 for e in llm_all_in_window if getattr(e, "agent_id", None) is None)

    llm_end_events: List[LlmSpanEnd] = [e for e in llm_all_in_window if getattr(e, "agent_id", None) == agent_id]
    if not llm_end_events:
        # Nothing to aggregate for this agent; still provide diagnostics
        return None, None, diag

    # Keep deterministic, chronological order for per-LLM details
    llm_end_events.sort(key=lambda e: getattr(e, "ts", None) or datetime.min)

    total = UsageCounters()
    llms: List[LlmUsage] = []

    for e in llm_end_events:
        env = await store.get_span_payload(system_session_id, run_id, e.span_id)
        if not env:
            continue
        part = env.capture.get("end")
        data = getattr(part, "data", None) if part else None
        if not isinstance(data, dict):
            continue
        try:
            payload = LlmEndPayload.model_validate(data)
        except Exception:
            continue  # malformed; skip safely
        if not payload.usage:
            continue

        # Ensure trivially derivable fields are present
        usage = payload.usage
        c = usage.counters
        if c and c.total_tokens is None and c.input_tokens is not None and c.output_tokens is not None:
            c.total_tokens = c.input_tokens + c.output_tokens
        llms.append(usage)
        if c:
            merge_counters(total, c)

    diag.included_llm_events = len(llms)
    if not llms and not has_any_value(total):
        return None, None, diag

    return (total if has_any_value(total) else None, llms or None, diag)
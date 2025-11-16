# wrp/server/runtime/telemetry/views.py
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from .events import (
    AgentSpanEnd,
    AgentSpanStart,
    AnnotationSpanPoint,
    GuardrailSpanPoint,
    HandoffSpanPoint,
    LlmSpanEnd,
    LlmSpanStart,
    RunSpanEnd,
    RunSpanStart,
    TelemetryEvent,
    TelemetrySpanView,
    ToolSpanEnd,
    ToolSpanStart,
)


def build_span_index(
    events: Iterable[TelemetryEvent],
    *,
    mask_model: bool = False,
    mask_tool: bool = False,
) -> Dict[str, TelemetrySpanView]:
    """
    Build an in-memory index of spans from span events.

    Returns:
        spans_by_id: span_id -> TelemetrySpanView
    """
    spans: Dict[str, Dict[str, Any]] = {}

    def ensure(sid: str, kind: str, name: str, ts: Optional[datetime]) -> Dict[str, Any]:
        if sid not in spans:
            spans[sid] = {
                "span_id": sid,
                "span_kind": kind,
                "name": name,
                "started_at": None,
                "ended_at": None,
                "duration_ms": None,
                "status": None,
                "ts": ts,
                # headers (filled best-effort by kind)
                "agent": None,
                "agent_id": None,
                "model": None,
                "tool": None,
                "level": None,
                "message_preview": None,
                "guardrail_kind": None,
                "guardrail_status": None,
                "guardrail_name": None,
                "tripwire_triggered": None,
                "from_agent": None,
                "to_agent": None,
                "payload_id": None,
            }
        return spans[sid]

    for ev in events:
        if getattr(ev, "kind", None) != "span":
            continue

        sid = ev.span_id
        name = getattr(ev, "name", ev.__class__.__name__)
        kind = getattr(ev, "span_kind", "unknown")
        ts = getattr(ev, "ts", None)

        row = ensure(sid, kind, name, ts)

        # pluck payload id if provided by service meta
        meta = getattr(ev, "meta", {}) or {}
        if isinstance(meta, dict) and "payload_id" in meta:
            row["payload_id"] = meta["payload_id"]

        # fill common header fields by event type
        if isinstance(ev, (AgentSpanStart, AgentSpanEnd)):
            row["agent"] = getattr(ev, "agent", row.get("agent"))
            row["agent_id"] = getattr(ev, "agent_id", row.get("agent_id"))

        if isinstance(ev, (LlmSpanStart, LlmSpanEnd)):
            row["agent"] = getattr(ev, "agent", row.get("agent"))
            row["agent_id"] = getattr(ev, "agent_id", row.get("agent_id"))
            row["model"] = getattr(ev, "model", row.get("model"))

        if isinstance(ev, (ToolSpanStart, ToolSpanEnd)):
            row["agent"] = getattr(ev, "agent", row.get("agent"))
            row["agent_id"] = getattr(ev, "agent_id", row.get("agent_id"))
            row["tool"] = getattr(ev, "tool", row.get("tool"))

        # lifecycle mapping
        if isinstance(ev, (RunSpanStart, AgentSpanStart, LlmSpanStart, ToolSpanStart)):
            row["started_at"] = ts or row.get("started_at")

        elif isinstance(ev, (RunSpanEnd, AgentSpanEnd, LlmSpanEnd, ToolSpanEnd)):
            row["ended_at"] = ts or row.get("ended_at")
            row["duration_ms"] = getattr(ev, "duration_ms", row.get("duration_ms"))
            row["status"] = getattr(ev, "status", row.get("status"))

        # point spans
        elif isinstance(ev, HandoffSpanPoint):
            row["started_at"] = ts
            row["ended_at"] = ts
            row["duration_ms"] = 0
            row["from_agent"] = ev.from_agent
            row["to_agent"] = ev.to_agent
            row["agent_id"] = row.get("agent_id") or ev.from_agent_id or ev.to_agent_id
            row["model"] = row.get("model") or ev.from_model or ev.to_model

        elif isinstance(ev, AnnotationSpanPoint):
            row["started_at"] = ts
            row["ended_at"] = ts
            row["duration_ms"] = 0
            row["level"] = ev.level
            row["message_preview"] = getattr(ev, "message_preview", None)

        elif isinstance(ev, GuardrailSpanPoint):
            row["started_at"] = ts
            row["ended_at"] = ts
            row["duration_ms"] = 0
            row["guardrail_kind"] = ev.guardrail_kind
            row["guardrail_status"] = ev.status
            row["guardrail_name"] = getattr(ev, "guardrail_name", None)
            row["tripwire_triggered"] = getattr(ev, "tripwire_triggered", None)
            row["agent"] = getattr(ev, "agent", row.get("agent"))
            row["agent_id"] = getattr(ev, "agent_id", row.get("agent_id"))

    typed: Dict[str, TelemetrySpanView] = {}
    for sid, data in spans.items():
        if mask_model:
            data["model"] = None
        if mask_tool:
            data["tool"] = None
        typed[sid] = TelemetrySpanView(**data)
    return typed


def _sort_key(span: TelemetrySpanView) -> str:
    dt = span.started_at or span.ts
    return dt.isoformat() if isinstance(dt, datetime) else ""


def list_span_views(spans: Dict[str, TelemetrySpanView]) -> List[TelemetrySpanView]:
    """Return span views sorted by start timestamp (or fallback ts)."""
    return sorted(spans.values(), key=_sort_key)


def get_span_view(spans: Dict[str, TelemetrySpanView], span_id: str) -> Optional[TelemetrySpanView]:
    """Return a single span view by id."""
    return spans.get(span_id)
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
    ToolSpanEnd,
    ToolSpanStart,
)


def _iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if isinstance(dt, datetime) else None


def build_span_index(events: Iterable[TelemetryEvent]) -> Dict[str, Dict[str, Any]]:
    """
    Build an in-memory index of spans from span events.

    Returns:
        spans_by_id: span_id -> span dict (json-friendly except datetime values)
    """
    spans: Dict[str, Dict[str, Any]] = {}

    def ensure(sid: str, kind: str, name: str, ts: datetime) -> Dict[str, Any]:
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
                "payload_uri": None,
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

        # pluck payload uri if provided by service meta
        meta = getattr(ev, "meta", {}) or {}
        refs = meta.get("refs") if isinstance(meta, dict) else None
        if isinstance(refs, dict) and "payload" in refs:
            row["payload_uri"] = refs["payload"]

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

    return spans


def serialize_span_list(spans: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return a JSON-friendly list view sorted by start/ts."""
    rows: List[Dict[str, Any]] = []
    for s in spans.values():
        rows.append(
            {
                "span_id": s["span_id"],
                "span_kind": s["span_kind"],
                "name": s["name"],
                "ts": _iso(s.get("ts")),
                "started_at": _iso(s.get("started_at")),
                "ended_at": _iso(s.get("ended_at")),
                "duration_ms": s.get("duration_ms"),
                "status": s.get("status"),
                # headers
                "agent": s.get("agent"),
                "agent_id": s.get("agent_id"),
                "model": s.get("model"),
                "tool": s.get("tool"),
                "level": s.get("level"),
                "message_preview": s.get("message_preview"),
                "guardrail_kind": s.get("guardrail_kind"),
                "guardrail_status": s.get("guardrail_status"),
                "guardrail_name": s.get("guardrail_name"),
                "tripwire_triggered": s.get("tripwire_triggered"),
                "from_agent": s.get("from_agent"),
                "to_agent": s.get("to_agent"),
                "payload_uri": s.get("payload_uri"),
                "children_count": 0,
            }
        )

    rows.sort(key=lambda r: (r["started_at"] or r["ts"] or ""))
    return rows


def serialize_span_detail(spans: Dict[str, Dict[str, Any]], span_id: str) -> Optional[Dict[str, Any]]:
    """Return a JSON-friendly detail view with child ordering."""
    s = spans.get(span_id)
    if not s:
        return None
    out = {
        "span_id": s["span_id"],
        "span_kind": s["span_kind"],
        "name": s["name"],
        "ts": _iso(s.get("ts")),
        "started_at": _iso(s.get("started_at")),
        "ended_at": _iso(s.get("ended_at")),
        "duration_ms": s.get("duration_ms"),
        "status": s.get("status"),
        # headers
        "agent": s.get("agent"),
        "agent_id": s.get("agent_id"),
        "model": s.get("model"),
        "tool": s.get("tool"),
        "level": s.get("level"),
        "message_preview": s.get("message_preview"),
        "guardrail_kind": s.get("guardrail_kind"),
        "guardrail_status": s.get("guardrail_status"),
        "guardrail_name": s.get("guardrail_name"),
        "tripwire_triggered": s.get("tripwire_triggered"),
        "from_agent": s.get("from_agent"),
        "to_agent": s.get("to_agent"),
        "payload_uri": s.get("payload_uri"),
        "children": [],
    }
    return out
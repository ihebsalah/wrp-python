# wrp/server/runtime/telemetry/events.py
from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Literal, Union, Optional
from pydantic import BaseModel, Field
from wrp.server.runtime.runs.types import RunOutcome  # for RunSpanEnd.outcome
from wrp.server.runtime.runs.types import RunSettings


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


class TelemetryBase(BaseModel):
    ts: datetime = Field(default_factory=_now_utc)
    # Be liberal in what we accept: tolerate unknown fields from older/newer writers.
    model_config = {"extra": "ignore"}


SpanKind = Literal["run", "agent", "llm", "tool", "handoff", "annotation", "guardrail"]
SpanPhase = Literal["start", "end", "point"]


class SpanBase(TelemetryBase):
    kind: Literal["span"] = "span"
    phase: SpanPhase
    span_kind: SpanKind
    span_id: str
    name: str
    meta: dict[str, Any] = Field(default_factory=dict)


class SpanEndMixin(BaseModel):
    duration_ms: Optional[int] = None
    status: Optional[Literal["ok", "error"]] = None

# -------- RUN --------
class RunSpanStart(SpanBase):
    span_kind: SpanKind = "run"
    phase: SpanPhase = "start"
    thread_id: str | None = None
    input_keys: list[str] | None = None
    input_size_bytes: int | None = None
    settings: RunSettings


class RunSpanEnd(SpanBase, SpanEndMixin):
    span_kind: SpanKind = "run"
    phase: SpanPhase = "end"
    outcome: RunOutcome | None = None
    output_keys: list[str] | None = None
    output_size_bytes: int | None = None


# -------- AGENT --------
class AgentSpanStart(SpanBase):
    span_kind: SpanKind = "agent"
    phase: SpanPhase = "start"
    agent: str  # same as name^
    agent_id: str | None = None
    model: str | None = None


class AgentSpanEnd(SpanBase, SpanEndMixin):
    span_kind: SpanKind = "agent"
    phase: SpanPhase = "end"
    agent: str
    agent_id: str | None = None


# -------- LLM --------
class LlmSpanStart(SpanBase):
    span_kind: SpanKind = "llm"
    phase: SpanPhase = "start"
    agent: str
    agent_id: str | None = None
    model: str | None = None


class LlmSpanEnd(SpanBase, SpanEndMixin):
    span_kind: SpanKind = "llm"
    phase: SpanPhase = "end"
    agent: str
    agent_id: str | None = None
    model: str | None = None


# -------- TOOL --------
class ToolSpanStart(SpanBase):
    span_kind: SpanKind = "tool"
    phase: SpanPhase = "start"
    agent: str | None = None
    agent_id: str | None = None
    tool: str


class ToolSpanEnd(SpanBase, SpanEndMixin):
    span_kind: SpanKind = "tool"
    phase: SpanPhase = "end"
    agent: str | None = None
    agent_id: str | None = None
    tool: str


# -------- POINT SPANS --------
class HandoffSpanPoint(SpanBase):
    span_kind: SpanKind = "handoff"
    phase: SpanPhase = "point"
    from_agent: str
    to_agent: str
    # minimal public headers only; full snapshots live in payload
    from_agent_id: str | None = None
    to_agent_id: str | None = None
    from_model: str | None = None
    to_model: str | None = None


class AnnotationSpanPoint(SpanBase):
    span_kind: SpanKind = "annotation"
    phase: SpanPhase = "point"
    # keep sensitive message/data in payload; only level is public
    level: Literal["debug", "info", "warning", "error"] = "info"
    # short preview shown in span headers (full text stays in payload)
    message_preview: str | None = None


class GuardrailSpanPoint(SpanBase):
    span_kind: SpanKind = "guardrail"
    phase: SpanPhase = "point"
    guardrail_kind: Literal["input", "output"]
    status: Literal["ok", "trip", "error"]
    # quick filters in headers (non-sensitive):
    guardrail_name: str | None = None
    tripwire_triggered: bool | None = None
    agent: str | None = None
    agent_id: str | None = None


TelemetryEvent = Union[
    RunSpanStart,
    RunSpanEnd,
    AgentSpanStart,
    AgentSpanEnd,
    LlmSpanStart,
    LlmSpanEnd,
    ToolSpanStart,
    ToolSpanEnd,
    HandoffSpanPoint,
    AnnotationSpanPoint,
    GuardrailSpanPoint,
]


class TelemetrySpanView(BaseModel):
    """Typed span view used in list/read APIs."""

    span_id: str
    span_kind: SpanKind
    name: str
    ts: datetime | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None
    duration_ms: int | None = None
    status: Literal["ok", "error"] | None = None
    agent: str | None = None
    agent_id: str | None = None
    model: str | None = None
    tool: str | None = None
    level: Literal["debug", "info", "warning", "error"] | None = None
    message_preview: str | None = None
    guardrail_kind: Literal["input", "output"] | None = None
    guardrail_status: Literal["ok", "trip", "error"] | None = None
    guardrail_name: str | None = None
    tripwire_triggered: bool | None = None
    from_agent: str | None = None
    to_agent: str | None = None
    payload_uri: str | None = None
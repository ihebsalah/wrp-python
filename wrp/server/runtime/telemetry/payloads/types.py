# wrp/server/runtime/telemetry/payloads/types.py
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional, Type, Union

from pydantic import BaseModel, Field

from wrp.server.runtime.runs.types import RunOutcome


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------- Core envelope ----------------

class PayloadPart(BaseModel):
    mime_type: str = "application/json"
    size_bytes: Optional[int] = None
    sha256: Optional[str] = None
    data: Any | None = None


class SpanPayloadEnvelope(BaseModel):
    """
    On-disk/in-store payload capture for a span.
    Either/both of capture.start and capture.end may be present.
    """

    run_id: str
    span_id: str
    span_kind: Literal["run", "agent", "llm", "tool", "handoff", "annotation", "guardrail"]
    capture: dict[str, PayloadPart] = Field(default_factory=dict)  # keys: "start", "end", "point"
    redacted: bool = False
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)

    class ResolvedPayload(BaseModel):
        """
        Single resolved payload for this envelope.
        - kind: canonical "<span_kind>.<part>" (e.g., "llm.end", "annotation.point")
        - part: which capture part was used
        - data: best-effort typed payload instance, or the original dict/primitive if coercion failed
        """
        kind: str
        part: Literal["start", "end", "point"]
        data: "PayloadData | dict[str, Any] | None"

    def resolve_payload_data(self) -> "SpanPayloadEnvelope.ResolvedPayload | None":
        """
        Return the single most relevant payload for this envelope, already coerced to a
        typed model when possible. If no suitable capture part exists, return None.

        Preference order:
          - run/agent/llm/tool: "end" then "start"
          - annotation/handoff/guardrail: "point"
        """
        order_by_kind = {
            "run": ("end", "start"),
            "agent": ("end", "start"),
            "llm": ("end", "start"),
            "tool": ("end", "start"),
            "annotation": ("point",),
            "handoff": ("point",),
            "guardrail": ("point",),
        }
        for part in order_by_kind.get(self.span_kind, ("end", "start", "point")):
            p = self.capture.get(part)
            if p is None:
                continue
            coerced = coerce_payload_data_by_context(self.span_kind, part, p.data)
            return SpanPayloadEnvelope.ResolvedPayload(kind=f"{self.span_kind}.{part}", part=part, data=coerced)
        return None


# ---------------- Usage types (unchanged API) ----------------

class UsageCounters(BaseModel):
    """
    Provider-agnostic counters. Authors can fill what they have; we don't require all.
    """
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    # Optional extras
    cached_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    reasoning_tokens: int | None = None
    # Non-token units (optional)
    input_characters: int | None = None
    output_characters: int | None = None
    requests: int | None = None

    model_config = {"extra": "allow"}  # provider-specific extensions ok


class LlmUsage(BaseModel):
    kind: Literal["usage.llm"] = "usage.llm"
    provider: str | None = None  # e.g., "openai", "anthropic"
    model: str | None = None
    counters: UsageCounters
    cache_hit: bool | None = None
    details: dict[str, Any] | None = None

    model_config = {"extra": "allow"}


class AgentUsageAggregate(BaseModel):
    """
    Aggregate usage at agent level. Either fill `total`, or provide `llms`, or both.
    """
    kind: Literal["usage.agent"] = "usage.agent"
    agent: str
    total: UsageCounters | None = None
    llms: list[LlmUsage] | None = None


# ---------------- Payload base classes ----------------

class PayloadBase(BaseModel):
    """
    Common fields available to all payloads.
    """
    # Every concrete subclass still sets its own `kind` Literal
    meta: dict[str, Any] | None = None
    tags: list[str] | None = None

    model_config = {"extra": "allow"}  # forgiving by default


class AgentScope(BaseModel):
    """
    Mixin for payloads that relate to an agent (agent name/ID).
    """
    agent: str | None = None
    agent_id: str | None = None

    model_config = {"extra": "allow"}


class RunPayloadBase(PayloadBase):
    workflow_name: str


class AgentPayloadBase(PayloadBase, AgentScope):
    pass


class LlmPayloadBase(PayloadBase, AgentScope):
    pass


class ToolPayloadBase(PayloadBase, AgentScope):
    tool: str


# ---------------- Run payloads ----------------

class RunInputPayload(RunPayloadBase):
    kind: Literal["run.start"] = "run.start"
    # Serialized WorkflowInput (already validated/cleaned by the caller)
    run_input: dict[str, Any]


class RunOutputPayload(RunPayloadBase):
    kind: Literal["run.end"] = "run.end"
    outcome: RunOutcome
    # Serialized WorkflowOutput when available
    output: dict[str, Any] | None = None
    # Optional error text when outcome != success
    error: str | None = None


# ---------------- Agent payloads ----------------

class AgentStartPayload(AgentPayloadBase):
    kind: Literal["agent.start"] = "agent.start"
    # Snapshot of agent configuration at start
    system_prompt: str | None = None
    # A snapshot of the agent-level model settings (note: may differ from resolved per-call)
    model_settings: dict[str, Any] | None = None
    # The expected output type, e.g., schema or descriptive shape
    output_type: Any | None = None
    # List of tool descriptors available to the agent
    tools: list[dict[str, Any]] | None = None
    # MCP servers configuration
    mcp_servers: list[dict[str, Any]] | None = None
    mcp_config: dict[str, Any] | None = None
    # Guardrails configuration
    input_guardrails: list[dict[str, Any]] | None = None
    output_guardrails: list[dict[str, Any]] | None = None


class AgentEndPayload(AgentPayloadBase):
    kind: Literal["agent.end"] = "agent.end"
    # The final, structured output from the agent
    final_output: Any | None = None
    # Optional error string if the agent failed to produce a final output
    error: str | None = None
    # Either provide a pre-aggregated total, or a per-LLM breakdown, or both:
    usage_total: UsageCounters | None = None  # simple aggregate
    usage_llms: list[LlmUsage] | None = None  # detailed per-LLM breakdown


# ---------------- Point-span payloads ----------------

class HandoffAgentSnapshot(BaseModel):
    """
    Mirrors agent_start fields for snapshotting both sides of a handoff.
    """
    agent: str
    agent_id: str | None = None
    model: str | None = None
    system_prompt: str | None = None
    model_settings: dict[str, Any] | None = None
    output_type: Any | None = None
    tools: list[dict[str, Any]] | None = None
    mcp_servers: list[dict[str, Any]] | None = None
    mcp_config: dict[str, Any] | None = None
    input_guardrails: list[dict[str, Any]] | None = None
    output_guardrails: list[dict[str, Any]] | None = None


class HandoffPayload(PayloadBase):
    kind: Literal["handoff"] = "handoff"
    from_agent: HandoffAgentSnapshot
    to_agent: HandoffAgentSnapshot
    meta: dict[str, Any] | None = None


class AnnotationPayload(PayloadBase):
    kind: Literal["annotation"] = "annotation"
    level: Literal["debug", "info", "warning", "error"] = "info"
    # message/data stay only in encrypted payload
    message: str | None = None
    data: dict[str, Any] | None = None


class GuardrailResultPayload(PayloadBase):
    kind: Literal["guardrail_result"] = "guardrail_result"
    # "input" or "output"
    guardrail_kind: Literal["input", "output"]
    # human-readable name from guardrail.get_name()
    guardrail_name: str | None = None
    # normalized status: "ok" | "trip" | "error"
    status: Literal["ok", "trip", "error"] = "ok"
    # explicit flag (redundant with status, but convenient)
    tripwire_triggered: bool = False
    # who this pertains to (optional but useful)
    agent: str | None = None
    agent_id: str | None = None
    # Structured output from the guardrail function (GuardrailFunctionOutput.output_info)
    output_info: Any | None = None
    # For OUTPUT guardrails only: the agent output that was checked
    agent_output: Any | None = None
    # For INPUT guardrails (optional, privacy-sensitive): the input items that were checked
    input_items: list[Any] | None = None
    # If the guardrail itself failed
    error: str | None = None


# ---------------- LLM payloads ----------------

class LlmStartPayload(LlmPayloadBase):
    kind: Literal["llm.start"] = "llm.start"
    # The specific model used for the request
    model: str | None = None
    # A snapshot of the model settings (e.g., temperature, max_tokens)
    model_settings: dict[str, Any] | None = None
    # Optional system prompt (if present in the model call)
    system_prompt: str | None = None
    # Vendor-agnostic list of request input items (e.g., OpenAI Responses input items)
    input_items: list[Any] = Field(default_factory=list)


class LlmEndPayload(LlmPayloadBase):
    kind: Literal["llm.end"] = "llm.end"
    # Raw/vendor-agnostic response object (SDK should serialize as JSON-friendly)
    response: Any | None = None
    # Optional error string; when present we’ll infer span status="error"
    error: str | None = None
    # Standardized usage information
    usage: LlmUsage | None = None


# ---------------- Tool payloads ----------------

class ToolStartPayload(ToolPayloadBase):
    kind: Literal["tool.start"] = "tool.start"
    # Arbitrary tool arguments (already serialized/validated by SDK layer as needed)
    args: Any = None


class ToolEndPayload(ToolPayloadBase):
    kind: Literal["tool.end"] = "tool.end"
    # Raw tool result (JSON-serializable recommended)
    result: Any | None = None
    # Optional error string; if present we’ll mark the span as error
    error: str | None = None

# ---------------- Typed view helpers (context-based) ----------------

# Union of all concrete payload models we know how to parse.
PayloadData = Union[
    RunInputPayload, RunOutputPayload,
    AgentStartPayload, AgentEndPayload,
    LlmStartPayload, LlmEndPayload,
    ToolStartPayload, ToolEndPayload,
    AnnotationPayload, HandoffPayload, GuardrailResultPayload,
]

# Map (span_kind, part_name) → model class
_MODEL_BY_CONTEXT: dict[tuple[str, str], Type[PayloadBase]] = {
    ("run", "start"): RunInputPayload,
    ("run", "end"): RunOutputPayload,

    ("agent", "start"): AgentStartPayload,
    ("agent", "end"): AgentEndPayload,

    ("llm", "start"): LlmStartPayload,
    ("llm", "end"): LlmEndPayload,

    ("tool", "start"): ToolStartPayload,
    ("tool", "end"): ToolEndPayload,

    ("annotation", "point"): AnnotationPayload,
    ("handoff", "point"): HandoffPayload,
    ("guardrail", "point"): GuardrailResultPayload,
}

def coerce_payload_data_by_context(
    span_kind: Literal["run", "agent", "llm", "tool", "handoff", "annotation", "guardrail"],
    part_name: Literal["start", "end", "point"],
    data: Any,
) -> PayloadData | dict[str, Any] | None:
    """
    Try to parse an arbitrary `data` into the expected payload model based on
    (span_kind, capture part_name). On failure, return the original `data`.
    """
    if data is None:
        return None

    # Already a pydantic payload instance?
    if isinstance(data, PayloadBase):
        return data  # type: ignore[return-value]

    model = _MODEL_BY_CONTEXT.get((span_kind, part_name))
    if model is None:
        # Unknown or mismatched context — do not coerce.
        return data

    # Dict-like: attempt validation. If it fails (e.g., because author redacted fields),
    # keep the original value to remain tolerant to masking.
    if isinstance(data, dict):
        try:
            return model.model_validate(data)  # type: ignore[return-value]
        except Exception:
            return data

    # Anything else (string "***", list, etc.) — return as-is.
    return data
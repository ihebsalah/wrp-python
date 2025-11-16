# wrp/server/runtime/telemetry/service.py
from __future__ import annotations

import hashlib
import json
import time
import uuid
from typing import Any, Dict, Iterable, Literal, cast, Callable, Awaitable

from pydantic import BaseModel

import wrp.types as types
from wrp.server.runtime.runs.types import RunMeta, RunOutcome, RunState, RunSettings
from wrp.server.runtime.telemetry.payloads.types import (
    AgentEndPayload,
    AgentStartPayload,
    AnnotationPayload,
    GuardrailResultPayload,
    HandoffAgentSnapshot,
    HandoffPayload,
    LlmEndPayload,
    LlmStartPayload,
    PayloadPart,
    RunInputPayload,
    RunOutputPayload,
    SpanPayloadEnvelope,
    ToolEndPayload,
    ToolStartPayload,
)
from wrp.server.runtime.store.base import Store
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
    SpanKind,
    TelemetryEvent,
    ToolSpanEnd,
    ToolSpanStart,
)
from wrp.server.runtime.telemetry.usage.aggregate import aggregate_agent_usage
from wrp.server.runtime.telemetry.usage.types import AggregationDiagnostics
from wrp.server.runtime.telemetry.usage.utils import (
    build_counters_from_params,
    build_llm_usage_from_params,
    normalize_llm_usage,
)


class RunTelemetryService:
    """Structured, span-based telemetry API bound to a single run."""

    def __init__(
        self,
        store: Store,
        current_run: RunMeta,
        emit_system_event: Callable[..., Awaitable[None]] | None = None,
    ):
        self._store = store
        self._run = current_run
        self._emit_system_event = emit_system_event
        # minimal bookkeeping
        self._span_t0_ns: Dict[str, int] = {}
        self._span_name: Dict[str, str] = {}
        # fast path to drop events after conclusion (avoid store hit on every record)
        self._run_concluded: bool = False

    def _dump(self, obj: Any) -> Any:
        """Return JSON-ready dict with None fields removed for BaseModel inputs."""
        if isinstance(obj, BaseModel):
            return obj.model_dump(exclude_none=True)
        return obj

    # usage normalization lives in wrp.server.wrp.telemetry.usage.utils.normalize_llm_usage

    async def record(self, event_or_events: TelemetryEvent | Iterable[TelemetryEvent], *, force: bool = False) -> None:
        """
        Record a single telemetry event or an iterable of events.
        Events are silently dropped if the run has already concluded.
        """
        if (not force) and self._run_concluded:
            return
        meta = await self._store.get_run(self._run.system_session_id, self._run.run_id)
        if (not force) and meta and meta.state == RunState.concluded:
            # Silent no-op after conclusion to avoid surprises in hooks
            self._run_concluded = True
            return

        # Coerce to list to handle both single items and iterables
        if isinstance(event_or_events, Iterable) and not isinstance(event_or_events, (dict, BaseModel)):
            events = list(cast(Iterable[TelemetryEvent], event_or_events))
        else:
            events = [cast(TelemetryEvent, event_or_events)]

        if events:
            await self._store.append_telemetry_span_event(self._run.system_session_id, self._run.run_id, events)

    # ------------ minimal span plumbing ------------
    def _new_span_id(self) -> str:
        """Generate a new, unique span ID."""
        return f"spn_{uuid.uuid4().hex[:20]}"

    async def _start_common(self, *, span_id: str, name: str, kind: SpanKind):
        """Common logic to start tracking a new span (flat; no parenting)."""
        self._span_t0_ns[span_id] = time.monotonic_ns()
        self._span_name[span_id] = name

    def _end_duration(self, span_id: str) -> int | None:
        """Calculate and return a span's duration in ms, cleaning up its start time."""
        t0 = self._span_t0_ns.pop(span_id, None)
        return ((time.monotonic_ns() - t0) // 1_000_000) if t0 else None

    # flat model: no close/stack bookkeeping
    def _close_span(self, span_id: str) -> None:
        return

    def _to_bytes_for_hash(self, data: Any, mime: str) -> bytes:
        """Serialize data to bytes for hashing, with a canonical JSON fallback."""
        if data is None:
            return b""
        if isinstance(data, (bytes, bytearray)):
            return bytes(data)
        if isinstance(data, str):
            return data.encode("utf-8")
        # default: JSON canonical-ish form
        return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def _mk_part(self, data: Any, mime: str) -> PayloadPart:
        """Create a PayloadPart from data, calculating size and hash."""
        buf = self._to_bytes_for_hash(data, mime)
        return PayloadPart(
            mime_type=mime,
            size_bytes=len(buf),
            sha256=hashlib.sha256(buf).hexdigest(),
            data=data,
        )

    async def _capture_payload(
        self,
        *,
        span_id: str,
        span_kind: SpanKind,
        phase: Literal["start", "end", "point"],
        payload: Any | None,
        mime: str = "application/json",
        redacted: bool = False,
    ) -> str:
        """
        Create/merge payload envelope for this span and persist it.
        Returns the payload id for this span (currently equal to span_id).
        """
        parts: dict[str, PayloadPart] = {}
        if payload is not None:
            parts[phase] = self._mk_part(payload, mime)

        envelope = SpanPayloadEnvelope(
            run_id=self._run.run_id,
            span_id=span_id,
            span_kind=span_kind,  # type: ignore[arg-type]
            capture=parts,
            redacted=redacted,
        )
        await self._store.upsert_span_payload(self._run.system_session_id, self._run.run_id, envelope)
        payload_id = span_id
        # push payload-updated system event if wired
        if self._emit_system_event:
            try:
                span_scope = types.SpanScope(
                    system_session_id=self._run.system_session_id,
                    run_id=self._run.run_id,
                    span_id=span_id,
                )
                await self._emit_system_event(
                    topic="telemetry/payload",
                    change="refetch",
                    span=span_scope,
                )
            except Exception:
                # best-effort only; never break telemetry on fan-out issues
                pass
        return payload_id

    # ------------ RUN (flat) ------------
    async def run_start(
        self,
        *,
        name: str,
        thread_id: str | None,
        settings: RunSettings,
        workflow_name: str,
        run_input: dict[str, Any],
        payload_mime: str = "application/json",
    ) -> str:
        """Start a run span, persist the run start payload on the span."""
        sid = self._new_span_id()
        await self._start_common(span_id=sid, name=name, kind="run")

        # build + capture payload under the span
        start_payload = RunInputPayload(workflow_name=workflow_name, run_input=run_input)
        payload_id = await self._capture_payload(
            span_id=sid,
            span_kind="run",
            phase="start",
            payload=self._dump(start_payload),
            mime=payload_mime,
        )

        # convenience fields (size/keys) for dashboards/queries
        keys = list(run_input.keys())
        size_bytes = len(self._to_bytes_for_hash(self._dump(start_payload), payload_mime))

        meta = {"payload_id": payload_id}

        await self.record(
            RunSpanStart(
                span_id=sid,
                name=name,
                thread_id=thread_id,
                input_keys=keys,
                input_size_bytes=size_bytes,
                settings=settings,
                meta=meta,
            )
        )
        return sid

    async def run_end(
        self,
        *,
        span_id: str,
        status: Literal["ok", "error"] | None = None,
        outcome: RunOutcome | None = None,
        workflow_name: str,
        output: dict[str, Any] | None = None,
        error: str | None = None,
        payload_mime: str = "application/json",
    ) -> None:
        """End a run span, persist the run end payload on the span."""
        duration = self._end_duration(span_id)
        self._close_span(span_id)
        name = self._span_name.pop(span_id, "run")

        meta = {}
        output_keys: list[str] | None = None
        output_size_bytes: int | None = None

        end_payload = RunOutputPayload(
            workflow_name=workflow_name,
            outcome=outcome or RunOutcome.success,
            output=output,
            error=error,
        )
        payload_id = await self._capture_payload(
            span_id=span_id,
            span_kind="run",
            phase="end",
            payload=self._dump(end_payload),
            mime=payload_mime,
        )
        meta = {"payload_id": payload_id}
        if isinstance(output, dict):
            output_keys = list(output.keys())
        output_size_bytes = len(self._to_bytes_for_hash(self._dump(end_payload), payload_mime))

        await self.record(
            RunSpanEnd(
                span_id=span_id,
                name=name,
                duration_ms=duration,
                status=status,
                outcome=outcome,
                output_keys=output_keys,
                output_size_bytes=output_size_bytes,
                meta=meta,
            ),
            force=True,
        )
        # mark concluded locally to skip future store checks
        self._run_concluded = True

    # ------------ AGENT (flat) ------------
    async def agent_start(
        self,
        *,
        agent: str,
        model: str | None = None,
        agent_id: str | None = None,
        system_prompt: str | None = None,
        model_settings: dict[str, Any] | None = None,
        output_type: Any | None = None,
        tools: list[dict[str, Any]] | None = None,
        mcp_servers: list[dict[str, Any]] | None = None,
        mcp_config: dict[str, Any] | None = None,
        input_guardrails: list[dict[str, Any]] | None = None,
        output_guardrails: list[dict[str, Any]] | None = None,
        payload_mime: str = "application/json",
    ) -> str:
        """Start an agent span; capture agent configuration as typed payload."""
        sid = self._new_span_id()
        # flat: no stacks
        await self._start_common(span_id=sid, name=agent, kind="agent")

        meta = {}
        start_payload = AgentStartPayload(
            agent=agent,
            system_prompt=system_prompt,
            model_settings=model_settings,
            output_type=output_type,
            tools=tools,
            mcp_servers=mcp_servers,
            mcp_config=mcp_config,
            input_guardrails=input_guardrails,
            output_guardrails=output_guardrails,
        )
        payload_id = await self._capture_payload(
            span_id=sid,
            span_kind="agent",
            phase="start",
            payload=self._dump(start_payload),
            mime=payload_mime,
        )
        meta = {"payload_id": payload_id}

        await self.record(
            AgentSpanStart(
                span_id=sid,
                name=agent,
                agent=agent,
                agent_id=agent_id,
                model=model,
                meta=meta,
            )
        )
        return sid

    async def agent_end(
        self,
        *,
        span_id: str,
        final_output: Any | None = None,
        error: str | None = None,
        # flat totals (optional)
        usage_total_input_tokens: int | None = None,
        usage_total_output_tokens: int | None = None,
        usage_total_total_tokens: int | None = None,
        usage_total_cached_tokens: int | None = None,
        usage_total_cache_read_tokens: int | None = None,
        usage_total_cache_write_tokens: int | None = None,
        usage_total_reasoning_tokens: int | None = None,
        usage_total_input_characters: int | None = None,
        usage_total_output_characters: int | None = None,
        usage_total_requests: int | None = None,
        agent_id: str | None = None,
        payload_mime: str = "application/json",
    ) -> None:
        """End an agent span; capture final output and infer status from payload.error."""
        duration = self._end_duration(span_id)
        self._close_span(span_id)
        name = self._span_name.pop(span_id, "agent")
        # flat: agent_id is whatever the caller tagged (optional)
        inferred_agent_id = agent_id

        # build totals if provided
        total_uc = None
        if any(
            v is not None
            for v in [
                usage_total_input_tokens,
                usage_total_output_tokens,
                usage_total_total_tokens,
                usage_total_cached_tokens,
                usage_total_cache_read_tokens,
                usage_total_cache_write_tokens,
                usage_total_reasoning_tokens,
                usage_total_input_characters,
                usage_total_output_characters,
                usage_total_requests,
            ]
        ):
            total_uc = build_counters_from_params(
                input_tokens=usage_total_input_tokens,
                output_tokens=usage_total_output_tokens,
                total_tokens=usage_total_total_tokens,
                cached_tokens=usage_total_cached_tokens,
                cache_read_tokens=usage_total_cache_read_tokens,
                cache_write_tokens=usage_total_cache_write_tokens,
                reasoning_tokens=usage_total_reasoning_tokens,
                input_characters=usage_total_input_characters,
                output_characters=usage_total_output_characters,
                requests=usage_total_requests,
            )

        end_payload = AgentEndPayload(
            agent=name,
            final_output=final_output,
            error=error,
            usage_total=total_uc,
            usage_llms=None,
        )

        # If caller didn’t provide usage, aggregate from store using flat rules (tag+window)
        if (end_payload.usage_total is None) or (not end_payload.usage_llms):
            now_ts = time.time()
            # build a timezone-aware datetime for diagnostics/window end
            from datetime import datetime, timezone

            agent_end_ts = datetime.fromtimestamp(now_ts, tz=timezone.utc)
            total, llms, diag = await aggregate_agent_usage(
                self._store,
                self._run.system_session_id,
                self._run.run_id,
                agent_span_id=span_id,
                agent_id=inferred_agent_id,
                agent_end_ts=agent_end_ts,
            )
            if end_payload.usage_total is None:
                end_payload.usage_total = total
            if not end_payload.usage_llms:
                end_payload.usage_llms = llms
            # Emit diagnostics as flat annotations (warning/info)
            await self._emit_usage_diagnostics_annotations(name, inferred_agent_id, diag)

        payload_id = await self._capture_payload(
            span_id=span_id,
            span_kind="agent",
            phase="end",
            payload=self._dump(end_payload),
            mime=payload_mime,
        )

        status: Literal["ok", "error"] = "error" if end_payload.error else "ok"

        await self.record(
            AgentSpanEnd(
                span_id=span_id,
                name=name,
                agent=name,
                agent_id=inferred_agent_id,
                duration_ms=duration,
                status=status,
                meta={"payload_id": payload_id},
            )
        )

    # ------------ LLM ------------
    async def llm_start(
        self,
        *,
        agent: str,
        model: str | None,
        agent_id: str | None = None,
        model_settings: dict[str, Any] | None = None,
        system_prompt: str | None = None,
        input_items: list[Any] | None = None,
        payload_mime: str = "application/json",
    ) -> str:
        """Start an LLM span. Capture system prompt + input items as payload."""
        sid = self._new_span_id()
        name_for_span = f"llm:{model}" if model else "llm:unknown"
        await self._start_common(span_id=sid, name=name_for_span, kind="llm")

        start_payload = LlmStartPayload(
            agent=agent,
            model=model,
            model_settings=model_settings,
            system_prompt=system_prompt,
            input_items=list(input_items or []),
        )
        payload_id = await self._capture_payload(
            span_id=sid,
            span_kind="llm",
            phase="start",
            payload=self._dump(start_payload),
            mime=payload_mime,
        )

        await self.record(
            LlmSpanStart(
                span_id=sid,
                name=name_for_span,
                agent=agent,
                agent_id=agent_id,
                model=model,
                meta={"payload_id": payload_id},
            )
        )
        return sid

    async def llm_end(
        self,
        *,
        span_id: str,
        response: Any | None = None,
        error: str | None = None,
        # flattened usage fields (optional)
        provider: str | None = None,
        model: str | None = None,
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
        cache_hit: bool | None = None,
        details: dict | None = None,
        agent: str | None = None,
        agent_id: str | None = None,
        payload_mime: str = "application/json",
    ) -> None:
        """End an LLM span. Capture response + usage + error in payload, infer status."""
        duration = self._end_duration(span_id)
        self._close_span(span_id)
        span_name = self._span_name.pop(span_id, "llm")
        # flat: take tags as provided
        inferred_agent_id = agent_id
        agent_name = agent or "unknown"

        # build usage from flat params
        counters = build_counters_from_params(
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
        )
        usage_obj = build_llm_usage_from_params(
            provider=provider, model=model, cache_hit=cache_hit, details=details, counters=counters
        )
        end_payload = LlmEndPayload(agent=agent_name, response=response, error=error, usage=usage_obj)
        end_payload.usage = normalize_llm_usage(end_payload.usage)

        payload_id = await self._capture_payload(
            span_id=span_id,
            span_kind="llm",
            phase="end",
            payload=self._dump(end_payload),
            mime=payload_mime,
        )

        status: Literal["ok", "error"] = "error" if error else "ok"

        # Note: no in-memory accumulation; agent totals are computed from store at agent_end

        await self.record(
            LlmSpanEnd(
                span_id=span_id,
                name=span_name,  # e.g., "llm:gpt-4o"
                agent=agent_name,  # proper agent label
                agent_id=inferred_agent_id,
                model=model,
                duration_ms=duration,
                status=status,
                meta={"payload_id": payload_id},
            )
        )

    # ------------ TOOL ------------
    async def tool_start(
        self,
        *,
        tool: str,
        agent: str | None = None,
        agent_id: str | None = None,
        args: Any = None,
        payload_mime: str = "application/json",
    ) -> str:
        """Start a tool span. Capture tool args in a payload."""
        sid = self._new_span_id()
        await self._start_common(span_id=sid, name=tool, kind="tool")

        start_payload = ToolStartPayload(agent=agent, tool=tool, args=args)
        payload_id = await self._capture_payload(
            span_id=sid,
            span_kind="tool",
            phase="start",
            payload=self._dump(start_payload),
            mime=payload_mime,
        )

        await self.record(
            ToolSpanStart(
                span_id=sid,
                name=tool,
                agent=agent,
                agent_id=agent_id,
                tool=tool,
                meta={"payload_id": payload_id},
            )
        )
        return sid

    async def tool_end(
        self,
        *,
        span_id: str,
        tool_result: Any | None = None,
        error: str | None = None,
        agent: str | None = None,
        agent_id: str | None = None,
        payload_mime: str = "application/json",
    ) -> None:
        """End a tool span. Capture result (or error) in payload, infer status."""
        duration = self._end_duration(span_id)
        self._close_span(span_id)
        tool = self._span_name.pop(span_id, "tool")

        agent_name = agent or "unknown"
        end_payload = ToolEndPayload(agent=agent_name, tool=tool, result=tool_result, error=error)
        payload_id = await self._capture_payload(
            span_id=span_id,
            span_kind="tool",
            phase="end",
            payload=self._dump(end_payload),
            mime=payload_mime,
        )

        status: Literal["ok", "error"] = "error" if error else "ok"
        inferred_agent_id = agent_id

        await self.record(
            ToolSpanEnd(
                span_id=span_id,
                name=tool,
                agent=agent_name,
                agent_id=inferred_agent_id,
                tool=tool,
                duration_ms=duration,
                status=status,
                meta={"payload_id": payload_id},
            )
        )

    # ------------ POINT SPANS ------------
    async def annotation(
        self,
        *,
        message: str,
        level: Literal["debug", "info", "warning", "error"] = "info",
        data: dict | None = None,
        payload_mime: str = "application/json",
        preview_chars: int = 160,
    ) -> str:
        """Record an annotation as a point-span (message/data in encrypted payload)."""
        sid = self._new_span_id()
        name = f"annotation:{level}"
        await self._start_common(span_id=sid, name=name, kind="annotation")
        # payload contains full message/data
        payload = AnnotationPayload(level=level, message=message, data=data)
        payload_id = await self._capture_payload(
            span_id=sid, span_kind="annotation", phase="point", payload=self._dump(payload), mime=payload_mime
        )
        meta = {"payload_id": payload_id}
        # span header carries only a safe, truncated preview
        msg = (message or "").strip()
        if preview_chars and preview_chars > 0 and len(msg) > preview_chars:
            preview = msg[: max(1, preview_chars - 1)] + "…"
        else:
            preview = msg
        await self.record(
            AnnotationSpanPoint(
                span_id=sid,
                name=name,
                level=level,
                message_preview=preview or None,
                meta=meta,
            )
        )
        # close bookkeeping for point span
        self._end_duration(sid)
        self._close_span(sid)
        self._span_name.pop(sid, None)
        return sid

    async def guardrail_input_result(
        self,
        *,
        guardrail_name: str | None,
        tripwire_triggered: bool,
        output_info: Any | None = None,
        agent: str | None = None,
        agent_id: str | None = None,
        input_items: list[Any] | None = None,
        error: str | None = None,
        payload_mime: str = "application/json",
    ) -> str:
        """Record an **input guardrail** result as a point span."""
        sid = self._new_span_id()
        await self._start_common(span_id=sid, name="guardrail:input", kind="guardrail")
        status: Literal["ok", "trip", "error"] = "error" if error else ("trip" if tripwire_triggered else "ok")
        payload = GuardrailResultPayload(
            guardrail_kind="input",
            guardrail_name=guardrail_name,
            status=status,
            tripwire_triggered=tripwire_triggered,
            agent=agent,
            agent_id=agent_id,
            output_info=output_info,
            input_items=input_items,
            error=error,
        )
        payload_id = await self._capture_payload(
            span_id=sid, span_kind="guardrail", phase="point", payload=self._dump(payload), mime=payload_mime
        )
        meta = {"payload_id": payload_id}
        await self.record(
            GuardrailSpanPoint(
                span_id=sid,
                name="guardrail:input",
                guardrail_kind="input",
                status=status,
                guardrail_name=guardrail_name,
                tripwire_triggered=tripwire_triggered,
                agent=agent,
                agent_id=agent_id,
                meta=meta,
            )
        )
        if tripwire_triggered and not error:
            await self.annotation(
                message=f"guardrail trip • kind={payload.guardrail_kind} • name={guardrail_name or '-'}",
                level="warning",
            )
        self._end_duration(sid)
        self._close_span(sid)
        self._span_name.pop(sid, None)
        return sid

    async def guardrail_output_result(
        self,
        *,
        guardrail_name: str | None,
        tripwire_triggered: bool,
        agent_output: Any,
        output_info: Any | None = None,
        agent: str | None = None,
        agent_id: str | None = None,
        error: str | None = None,
        payload_mime: str = "application/json",
    ) -> str:
        """Record an **output guardrail** result as a point span."""
        sid = self._new_span_id()
        await self._start_common(span_id=sid, name="guardrail:output", kind="guardrail")
        status: Literal["ok", "trip", "error"] = "error" if error else ("trip" if tripwire_triggered else "ok")
        payload = GuardrailResultPayload(
            guardrail_kind="output",
            guardrail_name=guardrail_name,
            status=status,
            tripwire_triggered=tripwire_triggered,
            agent=agent,
            agent_id=agent_id,
            output_info=output_info,
            agent_output=agent_output,
            error=error,
        )
        payload_id = await self._capture_payload(
            span_id=sid, span_kind="guardrail", phase="point", payload=self._dump(payload), mime=payload_mime
        )
        meta = {"payload_id": payload_id}
        await self.record(
            GuardrailSpanPoint(
                span_id=sid,
                name="guardrail:output",
                guardrail_kind="output",
                status=status,
                guardrail_name=guardrail_name,
                tripwire_triggered=tripwire_triggered,
                agent=agent,
                agent_id=agent_id,
                meta=meta,
            )
        )
        if tripwire_triggered and not error:
            await self.annotation(
                message=f"guardrail trip • kind={payload.guardrail_kind} • name={guardrail_name or '-'}",
                level="warning",
            )
        self._end_duration(sid)
        self._close_span(sid)
        self._span_name.pop(sid, None)
        return sid

    async def handoff(
        self,
        *,
        from_agent: str,
        to_agent: str,
        # FROM snapshot (payload only)
        from_model: str | None = None,
        from_agent_id: str | None = None,
        from_system_prompt: str | None = None,
        from_model_settings: dict[str, Any] | None = None,
        from_output_type: Any | None = None,
        from_tools: list[dict[str, Any]] | None = None,
        from_mcp_servers: list[dict[str, Any]] | None = None,
        from_mcp_config: dict[str, Any] | None = None,
        from_input_guardrails: list[dict[str, Any]] | None = None,
        from_output_guardrails: list[dict[str, Any]] | None = None,
        # TO snapshot (payload only)
        to_model: str | None = None,
        to_agent_id: str | None = None,
        to_system_prompt: str | None = None,
        to_model_settings: dict[str, Any] | None = None,
        to_output_type: Any | None = None,
        to_tools: list[dict[str, Any]] | None = None,
        to_mcp_servers: list[dict[str, Any]] | None = None,
        to_mcp_config: dict[str, Any] | None = None,
        to_input_guardrails: list[dict[str, Any]] | None = None,
        to_output_guardrails: list[dict[str, Any]] | None = None,
        meta: dict | None = None,
        payload_mime: str = "application/json",
    ) -> str:
        """Record a handoff as a point-span (full snapshots in encrypted payload)."""
        sid = self._new_span_id()
        name = f"handoff:{from_agent}->{to_agent}"
        await self._start_common(span_id=sid, name=name, kind="handoff")
        payload = HandoffPayload(
            from_agent=HandoffAgentSnapshot(
                agent=from_agent,
                agent_id=from_agent_id,
                model=from_model,
                system_prompt=from_system_prompt,
                model_settings=from_model_settings,
                output_type=from_output_type,
                tools=from_tools,
                mcp_servers=from_mcp_servers,
                mcp_config=from_mcp_config,
                input_guardrails=from_input_guardrails,
                output_guardrails=from_output_guardrails,
            ),
            to_agent=HandoffAgentSnapshot(
                agent=to_agent,
                agent_id=to_agent_id,
                model=to_model,
                system_prompt=to_system_prompt,
                model_settings=to_model_settings,
                output_type=to_output_type,
                tools=to_tools,
                mcp_servers=to_mcp_servers,
                mcp_config=to_mcp_config,
                input_guardrails=to_input_guardrails,
                output_guardrails=to_output_guardrails,
            ),
            meta=(meta or {}),
        )
        payload_id = await self._capture_payload(
            span_id=sid, span_kind="handoff", phase="point", payload=self._dump(payload), mime=payload_mime
        )
        meta_refs = {"payload_id": payload_id}
        await self.record(
            HandoffSpanPoint(
                span_id=sid,
                name=name,
                from_agent=from_agent,
                to_agent=to_agent,
                from_agent_id=from_agent_id,
                to_agent_id=to_agent_id,
                from_model=from_model,
                to_model=to_model,
                meta=meta_refs,
            )
        )
        self._end_duration(sid)
        self._close_span(sid)
        self._span_name.pop(sid, None)
        return sid

    # ------------ internal: emit usage diagnostics as annotations (flat) ------------
    async def _emit_usage_diagnostics_annotations(
        self,
        agent_label: str,
        agent_id: str | None,
        diag: AggregationDiagnostics | None,
    ) -> None:
        if not diag:
            return
        prefix = f"usage ({agent_label}"
        if agent_id:
            prefix += f" • agent_id={agent_id}"
        prefix += "): "
        if diag.missing_agent_id:
            await self.annotation(message=prefix + "missing agent_id; counted only in run totals", level="warning")
        if diag.window_missing:
            await self.annotation(
                message=prefix + "could not determine agent time window; per-agent usage disabled", level="warning"
            )
        if diag.ambiguous_agent_id:
            await self.annotation(
                message=prefix + "overlapping agent windows detected for the same agent_id; per-agent usage disabled",
                level="warning",
            )
        if diag.untagged_in_window:
            await self.annotation(
                message=prefix
                + f"{diag.untagged_in_window} LLM events in window lacked agent_id; included only in run totals",
                level="warning",
            )
        # Informational summary
        await self.annotation(
            message=prefix
            + f"included {diag.included_llm_events}/{diag.considered_llm_events} LLM events for per-agent totals",
            level="info",
        )
        for note in diag.notes or []:
            await self.annotation(message=prefix + note, level="debug")
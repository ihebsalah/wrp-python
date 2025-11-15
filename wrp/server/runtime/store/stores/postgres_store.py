# wrp/server/runtime/store/stores/postgres_store.py
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Iterable, List, Optional

from wrp.server.runtime.runs.types import RunMeta, RunOutcome, RunState
from wrp.server.runtime.conversations.types import ChannelItem, ChannelMeta
from wrp.server.runtime.telemetry.events import (
    TelemetryEvent,
    RunSpanStart, RunSpanEnd,
    AgentSpanStart, AgentSpanEnd,
    LlmSpanStart, LlmSpanEnd,
    ToolSpanStart, ToolSpanEnd,
    HandoffSpanPoint, AnnotationSpanPoint, GuardrailSpanPoint,
)
from wrp.server.runtime.telemetry.payloads.types import SpanPayloadEnvelope

from ..base import Store
from ..engines.postgres_engine import PostgresEngine
from ..dao.runs_dao import RunsDAO
from ..dao.conversations_dao import ConversationsDAO
from ..dao.spans_dao import SpansDAO
from ..dao.span_payloads_dao import SpanPayloadsDAO
from ..dao.workflow_settings_dao import WorkflowSettingsDAO
from ..dao.system_sessions_dao import SystemSessionsDAO
from ..dao.provider_settings_dao import ProviderSettingsDAO
from ..dao.agent_settings_dao import AgentSettingsDAO
from ..codecs import serializer, envelope_codec
from ..ops.migrations import apply_initial_schema_postgres

if TYPE_CHECKING:
    from wrp.server.runtime.system_sessions.types import SystemSession


def _iso(dt: datetime) -> str:
    # store as explicit UTC ISO-8601
    return dt.astimezone(timezone.utc).isoformat()

def _parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s)


def _event_from_dict(d: dict) -> TelemetryEvent:
    # reconstruct union instance
    k = d.get("kind")
    if k == "span":
        sk = d.get("span_kind")
        ph = d.get("phase")
        if sk == "run" and ph == "start":
            return RunSpanStart.model_validate(d)
        if sk == "run" and ph == "end":
            return RunSpanEnd.model_validate(d)
        if sk == "agent" and ph == "start":
            return AgentSpanStart.model_validate(d)
        if sk == "agent" and ph == "end":
            return AgentSpanEnd.model_validate(d)
        if sk == "llm" and ph == "start":
            return LlmSpanStart.model_validate(d)
        if sk == "llm" and ph == "end":
            return LlmSpanEnd.model_validate(d)
        if sk == "tool" and ph == "start":
            return ToolSpanStart.model_validate(d)
        if sk == "tool" and ph == "end":
            return ToolSpanEnd.model_validate(d)
        if sk == "handoff" and ph == "point":
            return HandoffSpanPoint.model_validate(d)
        if sk == "annotation" and ph == "point":
            return AnnotationSpanPoint.model_validate(d)
        if sk == "guardrail" and ph == "point":
            return GuardrailSpanPoint.model_validate(d)
    # conservative: try RunSpanStart as a fallback
    return RunSpanStart.model_validate(d)


class PostgresStore(Store):
    """
    Durable store backed by Postgres.
    """

    def __init__(self, dsn: str, *, key: bytes | None = None):
        self.engine = PostgresEngine(dsn)
        self.engine.connect()
        apply_initial_schema_postgres(self.engine)  # idempotent
        self.runs = RunsDAO(self.engine)
        self.conv = ConversationsDAO(self.engine)
        self.span = SpansDAO(self.engine)
        self.payload = SpanPayloadsDAO(self.engine)
        self.wfs = WorkflowSettingsDAO(self.engine)
        self.sessions = SystemSessionsDAO(self.engine)
        self._key = key
        self.providers = ProviderSettingsDAO(self.engine)
        self.agents = AgentSettingsDAO(self.engine)

    # -------- system sessions ----------------------------------------------

    async def ensure_system_session(self, system_session_id: str, name: str | None = None) -> None:
        self.sessions.upsert(system_session_id, name, _iso(datetime.now(timezone.utc)))

    async def get_system_session(self, system_session_id: str) -> "SystemSession | None":
        r = self.sessions.get(system_session_id)
        if not r:
            return None
        from wrp.server.runtime.system_sessions.types import SystemSession
        return SystemSession(system_session_id=r["system_session_id"], name=r["name"], created_at=_parse_iso(r["created_at"]))

    async def list_system_sessions(self) -> list["SystemSession"]:
        from wrp.server.runtime.system_sessions.types import SystemSession
        return [SystemSession(system_session_id=r["system_session_id"], name=r["name"], created_at=_parse_iso(r["created_at"])) for r in self.sessions.list_all()]

    # -------- allocation ---------------------------------------------------

    async def alloc_run_id(self, system_session_id: str, workflow_name: str, thread_id: str | None) -> str:  # noqa: ARG002
        rid = self.runs.alloc_run_id(system_session_id)
        if rid > 999:
            raise ValueError("Run ID capacity reached (001..999). Please rotate or archive old runs.")
        return f"{rid:03d}"

    # -------- lifecycle ----------------------------------------------------

    async def create_run(self, meta: RunMeta) -> None:
        self.runs.insert(
            {
                "system_session_id": meta.system_session_id,
                "run_id": meta.run_id,
                "workflow_name": meta.workflow_name,
                "thread_id": meta.thread_id,
                "created_at": _iso(meta.created_at),
                "state": meta.state.value,
                "updated_at": _iso(meta.created_at),
            }
        )

    async def conclude_run(
        self,
        system_session_id: str,
        run_id: str,
        outcome: RunOutcome,
        *,
        error: str | None = None,
        run_output: dict | None = None,
    ) -> None:
        updated_at = _iso(datetime.now(timezone.utc))
        blob = envelope_codec.pack(run_output, key=self._key) if run_output is not None else None
        self.runs.update_conclude(system_session_id, run_id, outcome.value, error, blob, updated_at)

    # -------- lookups ------------------------------------------------------

    async def get_run(self, system_session_id: str, run_id: str) -> RunMeta | None:
        row = self.runs.get(system_session_id, run_id)
        if not row:
            return None
        meta = RunMeta(
            system_session_id=row["system_session_id"],
            run_id=row["run_id"],
            workflow_name=row["workflow_name"],
            created_at=_parse_iso(row["created_at"]),
            state=RunState(row["state"]),
            thread_id=row["thread_id"],
            outcome=(RunOutcome(row["outcome"]) if row["outcome"] else None),
            error=row["error_text"],
            run_output=(envelope_codec.unpack(row["run_output_blob"], key=self._key) if row["run_output_blob"] else None),
        )
        return meta

    async def runs_in_thread(self, system_session_id: str, workflow_name: str, thread_id: str) -> list[RunMeta]:
        rows = self.runs.list_by_thread(system_session_id, workflow_name, thread_id)
        out: list[RunMeta] = []
        for r in rows:
            out.append(
                RunMeta(
                    system_session_id=r["system_session_id"],
                    run_id=r["run_id"],
                    workflow_name=r["workflow_name"],
                    created_at=_parse_iso(r["created_at"]),
                    state=RunState(r["state"]),
                    thread_id=r["thread_id"],
                    outcome=(RunOutcome(r["outcome"]) if r["outcome"] else None),
                    error=r["error_text"],
                    run_output=(envelope_codec.unpack(r["run_output_blob"], key=self._key) if r["run_output_blob"] else None),
                )
            )
        return out

    async def list_runs(
        self,
        system_session_id: str,
        *,
        workflow_name: str | None = None,
        thread_id: str | None = None,
        state: RunState | None = None,
        outcome: RunOutcome | None = None,
    ) -> list[RunMeta]:
        rows = self.runs.list_runs(
            system_session_id,
            workflow_name=workflow_name,
            thread_id=thread_id,
            state=(state.value if state else None),
            outcome=(outcome.value if outcome else None),
        )
        out: list[RunMeta] = []
        for r in rows:
            out.append(
                RunMeta(
                    system_session_id=r["system_session_id"],
                    run_id=r["run_id"],
                    workflow_name=r["workflow_name"],
                    created_at=_parse_iso(r["created_at"]),
                    state=RunState(r["state"]),
                    thread_id=r["thread_id"],
                    outcome=(RunOutcome(r["outcome"]) if r["outcome"] else None),
                    error=r["error_text"],
                    run_output=(
                        envelope_codec.unpack(r["run_output_blob"], key=self._key) if r["run_output_blob"] else None
                    ),
                )
            )
        return out

    # -------- conversation -------------------------------------------------

    async def append_conversation_channel_item(self, system_session_id: str, run_id: str, items: Iterable[ChannelItem]) -> None:
        items = list(items)
        if not items:
            return

        start_idx = self._next_conv_idx(system_session_id, run_id)
        rows = []
        idx = start_idx
        for it in items:
            payload_blob = envelope_codec.pack(it.payload, key=self._key)
            rows.append({"idx": idx, "ts": _iso(it.ts), "channel": it.channel, "payload_blob": payload_blob})
            idx += 1

        self.conv.append_many(system_session_id, run_id, rows)

        # upsert channel meta counters
        bump = Counter(i.channel for i in items)
        for ch, n in bump.items():
            last_ts = max(i.ts for i in items if i.channel == ch)
            self.conv.upsert_channel_meta(
                system_session_id, run_id, ch,
                add_count=n,
                last_ts=_iso(last_ts),
                name=None,
                description=None,
            )

    # ---- channel meta & items --------------------------------------------
    async def list_channel_meta(self, system_session_id: str, run_id: str) -> List[ChannelMeta]:
        rows = self.conv.list_channels_meta(system_session_id, run_id)
        out: List[ChannelMeta] = []
        for r in rows:
            out.append(ChannelMeta(
                id=r["channel"],
                name=r["name"],
                description=r["description"],
                itemsCount=r["items_count"],
                lastItemTs=(_parse_iso(r["last_ts"]) if r["last_ts"] else None),
            ))
        return out

    async def load_channel_items(self, system_session_id: str, run_id: str, *, channel: str, limit: int | None = None) -> List[ChannelItem]:
        rows = self.conv.load_tail(system_session_id, run_id, limit=(limit or 10**9), channels={channel})
        out: list[ChannelItem] = []
        for r in rows:
            payload = envelope_codec.unpack(r["payload_blob"], key=self._key)
            out.append(ChannelItem(payload=payload, channel=r["channel"], ts=_parse_iso(r["ts"])))
        return out

    # -------- telemetry ----------------------------------------------------

    async def append_telemetry_span_event(self, system_session_id: str, run_id: str, events: Iterable[TelemetryEvent]) -> None:
        rows = []
        for ev in events:
            d = ev.model_dump(exclude_none=True, mode="json")
            ts_val = d.get("ts")
            # Ensure we have a valid datetime object, defaulting to now_utc if not present.
            ts_dt = ts_val if isinstance(ts_val, datetime) else datetime.now(timezone.utc)
            # Store telemetry events as plaintext JSON (queryable).
            # Sensitive content lives only in span payloads (encrypted separately).
            blob = serializer.dumps(d)
            rows.append(
                {
                    "ts": _iso(ts_dt),
                    "kind": d.get("kind"),
                    "payload_blob": blob,
                }
            )
        self.span.append_many(system_session_id, run_id, rows)

    async def load_telemetry(
        self,
        system_session_id: str,
        run_id: str,
        *,
        kinds: Optional[set[str]] = None,
        limit: Optional[int] = None,
    ) -> List[TelemetryEvent]:
        rows = self.span.load(system_session_id, run_id, kinds, limit)
        out: list[TelemetryEvent] = []
        for r in rows:
            d = serializer.loads(r["payload_blob"])
            out.append(_event_from_dict(d))
        return out

    # -------- span payloads -----------------------------------------------

    async def upsert_span_payload(self, system_session_id: str, run_id: str, payload: SpanPayloadEnvelope) -> None:
        # Merge capture parts if an existing record is present
        existing = await self.get_span_payload(system_session_id, run_id, payload.span_id)
        merged = payload
        if existing:
            merged = existing.model_copy(deep=True)
            if "start" in payload.capture:
                merged.capture["start"] = payload.capture["start"]
            if "end" in payload.capture:
                merged.capture["end"] = payload.capture["end"]
            if "point" in payload.capture:
                merged.capture["point"] = payload.capture["point"]
            merged.redacted = payload.redacted or existing.redacted
            merged.updated_at = payload.updated_at

        env_blob = envelope_codec.pack(merged.model_dump(mode="json"), key=self._key)
        self.payload.upsert(system_session_id, run_id, payload.span_id, env_blob, _iso(merged.updated_at))

    async def get_span_payload(self, system_session_id: str, run_id: str, span_id: str) -> SpanPayloadEnvelope | None:
        row = self.payload.get(system_session_id, run_id, span_id)
        if not row:
            return None
        d = envelope_codec.unpack(row["envelope_blob"], key=self._key)
        return SpanPayloadEnvelope.model_validate(d)

    # -------- workflow settings -------------------------------------------
    async def upsert_workflow_settings(self, workflow_name: str, values: dict, *, overridden: bool) -> None:
        updated_at = _iso(datetime.now(timezone.utc))
        values_json = serializer.dumps(values).decode("utf-8")
        self.wfs.upsert(workflow_name, values_json, bool(overridden), updated_at)

    async def get_workflow_settings(self, workflow_name: str) -> tuple[dict, bool] | None:
        row = self.wfs.get(workflow_name)
        if not row:
            return None
        values = serializer.loads(row["values_json"].encode("utf-8"))
        overridden = bool(row["overridden"])
        return values, overridden

    async def list_workflow_settings(self) -> dict[str, tuple[dict, bool]]:
        out: dict[str, tuple[dict, bool]] = {}
        for r in self.wfs.list_all():
            out[r["workflow_name"]] = (serializer.loads(r["values_json"].encode("utf-8")), bool(r["overridden"]))
        return out

    # -------- provider settings -------------------------------------------
    async def upsert_provider_settings(self, provider_name: str, values: dict, *, overridden: bool) -> None:
        updated_at = _iso(datetime.now(timezone.utc))
        blob = envelope_codec.pack(values, key=self._key)
        self.providers.upsert(provider_name, blob, bool(overridden), updated_at)

    async def get_provider_settings(self, provider_name: str) -> tuple[dict, bool] | None:
        row = self.providers.get(provider_name)
        if not row:
            return None
        values = envelope_codec.unpack(row["values_blob"], key=self._key)
        overridden = bool(row["overridden"])
        return values, overridden

    async def list_provider_settings(self) -> dict[str, tuple[dict, bool]]:
        out: dict[str, tuple[dict, bool]] = {}
        for r in self.providers.list_all():
            values = envelope_codec.unpack(r["values_blob"], key=self._key)
            out[r["provider_name"]] = (values, bool(r["overridden"]))
        return out

    # -------- agent settings ----------------------------------------------
    async def upsert_agent_settings(self, agent_name: str, values: dict, *, overridden: bool) -> None:
        updated_at = _iso(datetime.now(timezone.utc))
        values_json = serializer.dumps(values).decode("utf-8")
        self.agents.upsert(agent_name, values_json, bool(overridden), updated_at)

    async def get_agent_settings(self, agent_name: str) -> tuple[dict, bool] | None:
        row = self.agents.get(agent_name)
        if not row:
            return None
        values = serializer.loads(row["values_json"].encode("utf-8"))
        overridden = bool(row["overridden"])
        return values, overridden

    async def list_agent_settings(self) -> dict[str, tuple[dict, bool]]:
        out: dict[str, tuple[dict, bool]] = {}
        for r in self.agents.list_all():
            out[r["agent_name"]] = (serializer.loads(r["values_json"].encode("utf-8")), bool(r["overridden"]))
        return out

    # -------- internals ----------------------------------------------------

    def _next_conv_idx(self, system_session_id: str, run_id: str) -> int:
        row = self.engine.query_one(
            "SELECT COALESCE(MAX(idx), -1) AS last_idx FROM conversation_items WHERE system_session_id=%s AND run_id=%s;",
            (system_session_id, run_id,),
        )
        last = int(row["last_idx"]) if row else -1
        return last + 1

    # -------- channel bootstrap -------------------------------------------
    async def ensure_channel_meta(
        self,
        system_session_id: str,
        run_id: str,
        channel: str,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        # Insert-or-noop a meta row without touching counters/last_ts.
        self.conv.upsert_channel_meta(
            system_session_id,
            run_id,
            channel,
            add_count=0,
            last_ts=None,
            name=name,
            description=description,
        )
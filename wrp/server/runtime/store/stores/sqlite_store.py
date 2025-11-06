# wrp/server/runtime/store/stores/sqlite_store.py
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any, Iterable, List, Optional, Set, Tuple

from wrp.server.runtime.runs.types import RunMeta, RunOutcome, RunState
from wrp.server.runtime.conversations.types import ConversationItem
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
from ..engines.sqlite_engine import SqliteEngine
from ..dao.runs_dao import RunsDAO
from ..dao.conversations_dao import ConversationsDAO
from ..dao.telemetry_dao import TelemetryDAO
from ..dao.span_payloads_dao import SpanPayloadsDAO
from ..dao.workflow_settings_dao import WorkflowSettingsDAO
from ..codecs import serializer, envelope_codec
from ..ops.migrations import apply_initial_schema_sqlite


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


class SqliteStore(Store):
    """
    Durable store backed by SQLite.
    - Encryption at rest: yes (best-effort), using envelope_codec + local key.
    - All methods are async per ABC, but DB access is sync (short ops).
      If you need non-blocking, wrap this store with anyio.to_thread.
    """

    def __init__(self, path: str, *, key: bytes | None = None):
        self.engine = SqliteEngine(path)
        self.engine.connect()
        apply_initial_schema_sqlite(self.engine)  # idempotent
        self.runs = RunsDAO(self.engine)
        self.conv = ConversationsDAO(self.engine)
        self.tlm = TelemetryDAO(self.engine)
        self.span = SpanPayloadsDAO(self.engine)
        self.wfs = WorkflowSettingsDAO(self.engine)
        self._key = key

    # -------- allocation ---------------------------------------------------

    def supports_message_counts(self) -> bool:
        return True

    async def alloc_run_id(self, workflow_name: str, thread_id: str | None) -> str:  # noqa: ARG002
        rid = self.runs.alloc_run_id()
        return f"{rid:03d}"

    # -------- lifecycle ----------------------------------------------------

    async def create_run(self, meta: RunMeta) -> None:
        self.runs.insert(
            {
                "run_id": meta.run_id,
                "workflow_name": meta.workflow_name,
                "thread_id": meta.thread_id,
                "created_at": _iso(meta.created_at),
                "state": meta.state.value,
                "message_count": meta.message_count,
                "channel_counts_json": serializer.dumps(meta.channel_counts).decode("utf-8"),
                "updated_at": _iso(meta.created_at),
            }
        )

    async def conclude_run(
        self,
        run_id: str,
        outcome: RunOutcome,
        *,
        error: str | None = None,
        run_output: dict | None = None,
    ) -> None:
        updated_at = _iso(datetime.now(timezone.utc))
        blob = envelope_codec.pack(run_output, key=self._key) if run_output is not None else None
        self.runs.update_conclude(run_id, outcome.value, error, blob, updated_at)

    # -------- lookups ------------------------------------------------------

    async def get_run(self, run_id: str) -> RunMeta | None:
        row = self.runs.get(run_id)
        if not row:
            return None
        meta = RunMeta(
            run_id=row["run_id"],
            workflow_name=row["workflow_name"],
            created_at=_parse_iso(row["created_at"]),
            state=RunState(row["state"]),
            thread_id=row["thread_id"],
            message_count=row["message_count"],
            channel_counts=serializer.loads(row["channel_counts_json"].encode("utf-8")),
            outcome=(RunOutcome(row["outcome"]) if row["outcome"] else None),
            error=row["error_text"],
            run_output=(envelope_codec.unpack(row["run_output_blob"], key=self._key) if row["run_output_blob"] else None),
        )
        return meta

    async def runs_in_thread(self, workflow_name: str, thread_id: str) -> list[RunMeta]:
        rows = self.runs.list_by_thread(workflow_name, thread_id)
        out: list[RunMeta] = []
        for r in rows:
            out.append(
                RunMeta(
                    run_id=r["run_id"],
                    workflow_name=r["workflow_name"],
                    created_at=_parse_iso(r["created_at"]),
                    state=RunState(r["state"]),
                    thread_id=r["thread_id"],
                    message_count=r["message_count"],
                    channel_counts=serializer.loads(r["channel_counts_json"].encode("utf-8")),
                    outcome=(RunOutcome(r["outcome"]) if r["outcome"] else None),
                    error=r["error_text"],
                    run_output=(envelope_codec.unpack(r["run_output_blob"], key=self._key) if r["run_output_blob"] else None),
                )
            )
        return out

    # -------- conversation -------------------------------------------------

    async def append_conversation(self, run_id: str, items: Iterable[ConversationItem]) -> None:
        items = list(items)
        if not items:
            return
        now_iso = _iso(datetime.now(timezone.utc))

        # fetch current counts for channel deltas
        meta = await self.get_run(run_id)
        channel_counts = dict(meta.channel_counts) if meta else {}

        start_idx = self._next_conv_idx(run_id)
        rows = []
        idx = start_idx
        for it in items:
            payload_blob = envelope_codec.pack(it.payload, key=self._key)
            rows.append(
                {"idx": idx, "ts": _iso(it.ts), "channel": it.channel, "payload_blob": payload_blob}
            )
            idx += 1

        self.conv.append_many(run_id, rows)

        # update counters
        bump = Counter(i.channel for i in items)
        for ch, n in bump.items():
            channel_counts[ch] = channel_counts.get(ch, 0) + n

        self.runs.bump_counts(
            run_id=run_id,
            message_delta=len(items),
            channel_counts_json=serializer.dumps(channel_counts).decode("utf-8"),
            updated_at=now_iso,
        )

    async def load_conversation(self, run_id: str) -> list[ConversationItem]:
        rows = self.conv.load_all(run_id)
        out: list[ConversationItem] = []
        for r in rows:
            payload = envelope_codec.unpack(r["payload_blob"], key=self._key)
            out.append(ConversationItem(payload=payload, channel=r["channel"], ts=_parse_iso(r["ts"])))
        return out

    async def load_conversation_tail(
        self,
        run_id: str,
        *,
        limit: int,
        channels: Optional[Set[str]] = None,
    ) -> list[ConversationItem]:
        rows = self.conv.load_tail(run_id, limit=limit, channels=channels or None)
        out: list[ConversationItem] = []
        for r in rows:
            payload = envelope_codec.unpack(r["payload_blob"], key=self._key)
            out.append(ConversationItem(payload=payload, channel=r["channel"], ts=_parse_iso(r["ts"])))
        return out

    # -------- telemetry ----------------------------------------------------

    async def append_telemetry(self, run_id: str, events: Iterable[TelemetryEvent]) -> None:
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
        self.tlm.append_many(run_id, rows)

    async def load_telemetry(
        self,
        run_id: str,
        *,
        kinds: Optional[Set[str]] = None,
        limit: Optional[int] = None,
    ) -> List[TelemetryEvent]:
        rows = self.tlm.load(run_id, kinds, limit)
        out: list[TelemetryEvent] = []
        for r in rows:
            d = serializer.loads(r["payload_blob"])
            out.append(_event_from_dict(d))
        return out

    # -------- span payloads -----------------------------------------------

    async def upsert_span_payload(self, run_id: str, payload: SpanPayloadEnvelope) -> None:
        # Merge capture parts if an existing record is present
        existing = await self.get_span_payload(run_id, payload.span_id)
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
        self.span.upsert(run_id, payload.span_id, env_blob, _iso(merged.updated_at))

    async def get_span_payload(self, run_id: str, span_id: str) -> SpanPayloadEnvelope | None:
        row = self.span.get(run_id, span_id)
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

    # -------- internals ----------------------------------------------------

    def _next_conv_idx(self, run_id: str) -> int:
        # For simplicity, compute max idx+1
        row = self.engine.query_one(
            "SELECT COALESCE(MAX(idx), -1) AS last_idx FROM conversation_items WHERE run_id=%s;",
            (run_id,),
        )
        last = int(row["last_idx"]) if row else -1
        return last + 1
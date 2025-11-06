# wrp/server/runtime/store/stores/memory_store.py
from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, List, Optional, Set, Tuple

from wrp.server.runtime.runs.types import RunMeta, RunOutcome, RunState
from wrp.server.runtime.conversations.types import ConversationItem
from wrp.server.runtime.telemetry.events import TelemetryEvent
from wrp.server.runtime.telemetry.payloads.types import SpanPayloadEnvelope

from ..base import Store


class InMemoryStore(Store):
    """Non-durable v0 store."""

    def __init__(self):
        self._runs: dict[str, RunMeta] = {}
        self._conv: dict[str, list[ConversationItem]] = {}
        # global counter for human-friendly run ids (001..999)
        self._global_counter: int = 0
        # telemetry bucket
        self._telemetry: dict[str, list[TelemetryEvent]] = {}
        self._span_payloads: dict[Tuple[str, str], SpanPayloadEnvelope] = {}
        # optional quick lookup for concluded outputs (used by some callers)
        self._run_outputs: dict[str, dict] = {}
        # workflow settings
        self._wf_settings: dict[str, tuple[dict, bool]] = {}

    def supports_message_counts(self) -> bool:
        return True

    async def alloc_run_id(self, workflow_name: str, thread_id: str | None) -> str:  # noqa: ARG002
        if self._global_counter >= 999:
            raise ValueError(
                "Run ID capacity reached (001..999). Please rotate or archive old runs."
            )
        self._global_counter += 1
        return f"{self._global_counter:03d}"

    async def create_run(self, meta: RunMeta) -> None:
        self._runs[meta.run_id] = meta
        self._conv[meta.run_id] = []

    async def conclude_run(
        self,
        run_id: str,
        outcome: RunOutcome,
        *,
        error: str | None = None,
        run_output: dict | None = None,
    ) -> None:
        meta = self._runs[run_id]
        meta.state = RunState.concluded
        meta.outcome = outcome
        meta.error = error
        meta.run_output = run_output
        if run_output is not None:
            self._run_outputs[run_id] = run_output
        self._runs[run_id] = meta

    async def get_run(self, run_id: str) -> RunMeta | None:
        return self._runs.get(run_id)

    async def runs_in_thread(self, workflow_name: str, thread_id: str) -> list[RunMeta]:
        return sorted(
            (m for m in self._runs.values() if m.workflow_name == workflow_name and m.thread_id == thread_id),
            key=lambda m: m.created_at,
        )

    async def append_conversation(self, run_id: str, items: Iterable[ConversationItem]) -> None:
        items = list(items)
        self._conv.setdefault(run_id, [])
        self._conv[run_id].extend(items)
        # fast counter update
        meta = self._runs[run_id]
        meta.message_count += len(items)
        if items:
            bump = Counter(i.channel for i in items)
            for ch, n in bump.items():
                meta.channel_counts[ch] = meta.channel_counts.get(ch, 0) + n
        self._runs[run_id] = meta
        # NOTE: For durable stores, perform these counter updates in the same
        # transaction as the append to keep counts authoritative.

    async def load_conversation(self, run_id: str) -> list[ConversationItem]:
        return list(self._conv.get(run_id, []))

    async def load_conversation_tail(
        self,
        run_id: str,
        *,
        limit: int,
        channels: set[str] | None = None,
    ) -> list[ConversationItem]:
        items = self._conv.get(run_id, [])
        if channels:
            # scan from the end until we have `limit`
            acc = []
            for i in range(len(items) - 1, -1, -1):
                if items[i].channel in channels:
                    acc.append(items[i])
                    if len(acc) >= limit:
                        break
            acc.reverse()
            return acc
        # fast slice
        return items[-limit:]

    # ---- telemetry (new) ----
    async def append_telemetry(self, run_id: str, events: Iterable[TelemetryEvent]) -> None:
        bucket = self._telemetry.setdefault(run_id, [])
        bucket.extend(list(events))

    async def load_telemetry(
        self,
        run_id: str,
        *,
        kinds: Optional[Set[str]] = None,
        limit: Optional[int] = None,
    ) -> List[TelemetryEvent]:
        items = list(self._telemetry.get(run_id, []))
        if kinds:
            items = [e for e in items if getattr(e, "kind", None) in kinds]
        if limit is not None and limit >= 0:
            # ensure the last N are returned oldest→newest, matching SQL stores
            return items[-limit:]
        return items

    # ---- span payloads (start/end) --------------------------------------
    async def upsert_span_payload(self, run_id: str, payload: SpanPayloadEnvelope) -> None:
        key = (run_id, payload.span_id)
        existing = self._span_payloads.get(key)
        if existing:
            # merge capture parts & metadata
            if "start" in payload.capture:
                existing.capture["start"] = payload.capture["start"]
            if "end" in payload.capture:
                existing.capture["end"] = payload.capture["end"]
            if "point" in payload.capture:
                existing.capture["point"] = payload.capture["point"]
            existing.redacted = payload.redacted or existing.redacted
            existing.updated_at = payload.updated_at
            self._span_payloads[key] = existing
        else:
            self._span_payloads[key] = payload

    async def get_span_payload(self, run_id: str, span_id: str) -> SpanPayloadEnvelope | None:
        return self._span_payloads.get((run_id, span_id))

    # ---- workflow settings (new) -----------------------------------------
    async def upsert_workflow_settings(self, workflow_name: str, values: dict, *, overridden: bool) -> None:
        self._wf_settings[workflow_name] = (dict(values), bool(overridden))

    async def get_workflow_settings(self, workflow_name: str) -> tuple[dict, bool] | None:
        return self._wf_settings.get(workflow_name)

    async def list_workflow_settings(self) -> dict[str, tuple[dict, bool]]:
        return dict(self._wf_settings)
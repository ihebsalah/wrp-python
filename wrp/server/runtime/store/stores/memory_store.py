# wrp/server/runtime/store/stores/memory_store.py
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Iterable, List

from wrp.server.runtime.conversations.types import (
    ChannelMeta,
    ChannelItem,
)
from wrp.server.runtime.runs.types import RunMeta, RunOutcome, RunState
from wrp.server.runtime.telemetry.events import TelemetryEvent
from wrp.server.runtime.telemetry.payloads.types import SpanPayloadEnvelope

from ..base import Store


class InMemoryStore(Store):
    """Non-durable v0 store."""

    def __init__(self):
        self._runs: dict[tuple[str, str], RunMeta] = {}  # (system_session_id, run_id) -> meta
        self._conv: dict[tuple[str, str], list[ChannelItem]] = {}  # (system_session_id, run_id)
        # per-session counter for human-friendly run ids (001..999)
        self._counter: dict[str, int] = {}  # system_session_id -> last int
        # per-run channel meta
        self._conv_meta: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
        # telemetry bucket
        self._telemetry: dict[tuple[str, str], list[TelemetryEvent]] = {}  # (system_session_id, run_id)
        self._span_payloads: dict[tuple[str, str, str], SpanPayloadEnvelope] = {}  # (system_session_id, run_id, span_id)
        # optional quick lookup for concluded outputs (used by some callers)
        self._run_outputs: dict[str, dict] = {}
        # workflow settings
        self._wf_settings: dict[str, tuple[dict, bool]] = {}
        # provider/agent settings
        self._provider_settings: dict[str, tuple[dict, bool]] = {}
        self._agent_settings: dict[str, tuple[dict, bool]] = {}
        # system sessions
        self._sessions: dict[str, dict] = {}

    async def ensure_system_session(self, system_session_id: str, name: str | None = None) -> None:
        if system_session_id not in self._sessions:
            self._sessions[system_session_id] = {
                "system_session_id": system_session_id,
                "name": name,
                "created_at": datetime.now(timezone.utc),
            }
        else:
            if name is not None:
                self._sessions[system_session_id]["name"] = name
        self._counter.setdefault(system_session_id, 0)

    async def get_system_session(self, system_session_id: str):
        from wrp.server.runtime.system_sessions.types import SystemSession

        d = self._sessions.get(system_session_id)
        return (
            SystemSession(
                system_session_id=d["system_session_id"],
                name=d.get("name"),
                created_at=d.get("created_at") or datetime.now(timezone.utc),
            )
            if d
            else None
        )

    async def list_system_sessions(self):
        from wrp.server.runtime.system_sessions.types import SystemSession

        return [
            SystemSession(
                system_session_id=k,
                name=v.get("name"),
                created_at=v.get("created_at") or datetime.now(timezone.utc),
            )
            for k, v in sorted(
                self._sessions.items(), key=lambda kv: kv[1].get("created_at") or datetime.now(timezone.utc)
            )
        ]

    async def alloc_run_id(self, system_session_id: str, workflow_name: str, thread_id: str | None) -> str:  # noqa: ARG002
        cur = self._counter.get(system_session_id, 0) + 1
        if cur > 999:
            raise ValueError("Run ID capacity reached (001..999). Please rotate or archive old runs.")
        self._counter[system_session_id] = cur
        return f"{cur:03d}"

    async def create_run(self, meta: RunMeta) -> None:
        key = (meta.system_session_id, meta.run_id)
        self._runs[key] = meta
        self._conv[key] = []
        self._conv_meta[key] = {}

    async def conclude_run(
        self,
        system_session_id: str,
        run_id: str,
        outcome: RunOutcome,
        *,
        error: str | None = None,
        run_output: dict | None = None,
    ) -> None:
        key = (system_session_id, run_id)
        meta = self._runs[key]
        meta.state = RunState.concluded
        meta.outcome = outcome
        meta.error = error
        meta.run_output = run_output
        if run_output is not None:
            self._run_outputs[run_id] = run_output
        self._runs[key] = meta

    async def get_run(self, system_session_id: str, run_id: str) -> RunMeta | None:
        return self._runs.get((system_session_id, run_id))

    async def runs_in_thread(self, system_session_id: str, workflow_name: str, thread_id: str) -> list[RunMeta]:
        return sorted(
            (
                m
                for (sid, _), m in self._runs.items()
                if sid == system_session_id and m.workflow_name == workflow_name and m.thread_id == thread_id
            ),
            key=lambda m: m.created_at,
        )

    async def list_runs(
        self,
        system_session_id: str,
        *,
        workflow_name: str | None = None,
        thread_id: str | None = None,
        state: RunState | None = None,
        outcome: RunOutcome | None = None,
    ) -> list[RunMeta]:
        runs = [
            meta
            for (sid, _), meta in self._runs.items()
            if sid == system_session_id
        ]
        if workflow_name is not None:
            runs = [m for m in runs if m.workflow_name == workflow_name]
        if thread_id is not None:
            runs = [m for m in runs if m.thread_id == thread_id]
        if state is not None:
            runs = [m for m in runs if m.state == state]
        if outcome is not None:
            runs = [m for m in runs if m.outcome == outcome]
        runs.sort(key=lambda m: m.created_at)
        return runs

    async def append_conversation_channel_item(
        self, system_session_id: str, run_id: str, items: Iterable[ChannelItem]
    ) -> None:
        key = (system_session_id, run_id)
        items = list(items)
        self._conv.setdefault(key, [])
        self._conv[key].extend(items)
        # channel meta counters
        if items:
            bump = Counter(i.channel for i in items)
            for ch, n in bump.items():
                ch_meta = self._conv_meta[key].setdefault(
                    ch, {"items_count": 0, "last_ts": None, "name": None, "description": None}
                )
                ch_meta["items_count"] += n
                last_ts = max(i.ts for i in items if i.channel == ch)
                ch_meta["last_ts"] = (
                    last_ts if (ch_meta["last_ts"] is None or last_ts > ch_meta["last_ts"]) else ch_meta["last_ts"]
                )

    async def list_channel_meta(self, system_session_id: str, run_id: str) -> List[ChannelMeta]:
        key = (system_session_id, run_id)
        out: List[ChannelMeta] = []
        for ch, d in sorted(self._conv_meta.get(key, {}).items()):
            out.append(
                ChannelMeta(
                    id=ch,
                    name=d.get("name"),
                    description=d.get("description"),
                    itemsCount=int(d.get("items_count", 0)),
                    lastItemTs=d.get("last_ts"),
                )
            )
        return out

    async def ensure_channel_meta(
        self,
        system_session_id: str,
        run_id: str,
        channel: str,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        """Ensures that a metadata entry for a channel exists."""
        key = (system_session_id, run_id)
        bucket = self._conv_meta.setdefault(key, {})
        if channel not in bucket:
            bucket[channel] = {
                "items_count": 0,
                "last_ts": None,
                "name": name,
                "description": description,
            }

    async def load_channel_items(
        self, system_session_id: str, run_id: str, *, channel: str, limit: int | None = None
    ) -> List[ChannelItem]:
        items = [i for i in self._conv.get((system_session_id, run_id), []) if i.channel == channel]
        if limit is not None and limit >= 0:
            items = items[-limit:]
        return items

    # ---- telemetry (new) ----
    async def append_telemetry_span_event(self, system_session_id: str, run_id: str, events: Iterable[TelemetryEvent]) -> None:
        bucket = self._telemetry.setdefault((system_session_id, run_id), [])
        bucket.extend(list(events))

    async def load_telemetry(
        self,
        system_session_id: str,
        run_id: str,
        *,
        kinds: set[str] | None = None,
        limit: int | None = None,
    ) -> List[TelemetryEvent]:
        items = list(self._telemetry.get((system_session_id, run_id), []))
        if kinds:
            items = [e for e in items if getattr(e, "kind", None) in kinds]
        if limit is not None and limit >= 0:
            # ensure the last N are returned oldest→newest, matching SQL stores
            return items[-limit:]
        return items

    # ---- span payloads (start/end) --------------------------------------
    async def upsert_span_payload(self, system_session_id: str, run_id: str, payload: SpanPayloadEnvelope) -> None:
        key = (system_session_id, run_id, payload.span_id)
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

    async def get_span_payload(
        self, system_session_id: str, run_id: str, span_id: str
    ) -> SpanPayloadEnvelope | None:
        return self._span_payloads.get((system_session_id, run_id, span_id))

    # ---- workflow settings (new) -----------------------------------------
    async def upsert_workflow_settings(self, workflow_name: str, values: dict, *, overridden: bool) -> None:
        self._wf_settings[workflow_name] = (dict(values), bool(overridden))

    async def get_workflow_settings(self, workflow_name: str) -> tuple[dict, bool] | None:
        return self._wf_settings.get(workflow_name)

    async def list_workflow_settings(self) -> dict[str, tuple[dict, bool]]:
        return dict(self._wf_settings)

    # ---- provider settings -----------------------------------------------
    async def upsert_provider_settings(self, provider_name: str, values: dict, *, overridden: bool) -> None:
        self._provider_settings[provider_name] = (dict(values), bool(overridden))

    async def get_provider_settings(self, provider_name: str) -> tuple[dict, bool] | None:
        return self._provider_settings.get(provider_name)

    async def list_provider_settings(self) -> dict[str, tuple[dict, bool]]:
        return dict(self._provider_settings)

    # ---- agent settings ---------------------------------------------------
    async def upsert_agent_settings(self, agent_name: str, values: dict, *, overridden: bool) -> None:
        self._agent_settings[agent_name] = (dict(values), bool(overridden))

    async def get_agent_settings(self, agent_name: str) -> tuple[dict, bool] | None:
        return self._agent_settings.get(agent_name)

    async def list_agent_settings(self) -> dict[str, tuple[dict, bool]]:
        return dict(self._agent_settings)
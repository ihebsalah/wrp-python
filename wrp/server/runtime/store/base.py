# wrp/server/runtime/store/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Iterable, List

from wrp.server.runtime.conversations.types import (
    ChannelItem,
    ChannelMeta,
)
from wrp.server.runtime.runs.types import RunMeta, RunOutcome, RunState
from wrp.server.runtime.telemetry.events import TelemetryEvent
from wrp.server.runtime.telemetry.payloads.types import SpanPayloadEnvelope

if TYPE_CHECKING:
    from wrp.server.runtime.system_sessions.types import SystemSession


class Store(ABC):
    """Persistence contract for runs, conversations, and telemetry."""

    # ---- system sessions -------------------------------------------------
    @abstractmethod
    async def ensure_system_session(
        self, system_session_id: str, name: str | None = None
    ) -> None: ...

    @abstractmethod
    async def get_system_session(
        self, system_session_id: str
    ) -> "SystemSession | None": ...

    @abstractmethod
    async def list_system_sessions(self) -> list["SystemSession"]: ...

    # allocation
    @abstractmethod
    async def alloc_run_id(
        self, system_session_id: str, workflow_name: str, thread_id: str | None
    ) -> str: ...

    # lifecycle
    @abstractmethod
    async def create_run(self, meta: RunMeta) -> None: ...

    @abstractmethod
    async def conclude_run(
        self,
        system_session_id: str,
        run_id: str,
        outcome: RunOutcome,
        *,
        error: str | None = None,
        run_output: dict | None = None,
    ) -> None: ...

    # lookups
    @abstractmethod
    async def get_run(self, system_session_id: str, run_id: str) -> RunMeta | None: ...

    @abstractmethod
    async def runs_in_thread(
        self, system_session_id: str, workflow_name: str, thread_id: str
    ) -> list[RunMeta]: ...

    @abstractmethod
    async def list_runs(
        self,
        system_session_id: str,
        *,
        workflow_name: str | None = None,
        thread_id: str | None = None,
        state: RunState | None = None,
        outcome: RunOutcome | None = None,
    ) -> list[RunMeta]: ...

    # conversation: append + channel-scoped read paths
    @abstractmethod
    async def append_conversation_channel_item(
        self, system_session_id: str, run_id: str, items: Iterable[ChannelItem]
    ) -> None: ...

    @abstractmethod
    async def list_channel_meta(
        self,
        system_session_id: str,
        run_id: str,
    ) -> List[ChannelMeta]: ...

    # channel meta bootstrap (create-if-absent without affecting counters)
    @abstractmethod
    async def ensure_channel_meta(
        self,
        system_session_id: str,
        run_id: str,
        channel: str,
        name: str | None = None,
        description: str | None = None,
        item_type: str | None = None,
    ) -> None: ...

    @abstractmethod
    async def load_channel_items(
        self,
        system_session_id: str,
        run_id: str,
        *,
        channel: str,
        limit: int | None = None,
    ) -> List[ChannelItem]: ...

    # ---- telemetry (new) ----
    @abstractmethod
    async def append_telemetry_span_event(
        self, system_session_id: str, run_id: str, events: Iterable[TelemetryEvent]
    ) -> None: ...

    @abstractmethod
    async def load_telemetry(
        self,
        system_session_id: str,
        run_id: str,
        *,
        kinds: set[str] | None = None,
        limit: int | None = None,
    ) -> List[TelemetryEvent]: ...

    # ---- span payloads (start/end) --------------------------------------
    @abstractmethod
    async def upsert_span_payload(
        self, system_session_id: str, run_id: str, payload: SpanPayloadEnvelope
    ) -> None: ...

    @abstractmethod
    async def get_span_payload(
        self, system_session_id: str, run_id: str, span_id: str
    ) -> SpanPayloadEnvelope | None: ...

    # ---- workflow settings (new) -----------------------------------------
    @abstractmethod
    async def upsert_workflow_settings(
        self, workflow_name: str, values: dict, *, overridden: bool
    ) -> None: ...

    @abstractmethod
    async def get_workflow_settings(
        self, workflow_name: str
    ) -> tuple[dict, bool] | None: ...

    @abstractmethod
    async def list_workflow_settings(self) -> dict[str, tuple[dict, bool]]: ...

    # ---- provider settings (new) -----------------------------------------
    @abstractmethod
    async def upsert_provider_settings(
        self, provider_name: str, values: dict, *, overridden: bool
    ) -> None: ...

    @abstractmethod
    async def get_provider_settings(
        self, provider_name: str
    ) -> tuple[dict, bool] | None: ...

    @abstractmethod
    async def list_provider_settings(self) -> dict[str, tuple[dict, bool]]: ...

    # ---- agent settings (new) --------------------------------------------
    @abstractmethod
    async def upsert_agent_settings(
        self, agent_name: str, values: dict, *, overridden: bool
    ) -> None: ...

    @abstractmethod
    async def get_agent_settings(
        self, agent_name: str
    ) -> tuple[dict, bool] | None: ...

    @abstractmethod
    async def list_agent_settings(self) -> dict[str, tuple[dict, bool]]: ...
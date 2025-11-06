# wrp/server/runtime/store/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, List, Optional, Set

from wrp.server.runtime.runs.types import RunMeta, RunOutcome
from wrp.server.runtime.conversations.types import ConversationItem
from wrp.server.runtime.telemetry.events import TelemetryEvent
from wrp.server.runtime.telemetry.payloads.types import SpanPayloadEnvelope


class Store(ABC):
    """Persistence contract for runs, conversations, and telemetry."""

    def supports_message_counts(self) -> bool:
        """
        Optional capability flag. When True, `RunMeta.message_count` and
        `RunMeta.channel_counts` are reliable and reflect all appended items.
        Stores that return True must update counts atomically alongside
        conversation appends so seed-assembly fast paths can safely skip empties.
        """
        return False

    # allocation
    @abstractmethod
    async def alloc_run_id(self, workflow_name: str, thread_id: str | None) -> str: ...

    # lifecycle
    @abstractmethod
    async def create_run(self, meta: RunMeta) -> None: ...

    @abstractmethod
    async def conclude_run(
        self,
        run_id: str,
        outcome: RunOutcome,
        *,
        error: str | None = None,
        run_output: dict | None = None,
    ) -> None: ...

    # lookups
    @abstractmethod
    async def get_run(self, run_id: str) -> RunMeta | None: ...

    @abstractmethod
    async def runs_in_thread(self, workflow_name: str, thread_id: str) -> list[RunMeta]: ...

    # conversation
    @abstractmethod
    async def append_conversation(self, run_id: str, items: Iterable[ConversationItem]) -> None: ...

    @abstractmethod
    async def load_conversation(self, run_id: str) -> list[ConversationItem]: ...

    async def load_conversation_tail(
        self,
        run_id: str,
        *,
        limit: int,
        channels: Optional[Set[str]] = None,
    ) -> list[ConversationItem]:
        """
        Optional fast-path: return the last `limit` items (filtered by channels),
        in ascending ts order. Default fallback uses load_conversation().
        """
        items = await self.load_conversation(run_id)
        if channels:
            items = [i for i in items if i.channel in channels]
        return items[-limit:]

    # ---- telemetry (new) ----
    @abstractmethod
    async def append_telemetry(self, run_id: str, events: Iterable[TelemetryEvent]) -> None: ...

    @abstractmethod
    async def load_telemetry(
        self,
        run_id: str,
        *,
        kinds: Optional[Set[str]] = None,
        limit: Optional[int] = None,
    ) -> List[TelemetryEvent]: ...

    # ---- span payloads (start/end) --------------------------------------
    @abstractmethod
    async def upsert_span_payload(self, run_id: str, payload: SpanPayloadEnvelope) -> None: ...

    @abstractmethod
    async def get_span_payload(self, run_id: str, span_id: str) -> SpanPayloadEnvelope | None: ...

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
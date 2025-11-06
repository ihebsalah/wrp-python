# wrp/server/runtime/conversations/service.py
from __future__ import annotations

from typing import Any, Awaitable, Callable, Iterable, List, Set

from wrp.server.runtime.exceptions import RunStateError
from .types import ConversationItem
from wrp.server.runtime.runs.types import RunMeta, RunState
from wrp.server.runtime.store.base import Store
from wrp.server.runtime.store.codecs import serializer
from .assembler import assemble_seed, select_runs
from .seeding import ConversationSeeding, RunFilter, default_conversation_seeding


def _string_to_payload(text: str, *, role: str = "user", content_type: str = "text") -> dict:
    """
    Produce a minimal, neutral text payload:
      {"role": "<role>", "content": [{"type": "<content_type>", "text": "..."}]}
    If callers want OpenAI's "input_text", they can pass content_type="input_text".
    """
    return {"role": role, "content": [{"type": content_type, "text": text}]}


class ConversationsService:
    """Author-facing conversations API bound to a single run."""

    def __init__(
        self,
        store: Store,
        current_run: RunMeta,
        *,
        conversation_seeding: ConversationSeeding | None = None,
        run_filter: RunFilter | None = None,
        default_channels: List[str] | None = None,
        allowed_channels: Set[str] | None = None,
        on_update: Callable[[str], Awaitable[None]] | None = None,
    ):
        self._store = store
        self._run = current_run
        self._seeding = conversation_seeding or default_conversation_seeding()
        self._filter = run_filter or RunFilter()
        self._default_channels = default_channels or ["default"]
        self._allowed_channels = allowed_channels
        self._on_update = on_update

    async def add_item(
        self,
        item_or_items: str | dict[str, Any] | Iterable[str | dict[str, Any]],
        *,
        role: str | None = None,
        channel: str | None = None,
        content_type: str = "text",
    ) -> None:
        # refresh local snapshot
        meta = await self._store.get_run(self._run.run_id)
        if meta:
            self._run = meta
        if self._run.state != RunState.running:
            raise RunStateError("Cannot append conversation items after run concluded")

        lane = channel or "default"

        def to_item(x: Any) -> ConversationItem:
            if isinstance(x, (bytes, bytearray)):
                raise TypeError("add_item does not accept raw bytes; base64-encode and put them inside a dict payload.")
            if isinstance(x, str):
                payload = _string_to_payload(x, role=role or "user", content_type=content_type)
                # validate JSON-serializable (authors must encode binaries)
                serializer.dumps(payload)
                return ConversationItem(payload=payload, channel=lane)
            if isinstance(x, dict):
                # pass-through; just ensure JSON-serializable
                serializer.dumps(x)
                return ConversationItem(payload=x, channel=lane)
            raise TypeError("add_item only accepts str, dict, or iterables of those.")

        if isinstance(item_or_items, (str, dict)):
            items = [to_item(item_or_items)]
        else:
            items = [to_item(d) for d in item_or_items]
        await self._store.append_conversation(self._run.run_id, items)

        # notify resource subscribers (best-effort)
        if self._on_update:
            try:
                # whole-run conversation
                await self._on_update(f"resource://runs/{self._run.run_id}/conversations")
                # channel-specific (notify the lane actually used)
                await self._on_update(f"resource://runs/{self._run.run_id}/conversations/{lane}")
            except Exception:
                pass

    async def _collect_seed(self, *, channels: list[str] | None = None) -> list[dict[str, Any]]:
        """Internal: collect seed (seeding+filter) + current-run items for channels."""
        requested = channels if channels is not None else list(self._default_channels)
        # constrain by allowed_channels if configured
        if self._allowed_channels is not None:
            requested = [c for c in requested if c in self._allowed_channels]

        # if nothing survives filtering, return an empty list to avoid falling back to defaults
        channel_set = set(requested)
        if not channel_set:
            return []

        # prior runs (exclude current)
        candidates = await select_runs(
            self._store,
            workflow_name=self._run.workflow_name,
            thread_id=self._run.thread_id,
            run_filter=self._filter,
            exclude_run_id=self._run.run_id,
        )
        seed = await assemble_seed(self._store, candidates, self._seeding, channels=channel_set)

        # current run items
        current = [i for i in await self._store.load_conversation(self._run.run_id) if i.channel in channel_set]

        # return raw payloads ascending by ts
        merged = [*seed, *current]
        merged.sort(key=lambda i: i.ts)
        return [i.payload for i in merged]

    async def get_items(self, *, channels: list[str] | None = None, limit: int | None = None) -> list[dict[str, Any]]:
        """
        Convenience: return the seeded + current-run items (ascending ts).
        If `limit` is set, returns the last N.
        """
        items = await self._collect_seed(channels=channels)
        if limit is not None and limit >= 0:
            return items[-limit:]
        return items

    async def get(self, *, channels: list[str] | None = None) -> "Conversation":
        """
        Public: returns a live Conversation handle whose `.items` is the seeded list
        (seeding + run_filter + current-run items) and which can be appended to.
        """
        seed = await self._collect_seed(channels=channels)
        return Conversation(self, channels, seed)


class Conversation:
    """
    Seed + live handle:
      - .items → list passed to model
      - .add_item(...) → persists via service.add_item(...) and updates .items
    """

    def __init__(self, service: "ConversationsService", channels: list[str] | None, items: List[dict]):
        self._service = service
        self._channels = channels or []
        self.items: List[dict] = list(items)

    def get_items(self) -> List[dict]:
        """Return a shallow copy of the live items list."""
        return list(self.items)

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

    async def add_item(
        self,
        item_or_items: str | dict[str, Any] | Iterable[str | dict[str, Any]],
        *,
        role: str | None = None,
        channel: str | None = None,
        content_type: str = "text",
    ) -> None:
        lane = channel or (self._channels[0] if self._channels else "default")

        def normalize(x: Any) -> dict[str, Any]:
            if isinstance(x, str):
                return _string_to_payload(x, role=role or "user", content_type=content_type)
            if isinstance(x, dict):
                return x
            raise TypeError("Conversation.add_item only accepts str, dict, or iterables of those.")

        # persist first
        await self._service.add_item(item_or_items, role=role, channel=lane, content_type=content_type)
        # mirror into local items
        if isinstance(item_or_items, (str, dict)):
            self.items.append(normalize(item_or_items))
        else:
            self.items.extend(normalize(d) for d in item_or_items)
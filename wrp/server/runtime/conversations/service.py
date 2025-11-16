# wrp/server/runtime/conversations/service.py
from __future__ import annotations

from typing import Any, Awaitable, Callable, Iterable, List, Set

from wrp.server.runtime.exceptions import RunStateError
from .types import ChannelItem, ChannelMeta
from wrp.server.runtime.runs.types import RunMeta, RunState
from wrp.server.runtime.store.base import Store
from wrp.server.runtime.store.codecs import serializer
from .assembler import assemble_seed, select_runs
from .seeding import ConversationSeeding, SeedingRunFilter, default_conversation_seeding
import wrp.types as types


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
        seeding_run_filter: SeedingRunFilter | None = None,
        allowed_channels: Set[str] | None = None,
        emit_system_event: Callable[..., Awaitable[None]] | None = None,
    ):
        self._store = store
        self._run = current_run
        self._seeding = conversation_seeding or default_conversation_seeding()
        self._filter = seeding_run_filter or SeedingRunFilter()
        self._allowed_channels = allowed_channels
        self._emit_system_event = emit_system_event

    async def add_item(
        self,
        item_or_items: str | dict[str, Any] | Iterable[str | dict[str, Any]],
        *,
        role: str | None = None,
        channel: str | None = None,
        content_type: str = "text",
    ) -> None:
        # refresh local snapshot
        meta = await self._store.get_run(self._run.system_session_id, self._run.run_id)
        if meta:
            self._run = meta
        if self._run.state != RunState.running:
            raise RunStateError("Cannot append conversation items after run concluded")

        channel = channel or "default"

        def to_item(x: Any) -> ChannelItem:
            if isinstance(x, (bytes, bytearray)):
                raise TypeError("add_item does not accept raw bytes; base64-encode and put them inside a dict payload.")
            if isinstance(x, str):
                payload = _string_to_payload(x, role=role or "user", content_type=content_type)
                # validate JSON-serializable (authors must encode binaries)
                serializer.dumps(payload)
                return ChannelItem(payload=payload, channel=channel)
            if isinstance(x, dict):
                # pass-through; just ensure JSON-serializable
                serializer.dumps(x)
                return ChannelItem(payload=x, channel=channel)
            raise TypeError("add_item only accepts str, dict, or iterables of those.")

        if isinstance(item_or_items, (str, dict)):
            items = [to_item(item_or_items)]
        else:
            items = [to_item(d) for d in item_or_items]
        await self._store.append_conversation_channel_item(
            self._run.system_session_id, self._run.run_id, items
        )

        # notify conversation subscribers via system events (best-effort)
        if self._emit_system_event:
            try:
                runs_scope = types.RunsScope(
                    system_session_id=self._run.system_session_id,
                    run_id=self._run.run_id,
                )
                # channels index for this run
                await self._emit_system_event(
                    topic="conversations/channels",
                    change="refetch",
                    runs=runs_scope,
                )
                # specific channel for this run
                await self._emit_system_event(
                    topic="conversations/channel",
                    change="refetch",
                    channel=types.ChannelScope(
                        system_session_id=self._run.system_session_id,
                        run_id=self._run.run_id,
                        channel=channel,
                    ),
                )
            except Exception:
                # best-effort only; don't break authors' workflows
                pass

    async def _collect_seed(self, *, channel: str | None) -> list[dict[str, Any]]:
        """
        Internal: collect seed strictly for a single channel (same-channel only),
        then append current-run items for that channel. If `channel` is None,
        use 'default'.
        """
        ch = channel or "default"

        # Cross-run seeding is allowed only if channel is permitted (when a policy is set)
        allow_cross_run = (self._allowed_channels is None) or (ch in self._allowed_channels)

        seed_items: list[ChannelItem] = []
        if allow_cross_run:
            candidates = await select_runs(
                self._store,
                system_session_id=self._run.system_session_id,
                workflow_name=self._run.workflow_name,
                thread_id=self._run.thread_id,
                seeding_run_filter=self._filter,
                exclude_run_id=self._run.run_id,
            )
            seed_items = await assemble_seed(
                self._store,
                self._run.system_session_id,
                candidates,
                self._seeding,
                channels={ch},
            )

        # Current run items (single channel)
        current = await self._store.load_channel_items(
            self._run.system_session_id,
            self._run.run_id,
            channel=ch,
            limit=10**9,
        )

        # return raw payloads ascending by ts
        merged_items = [*seed_items, *current]
        merged_items.sort(key=lambda i: i.ts)
        return [i.payload for i in merged_items]

    async def get_channel_items(self, *, channel: str, limit: int | None = None) -> list[dict[str, Any]]:
        """
        Return sanitized-ready raw payloads for a single channel within this run,
        including seeded items from prior runs (per seeding/filter).
        Ordered ascending by ts. If `limit` is set, return the last N.
        """
        items = await self._collect_seed(channel=channel)
        if limit is not None and limit >= 0:
            return items[-limit:]
        return items

    async def get_channel(
        self,
        channel: str | None = None,
        name: str | None = None,
        description: str | None = None,
    ) -> "ChannelHandle":
        """
        Public: returns a live ChannelHandle whose `.items` is the seeded list
        (same-channel seeding + current-run items) and which can be appended to.
        Creation semantics:
        - If `channel` is None and store is in-memory, a random id is generated.
        - If the channel row does not exist yet, persist `name`/`description` (if provided).
          On subsequent calls, any provided `name`/`description` is ignored.
        """
        ch = channel
        if ch is None:
            # random id only for the in-memory store (per requirement)
            store_name = self._store.__class__.__name__
            if store_name == "InMemoryStore":
                import uuid
                ch = f"ch_{uuid.uuid4().hex[:8]}"
            else:
                raise ValueError("channel id must be provided for durable stores")

        # if meta row is absent, create it with provided name/description; otherwise ignore
        metas = await self._store.list_channel_meta(self._run.system_session_id, self._run.run_id)
        exists = any(m.id == ch for m in metas)
        if not exists:
            await self._store.ensure_channel_meta(
                self._run.system_session_id,
                self._run.run_id,
                ch,
                name=name,
                description=description,
            )

        seed = await self._collect_seed(channel=ch)
        return ChannelHandle(self, ch, seed)

    async def list_channels_meta(self) -> List[ChannelMeta]:
        """Return per-channel metadata for this run (cheap)."""
        return await self._store.list_channel_meta(self._run.system_session_id, self._run.run_id)


class ChannelHandle:
    """
    Single-channel, seed + live handle:
      - .items → list passed to model
      - .add_item(...) → persists via service.add_item(...) and updates .items
    """

    def __init__(self, service: "ConversationsService", channel: str, items: List[dict]):
        self._service = service
        self._channel = channel
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
        content_type: str = "text",
    ) -> None:
        def normalize(x: Any) -> dict[str, Any]:
            if isinstance(x, str):
                return _string_to_payload(x, role=role or "user", content_type=content_type)
            if isinstance(x, dict):
                return x
            raise TypeError("ChannelHandle.add_item only accepts str, dict, or iterables of those.")

        # persist first
        await self._service.add_item(item_or_items, role=role, channel=self._channel, content_type=content_type)
        # mirror into local items
        if isinstance(item_or_items, (str, dict)):
            self.items.append(normalize(item_or_items))
        else:
            self.items.extend(normalize(d) for d in item_or_items)
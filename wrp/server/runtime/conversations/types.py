# wrp/server/runtime/conversations/types.py
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

class ChannelItem(BaseModel):
    """Opaque, SDK-agnostic message item stored for conversations."""
    payload: dict[str, Any]
    # Logical lane inside a run;
    channel: str = "default"
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class SanitizedChannelItem(BaseModel):
    """Public, sanitized view of a channel item, indicating if any data was redacted."""
    payload: dict[str, Any]
    channel: str
    ts: datetime
    redacted: bool

class ChannelMeta(BaseModel):
    """
    Lightweight metadata for a conversation channel (per run).
    - id: persistent channel identifier (used in storage; previously `channel`)
    - name: human-friendly display name
    - description: optional description
    - itemType: persisted item type token (declared by author; for discovery/introspection)
    """
    id: str
    name: str | None = None
    description: str | None = None
    itemsCount: int | None = None
    lastItemTs: datetime | None = None
    # Persisted item type token (declared by the author; for discovery/introspection).
    itemType: str | None = None

class ChannelView(BaseModel):
    """
    Rich channel view: metadata + items (items are the sanitized view).
    Returned by channel/read.
    """
    meta: ChannelMeta
    items: list[SanitizedChannelItem]
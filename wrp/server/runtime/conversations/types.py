# wrp/server/runtime/conversations/types.py
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

class ConversationItem(BaseModel):
    """Opaque, SDK-agnostic message item stored for conversations."""
    payload: dict[str, Any]
    # Logical lane inside a run;
    channel: str = "default"
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
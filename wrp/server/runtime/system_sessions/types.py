# wrp/server/runtime/system_sessions/types.py
from __future__ import annotations
from datetime import datetime, timezone
from pydantic import BaseModel, Field


class SystemSession(BaseModel):
    """Top-level namespace for all persisted artifacts on a server."""
    system_session_id: str = Field(description="Opaque, client-chosen id")
    name: str | None = Field(default=None, description="Optional human label")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

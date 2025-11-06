# wrp/server/runtime/runs/types.py
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from wrp.server.runtime.conversations.seeding import ConversationSeeding, RunFilter

# Structured settings emitted at run_start
class RunSettings(BaseModel):
    conversation_seeding: ConversationSeeding
    run_filter: RunFilter
    ignore_thread: bool = False

    # effective author seeding bits (what the server enforced)
    default_channels: list[str] | None = None
    allowed_channels: set[str] | None = None
    seeding_deny: bool = False

    # whether caller provided/overrode these
    conversation_seeding_specified: bool | None = None
    run_filter_specified: bool | None = None


class RunState(str, Enum):
    running = "running"
    concluded = "concluded"


class RunOutcome(str, Enum):
    success = "success"
    failed = "failed"
    error = "error"


class RunMeta(BaseModel):
    """Lightweight header for a single workflow run."""
    run_id: str
    workflow_name: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    state: RunState = RunState.running

    # end-user provided grouping (scoped per workflow)
    thread_id: str | None = None

    # lightweight, optional hints used for seed-assembly fast paths
    message_count: int = 0
    channel_counts: dict[str, int] = Field(default_factory=dict)

    # conclusion
    outcome: RunOutcome | None = None
    error: str | None = None
    run_output: dict[str, Any] | None = None
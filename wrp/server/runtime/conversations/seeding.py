# wrp/server/runtime/conversations/seeding.py
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ---- Conversation seeding (v0) ----------------------------------------------------

class ConversationSeedingNone(BaseModel):
    kind: Literal["none"] = "none"


class ConversationSeedingWindow(BaseModel):
    kind: Literal["window"] = "window"
    messages: int = Field(gt=0)


ConversationSeeding = ConversationSeedingNone | ConversationSeedingWindow


def default_conversation_seeding() -> ConversationSeeding:
    return ConversationSeedingNone()


# ---- Run filter (v0) ---------------------------------------------------------

class SeedingRunFilter(BaseModel):
    """Select which prior runs contribute to the seed."""
    include_runs: list[str] | None = None
    since_run_id: str | None = None     # exclusive
    until_run_id: str | None = None     # inclusive (v0 treat as inclusive for simplicity)
    exclude_runs: list[str] | None = None

    model_config = {"extra": "allow"}  # future-proof


def normalize_conversation_seeding(value: Any | None) -> ConversationSeeding:
    if value is None:
        return default_conversation_seeding()
    if isinstance(value, dict):
        kind = value.get("kind", "none")
        if kind == "none":
            return ConversationSeedingNone()
        if kind == "window":
            if "messages" not in value:
                raise ValueError("conversation_seeding.window requires 'messages'")
            return ConversationSeedingWindow(messages=int(value["messages"]))
    if value == "none":
        return ConversationSeedingNone()
    raise ValueError("Invalid conversation_seeding")

class WorkflowConversationSeeding(BaseModel):
    """
    Author-controlled constraints for cross-run seeding and channel access.
    Enforced by the server regardless of caller meta.

    - deny_seeding: hard-disable cross-run seeding (forces ConversationSeedingNone and ignores seeding_run_filter)
    - default_seeding: used ONLY when caller does not provide a conversation_seeding
    - allowed_channels: if set, cross-run seeding is only performed for channels in this set
    """
    deny_seeding: bool = False
    default_seeding: ConversationSeeding = Field(default_factory=ConversationSeedingNone)
    allowed_channels: set[str] | None = None
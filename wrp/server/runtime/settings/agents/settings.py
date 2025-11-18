# wrp/server/runtime/settings/agents/settings.py
from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, PrivateAttr


class AgentSettings(BaseModel):
    """
    Base class for author-defined agent configuration.

    This is a global (non-run) configuration blob keyed by agent name.
    It is provider-aware via the required `provider_name` field and typically
    includes at least a default model.

    Authors are free to extend subclasses with whatever tuning knobs they
    care about (temperature, max_tokens, allowed model list, metadata, etc.).
    """

    provider_name: str
    model: str
    # Always-locked coarse/fine allowlists (author-defined; end-users cannot override)
    # - allowed_providers: if present, only these providers are selectable
    # - allowed_models:    per-provider list of allowed model names
    allowed_providers: list[str] | None = None
    allowed_models: dict[str, list[str]] | None = None

    model_config = ConfigDict(extra="allow")

    _override_status: bool = PrivateAttr(default=False)
    # NEVER overrideable at runtime
    locked: ClassVar[set[str]] = {"allowed_providers", "allowed_models"}

    def settings_overridden(self) -> bool:
        return bool(self._override_status)
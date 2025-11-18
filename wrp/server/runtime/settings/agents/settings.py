# wrp/server/runtime/settings/agents/settings.py
from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator


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

    # Runtime-injected provider settings (Auto-Merge).
    # Excluded from serialization so it is never persisted with the agent.
    provider: Any | None = Field(default=None, exclude=True)

    model_config = ConfigDict(extra="allow")

    _override_status: bool = PrivateAttr(default=False)
    # NEVER overrideable at runtime
    locked: ClassVar[set[str]] = {"allowed_providers", "allowed_models"}

    def settings_overridden(self) -> bool:
        return bool(self._override_status)

    @model_validator(mode="after")
    def validate_allowlists(self) -> AgentSettings:
        """
        Enforce allowed_providers and allowed_models constraints if they are defined.
        """
        # 1. Check Provider
        if self.allowed_providers is not None:
            if self.provider_name not in self.allowed_providers:
                raise ValueError(
                    f"Provider '{self.provider_name}' is not allowed. "
                    f"Allowed: {self.allowed_providers}"
                )

        # 2. Check Model (if an allowlist exists for this provider)
        if self.allowed_models is not None:
            valid_models = self.allowed_models.get(self.provider_name)
            if valid_models is not None and self.model not in valid_models:
                raise ValueError(
                    f"Model '{self.model}' is not allowed for provider '{self.provider_name}'. "
                    f"Allowed models: {valid_models}"
                )

        return self
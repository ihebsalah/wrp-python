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

    model_config = ConfigDict(extra="allow")

    _override_status: bool = PrivateAttr(default=False)
    locked: ClassVar[set[str]] = set()

    def settings_overridden(self) -> bool:
        return bool(self._override_status)

# wrp/server/runtime/settings/providers/settings.py
from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, PrivateAttr


class ProviderSettings(BaseModel):
    """
    Base class for author-defined provider configuration.

    Instances are global (not per-run) and may contain secrets such as API keys.
    Secrets are persisted via the Store using the existing encryption-at-rest
    codecs and are always masked when serialized back to clients.
    """

    model_config = ConfigDict(extra="allow")

    # Internal, instance-scoped flag; registry also exposes a canonical view.
    _override_status: bool = PrivateAttr(default=False)

    # Field-level locks (declare in subclasses: e.g., `locked = {"endpoint"}`).
    locked: ClassVar[set[str]] = set()

    def settings_overridden(self) -> bool:
        """
        True if this settings object reflects an explicit client override.
        The registry keeps this flag in sync when overrides are applied.
        """
        return bool(self._override_status)

# wrp/server/runtime/settings/providers/settings.py
from __future__ import annotations

import os
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, PrivateAttr, SecretStr


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

    def on_provider_set(self, provider_name: str) -> None:
        """
        Hook called when these settings are hydrated (loaded) or updated.

        Default behavior:
        Automatically sets environment variables for every field in the settings.
        Format: {PROVIDER_NAME}_{FIELD_NAME} (uppercased).

        Example:
            provider="openai", api_key="sk-..." -> OPENAI_API_KEY="sk-..."

        Authors can override this method in subclasses to customize env var mapping.
        """
        prefix = provider_name.upper().replace("-", "_")
        for key, value in self.model_dump().items():
            if value is None:
                continue

            env_key = f"{prefix}_{key.upper()}"
            # Reveal secret values for the environment
            env_val = value.get_secret_value() if isinstance(value, SecretStr) else str(value)
            os.environ[env_key] = env_val
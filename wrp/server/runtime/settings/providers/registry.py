# wrp/server/runtime/settings/providers/registry.py
from __future__ import annotations

from typing import Any, Dict

from pydantic import SecretStr
from mcp.server.fastmcp.utilities.logging import get_logger

from wrp.server.runtime.store.base import Store
from .settings import ProviderSettings
from .builtin import (
    OpenAIProviderSettings, AnthropicProviderSettings, GoogleProviderSettings, LiteLLMProviderSettings
)

logger = get_logger(__name__)


class ProviderSettingsRegistry:
    """
    Registry + persistence bridge for provider settings.

    - Authors register defaults once at startup.
    - Overrides are merged/validated and persisted via the Store.
    - Secrets (SecretStr fields) are persisted as raw strings (encrypted at rest
      by the Store) and are masked when exposed back to clients.
    """

    def __init__(self, store: Store | None = None):
        self._defaults: dict[str, ProviderSettings] = {}
        self._settings: dict[str, ProviderSettings] = {}
        self._settings_overridden: dict[str, bool] = {}
        self._allow_override: dict[str, bool] = {}
        self._store = store
        # Names reserved for builtins; authors cannot register these.
        self._reserved: set[str] = set()

        # Auto-register builtin providers here (NOT in server.py).
        # End-users will override values (api keys, etc.) via the settings API.
        self.register_builtin("openai", OpenAIProviderSettings())
        self.register_builtin("anthropic", AnthropicProviderSettings())
        self.register_builtin("google", GoogleProviderSettings())
        self.register_builtin("litellm", LiteLLMProviderSettings())

    # ---- registration -----------------------------------------------------

    def register_builtin(self, name: str, default: ProviderSettings) -> None:
        """
        Internal use: register a predefined provider and reserve the name.
        """
        if name in self._defaults:
            logger.warning("Builtin provider '%s' already registered; keeping existing default", name)
            self._reserved.add(name)
            return
        self._reserved.add(name)
        self._register_internal(name, default, allow_override=True)

    def register(self, name: str, default: ProviderSettings, *, allow_override: bool = True) -> None:
        """
        Register a custom provider with its default settings.
        """
        if name in self._reserved:
            raise ValueError(f"Provider '{name}' is predefined; do not register it.")
        if name in self._defaults:
            logger.warning("Provider settings already registered for '%s'; keeping existing default", name)
            return
        self._register_internal(name, default, allow_override)

    def _register_internal(self, name: str, default: ProviderSettings, allow_override: bool) -> None:
        """
        Core logic for adding a provider's settings to the registry.
        """
        base = default.model_copy(deep=True)
        base._override_status = False
        self._defaults[name] = base
        self._settings[name] = base.model_copy(deep=True)
        self._settings_overridden[name] = False
        self._allow_override[name] = allow_override

    # ---- access -----------------------------------------------------------

    def get(self, name: str) -> ProviderSettings | None:
        if name not in self._defaults:
            return None
        cur = self._settings.get(name)
        if cur is not None:
            cur._override_status = bool(self._settings_overridden.get(name, False))
            return cur
        # fall back to default if somehow absent
        base = self._defaults[name].model_copy(deep=True)
        base._override_status = False
        self._settings[name] = base
        self._settings_overridden[name] = False
        return base

    def allow_override(self, name: str) -> bool:
        return bool(self._allow_override.get(name, True))

    def settings_overridden(self, name: str) -> bool:
        return bool(self._settings_overridden.get(name, False))

    def list_registered(self) -> list[str]:
        return sorted(self._defaults.keys())

    # ---- persistence ------------------------------------------------------

    async def update(self, name: str, values: dict[str, Any]) -> ProviderSettings:
        """
        Merge/update a provider's settings and persist the effective instance.
        """
        if name not in self._defaults:
            raise ValueError(f"Provider '{name}' is not registered")
        if not self.allow_override(name):
            raise ValueError(f"Provider '{name}' does not allow settings overrides")

        model = self._defaults[name].__class__
        current = self.get(name)
        base_dict = self._to_persistable_dict(current) if current else {}
        locked = getattr(model, "locked", set()) or set()
        for k in values or {}:
            if k in locked and (k in base_dict) and values[k] != base_dict[k]:
                raise ValueError(f"Setting '{k}' is locked and cannot be overridden")

        merged: Dict[str, Any] = {**base_dict, **(values or {})}
        inst = model.model_validate(merged)
        inst._override_status = True
        self._settings[name] = inst
        self._settings_overridden[name] = True

        if self._store is not None:
            try:
                await self._store.upsert_provider_settings(
                    name,
                    self._to_persistable_dict(inst),
                    overridden=True,
                )
            except Exception:
                logger.exception("Failed to persist provider settings for '%s'", name)

        return inst

    async def load_persisted_if_needed(self, name: str) -> None:
        """
        Hydrate in-memory cache for a provider from the store, if not already
        overridden in memory.
        """
        if self._store is None:
            return
        if name not in self._defaults:
            return
        if self._settings_overridden.get(name, False):
            return

        try:
            row = await self._store.get_provider_settings(name)
        except Exception:
            logger.exception("Failed to load persisted provider settings for '%s'", name)
            return

        if not row:
            return
        values, overridden = row
        model = self._defaults[name].__class__
        try:
            inst = model.model_validate(values)
            inst._override_status = bool(overridden)
            self._settings[name] = inst
            self._settings_overridden[name] = bool(overridden)
        except Exception:
            logger.exception("Persisted provider settings invalid for '%s'; ignoring", name)

    async def hydrate_registered(self) -> None:
        """
        Best-effort bulk hydration of all registered providers from the backing Store.

        Intended for server startup: loads persisted overrides for any providers that
        have been registered but not yet explicitly overridden in memory.
        """
        if self._store is None:
            return
        try:
            rows = await self._store.list_provider_settings()
        except Exception:
            logger.exception("Failed to list persisted provider settings")
            return

        for name, (values, overridden) in rows.items():
            # Only hydrate providers that have a registered default.
            if name not in self._defaults:
                continue
            # Do not stomp on explicit in-process overrides.
            if self._settings_overridden.get(name, False):
                continue

            model = self._defaults[name].__class__
            try:
                inst = model.model_validate(values)
                inst._override_status = bool(overridden)
                self._settings[name] = inst
                self._settings_overridden[name] = bool(overridden)
            except Exception:
                logger.exception("Persisted provider settings invalid for '%s'; ignoring", name)

    # ---- schema & masking helpers ----------------------------------------

    def schema_for(self, name: str) -> dict[str, Any] | None:
        default = self._defaults.get(name)
        if default is None:
            return None
        return default.__class__.model_json_schema(by_alias=True)

    def mask_values(self, inst: ProviderSettings) -> tuple[dict[str, Any], dict[str, dict[str, bool]]]:
        """
        Build a dict of values suitable for returning to clients, plus a
        parallel 'secrets' map describing which fields are secret and whether
        they currently have a value.

        The returned values dict keeps the same keys as the underlying settings
        model; secret fields are represented as masked strings.
        """
        values: Dict[str, Any] = {}
        secrets: Dict[str, dict[str, bool]] = {}

        # Use model_dump so extra fields (extra="allow") are preserved.
        # We still handle SecretStr values explicitly so they are masked.
        for field_name, value in inst.model_dump().items():
            if isinstance(value, SecretStr):
                raw = value.get_secret_value()
                has_val = bool(raw)
                masked = self._mask_secret(raw) if has_val else ""
                values[field_name] = masked
                secrets[field_name] = {"hasValue": has_val}
            else:
                values[field_name] = value

        return values, secrets

    @staticmethod
    def _mask_secret(raw: str) -> str:
        if not raw:
            return ""
        # keep a short prefix/suffix, mask the middle
        if len(raw) <= 8:
            return "*" * len(raw)
        prefix = raw[:6]
        suffix = raw[-4:]
        middle_len = max(len(raw) - len(prefix) - len(suffix), 4)
        return f"{prefix}{'*' * middle_len}{suffix}"

    @staticmethod
    def _to_persistable_dict(inst: ProviderSettings | None) -> dict[str, Any]:
        """
        Convert a ProviderSettings instance to a plain dict with raw str values
        for secrets, suitable for JSON serialization and encryption-at-rest.
        """
        if inst is None:
            return {}
        data: Dict[str, Any] = {}
        # model_dump preserves extra fields and respects field exclusions
        for field_name, value in inst.model_dump().items():
            if isinstance(value, SecretStr):
                data[field_name] = value.get_secret_value()
            else:
                data[field_name] = value
        return data
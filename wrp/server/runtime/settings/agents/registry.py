# wrp/server/runtime/settings/agents/registry.py
from __future__ import annotations

from typing import Any, Dict

from mcp.server.fastmcp.utilities.logging import get_logger

from wrp.server.runtime.store.base import Store
from .settings import AgentSettings

logger = get_logger(__name__)


class AgentSettingsRegistry:
    """
    Registry + persistence bridge for agent settings.

    - Authors register defaults once at startup.
    - Overrides are merged/validated and persisted via the Store.
    - Agent settings are plain data (no secrets); they are not masked.
    """

    def __init__(self, store: Store | None = None):
        self._defaults: dict[str, AgentSettings] = {}
        self._settings: dict[str, AgentSettings] = {}
        self._settings_overridden: dict[str, bool] = {}
        self._allow_override: dict[str, bool] = {}
        self._store = store

    # ---- registration -----------------------------------------------------

    def register(self, name: str, default: AgentSettings, *, allow_override: bool = True) -> None:
        if name in self._defaults:
            logger.warning("Agent settings already registered for '%s'; keeping existing default", name)
            return
        base = default.model_copy(deep=True)
        base._override_status = False
        self._defaults[name] = base
        self._settings[name] = base.model_copy(deep=True)
        self._settings_overridden[name] = False
        self._allow_override[name] = allow_override

    # ---- access -----------------------------------------------------------

    def get(self, name: str) -> AgentSettings | None:
        if name not in self._defaults:
            return None
        cur = self._settings.get(name)
        if cur is not None:
            cur._override_status = bool(self._settings_overridden.get(name, False))
            return cur
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

    async def update(self, name: str, values: dict[str, Any]) -> AgentSettings:
        if name not in self._defaults:
            raise ValueError(f"Agent '{name}' is not registered")
        if not self.allow_override(name):
            raise ValueError(f"Agent '{name}' does not allow settings overrides")

        model = self._defaults[name].__class__
        current = self.get(name)
        base_dict: Dict[str, Any] = current.model_dump() if current else {}
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
                await self._store.upsert_agent_settings(
                    name,
                    inst.model_dump(),
                    overridden=True,
                )
            except Exception:
                logger.exception("Failed to persist agent settings for '%s'", name)

        return inst

    async def load_persisted_if_needed(self, name: str) -> None:
        if self._store is None:
            return
        if name not in self._defaults:
            return
        if self._settings_overridden.get(name, False):
            return

        try:
            row = await self._store.get_agent_settings(name)
        except Exception:
            logger.exception("Failed to load persisted agent settings for '%s'", name)
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
            logger.exception("Persisted agent settings invalid for '%s'; ignoring", name)

    async def hydrate_registered(self) -> None:
        """
        Best-effort bulk hydration of all registered agents from the backing Store.

        Intended for server startup: loads persisted overrides for any agents that
        have been registered but not yet explicitly overridden in memory.
        """
        if self._store is None:
            return
        try:
            rows = await self._store.list_agent_settings()
        except Exception:
            logger.exception("Failed to list persisted agent settings")
            return

        for name, (values, overridden) in rows.items():
            # Only hydrate agents that have a registered default.
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
                logger.exception("Persisted agent settings invalid for '%s'; ignoring", name)

    # ---- schema helper ----------------------------------------------------

    def schema_for(self, name: str) -> dict[str, Any] | None:
        default = self._defaults.get(name)
        if default is None:
            return None
        return default.__class__.model_json_schema(by_alias=True)
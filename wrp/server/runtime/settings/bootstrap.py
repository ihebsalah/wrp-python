# wrp/server/runtime/settings/bootstrap.py
from __future__ import annotations

from typing import TYPE_CHECKING

from mcp.server.fastmcp.utilities.logging import get_logger

if TYPE_CHECKING:
    from wrp.server.runtime.server import WRP

logger = get_logger(__name__)


async def hydrate_provider_and_agent_settings(server: "WRP") -> None:
    """
    Best-effort settings bootstrap.

    Loads persisted overrides for all registered providers and agents into their
    registries so ctx.get_provider_settings / ctx.get_agent_settings see the
    effective values immediately after server startup.

    Safe to call multiple times; in-memory overrides always win.
    """
    try:
        await server._provider_settings_registry.hydrate_registered()
    except Exception:
        logger.exception("Failed to hydrate provider settings from store")

    try:
        await server._agent_settings_registry.hydrate_registered()
    except Exception:
        logger.exception("Failed to hydrate agent settings from store")

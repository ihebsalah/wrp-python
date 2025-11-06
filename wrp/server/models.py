# wrp/server/models.py
"""
This module provides simpler types to use with the server for managing workflows.
"""

from pydantic import BaseModel

from wrp.types import Icon, ServerCapabilities


class InitializationOptions(BaseModel):
    server_name: str
    server_version: str
    capabilities: ServerCapabilities
    instructions: str | None = None
    website_url: str | None = None
    icons: list[Icon] | None = None

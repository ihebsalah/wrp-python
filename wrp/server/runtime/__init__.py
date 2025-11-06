# wrp/server/runtime/__init__.py

from importlib.metadata import version

from mcp.server.fastmcp.utilities.types import Audio, Image

from wrp.types import Icon
from wrp.server import Context, WRP

__version__ = version("wrp")
__all__ = ["WRP", "Context", "Image", "Audio", "Icon"]

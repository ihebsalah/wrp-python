# wrp/shared/_httpx_utils.py
"""Utilities for creating standardized httpx AsyncClient instances."""

from contextlib import asynccontextmanager
from typing import Any, AsyncContextManager, AsyncGenerator, Protocol

import httpx

__all__ = ["create_wrp_http_client"]


class WrpHttpClientFactory(Protocol):
    def __call__(
        self,
        headers: dict[str, str] | None = None,
        timeout: httpx.Timeout | None = None,
        auth: httpx.Auth | None = None,
    ) -> AsyncContextManager[httpx.AsyncClient]: ...


@asynccontextmanager
async def create_wrp_http_client(
    headers: dict[str, str] | None = None,
    timeout: httpx.Timeout | None = None,
    auth: httpx.Auth | None = None,
) -> AsyncGenerator[httpx.AsyncClient, None]:
    """Provide a standardized httpx AsyncClient as an async context manager.

    This function provides common defaults used throughout the WRP codebase:
    - follow_redirects=True (always enabled)
    - Default timeout of 30 seconds if not specified

    The function returns an async context manager, which ensures that the client
    connection is properly closed after use.

    Args:
        headers: Optional headers to include with all requests.
        timeout: Request timeout as an httpx.Timeout object.
            Defaults to 30 seconds if not specified.
        auth: Optional authentication handler.

    Yields:
        A configured httpx.AsyncClient instance.

    Examples:
        # Basic usage with WRP defaults
        async with create_wrp_http_client() as client:
            response = await client.get("https://api.example.com")

        # With custom headers
        headers = {"Authorization": "Bearer token"}
        async with create_wrp_http_client(headers=headers) as client:
            response = await client.get("/endpoint")

        # With both custom headers and timeout
        timeout = httpx.Timeout(60.0, read=300.0)
        async with create_wrp_http_client(headers=headers, timeout=timeout) as client:
            response = await client.get("/long-request")

        # With authentication
        from httpx import BasicAuth
        auth = BasicAuth(username="user", password="pass")
        async with create_wrp_http_client(auth=auth) as client:
            response = await client.get("/protected-endpoint")
    """
    # Set WRP defaults
    kwargs: dict[str, Any] = {
        "follow_redirects": True,
    }

    # Handle timeout
    if timeout is None:
        kwargs["timeout"] = httpx.Timeout(30.0)
    else:
        kwargs["timeout"] = timeout

    # Handle headers
    if headers is not None:
        kwargs["headers"] = headers

    # Handle authentication
    if auth is not None:
        kwargs["auth"] = auth

    client = httpx.AsyncClient(**kwargs)
    try:
        yield client
    finally:
        await client.aclose()
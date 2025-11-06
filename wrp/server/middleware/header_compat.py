# wrp/server/middleware/header_compat.py
"""
HeaderCompatMiddleware
- Expose WRP-only headers at the edge.
- Translate to the internal header names expected by the underlying transport.
- Strip internal (MCP) headers from responses so clients see only WRP.
- Optionally, enforce a specific set of supported WRP protocol versions.

You pass `internal_protocol_value` — the value your internal transport accepts for
its protocol-version header. This keeps version checks purely internal.

You can also pass `wrp_supported_versions` to reject requests with unsupported
WRP protocol versions at the edge.
"""

from typing import Awaitable, Callable, Dict, Iterable, Tuple

ASGIApp = Callable[[dict, Callable[[], Awaitable[dict]], Callable[[dict], Awaitable[None]]], Awaitable[None]]

# External WRP header names
_WRP_PROTOCOL_HEADER = b"wrp-protocol-version"
_WRP_SESSION_HEADER = b"wrp-session-id"

# Internal (expected by the reused transport)
_MCP_PROTOCOL_HEADER = b"mcp-protocol-version"
_MCP_SESSION_HEADER = b"mcp-session-id"

def _hdrs_to_dict(headers: Iterable[Tuple[bytes, bytes]]) -> Dict[bytes, bytes]:
    d: Dict[bytes, bytes] = {}
    for k, v in headers:
        d[k.lower()] = v
    return d

class HeaderCompatMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        *,
        internal_protocol_value: str,
        wrp_supported_versions: Iterable[str] | None = None,
        require_wrp_header: bool = False,
    ) -> None:
        self.app = app
        self.internal_protocol_value = internal_protocol_value.encode()
        # A set of allowed WRP versions for quick lookups. If empty, all are allowed.
        self._wrp_supported = set(wrp_supported_versions or [])
        # When True and a session is present, the client must send wrp-protocol-version
        self.require_wrp_header = require_wrp_header

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        in_headers = list(scope.get("headers") or [])
        in_map = _hdrs_to_dict(in_headers)

        # --- Optional: require WRP header after init (i.e., when we see a session) ---
        if self._wrp_supported and self.require_wrp_header:
            if (in_map.get(_WRP_SESSION_HEADER) is not None) and (in_map.get(_WRP_PROTOCOL_HEADER) is None):
                start = {
                    "type": "http.response.start",
                    "status": 400,
                    "headers": [(b"content-type", b"application/json")],
                }
                body = {
                    "type": "http.response.body",
                    "body": b'{"error":"Missing WRP protocol version header"}',
                }
                await send(start); await send(body)
                return

        # --- Optional: enforce WRP protocol versions at the edge ---
        if self._wrp_supported:
            wrp_ver = in_map.get(_WRP_PROTOCOL_HEADER)
            if wrp_ver is not None and wrp_ver.decode() not in self._wrp_supported:
                # Hard fail early with a clear JSON error
                start = {
                    "type": "http.response.start",
                    "status": 400,
                    "headers": [(b"content-type", b"application/json")],
                }
                body = {
                    "type": "http.response.body",
                    "body": b'{"error":"Unsupported WRP protocol version"}',
                }
                await send(start); await send(body)
                return

        # -------- INCOMING: Translate WRP to internal headers --------

        # Session id: copy WRP → internal if present
        if _WRP_SESSION_HEADER in in_map and _MCP_SESSION_HEADER not in in_map:
            in_headers.append((_MCP_SESSION_HEADER, in_map[_WRP_SESSION_HEADER]))

        # Protocol: always override any inbound mcp-protocol-version.
        # We remove any existing internal protocol header and then add our own
        # known-good value. This decouples WRP versioning from the internal
        # transport’s version check and prevents a misconfigured client from
        # causing an internal protocol mismatch error.
        in_headers = [(k, v) for (k, v) in in_headers if k.lower() != _MCP_PROTOCOL_HEADER]
        in_headers.append((_MCP_PROTOCOL_HEADER, self.internal_protocol_value))

        new_scope = dict(scope)
        new_scope["headers"] = in_headers

        # -------- OUTGOING: strip internal headers; add WRP headers --------
        async def send_wrapper(message):
            if message.get("type") == "http.response.start":
                out_headers = list(message.get("headers") or [])
                out_map = _hdrs_to_dict(out_headers)

                # If internal session id is present, mirror to WRP
                if _MCP_SESSION_HEADER in out_map:
                    wrp_val = out_map[_MCP_SESSION_HEADER]
                    # Remove the internal header so clients never see it
                    out_headers = [(k, v) for (k, v) in out_headers if k.lower() != _MCP_SESSION_HEADER]
                    # Ensure WRP session header exists, but don't overwrite if present
                    if _WRP_SESSION_HEADER not in _hdrs_to_dict(out_headers):
                        out_headers.append((_WRP_SESSION_HEADER, wrp_val))

                # Remove the internal protocol header if present
                out_headers = [(k, v) for (k, v) in out_headers if k.lower() != _MCP_PROTOCOL_HEADER]

                msg = dict(message)
                msg["headers"] = out_headers
                await send(msg)
            else:
                await send(message)

        await self.app(new_scope, receive, send_wrapper)
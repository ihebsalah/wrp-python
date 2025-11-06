# wrp/server/runtime/limits.py
"""Runtime defaults and limits."""

# Default per-request workflow input cap (bytes).
# Acts as a safety net for all servers unless explicitly overridden.
DEFAULT_GLOBAL_INPUT_LIMIT_BYTES = 25 * 1024 * 1024  # 2 MiB

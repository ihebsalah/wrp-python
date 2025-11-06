# wrp/server/runtime/store/ops/health.py
from __future__ import annotations

from ..engines.engine import Engine


def basic_health_check(engine: Engine) -> dict:
    # minimal sanity queries
    out = {"ok": True, "details": {}}
    try:
        engine.query_one("SELECT 1;")
        row = engine.query_one("SELECT version FROM schema_version LIMIT 1;")
        out["details"]["schema_version"] = row["version"] if row else None
    except Exception as e:
        out["ok"] = False
        out["error"] = str(e)
    return out

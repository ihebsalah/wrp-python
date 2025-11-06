# wrp/server/runtime/store/ops/migrations.py
from __future__ import annotations

from pathlib import Path

from ..engines.engine import Engine


_SQLITE_INIT = (Path(__file__).parent.parent / "sql" / "sqlite" / "schema.sql").read_text()
_PG_INIT = (Path(__file__).parent.parent / "sql" / "postgres" / "schema.sql").read_text()


def apply_initial_schema_sqlite(engine: Engine) -> None:
    # idempotent: guard by schema_version existence
    engine.execute(
        "CREATE TABLE IF NOT EXISTS schema_version(version INTEGER NOT NULL);"
    )
    row = engine.query_one("SELECT version FROM schema_version LIMIT 1;")
    if row is None:
        with engine.transaction():
            for stmt in _SQLITE_INIT.split(";"):
                s = stmt.strip()
                if s:
                    engine.execute(s + ";")
            engine.execute("INSERT INTO schema_version(version) VALUES (1);")


def apply_initial_schema_postgres(engine: Engine) -> None:
    engine.execute(
        "CREATE TABLE IF NOT EXISTS schema_version(version INTEGER NOT NULL);"
    )
    row = engine.query_one("SELECT version FROM schema_version LIMIT 1;")
    if row is None:
        with engine.transaction():
            engine.execute(_PG_INIT)
            engine.execute("INSERT INTO schema_version(version) VALUES (1);")

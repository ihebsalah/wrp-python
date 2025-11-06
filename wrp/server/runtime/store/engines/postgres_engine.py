# wrp/server/runtime/store/engines/postgres_engine.py
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterable, Mapping, Optional

import psycopg
from psycopg.rows import dict_row

from .engine import Engine


class PostgresEngine(Engine):
    """
    Thin psycopg3 engine with connection pool.
    DSN examples:
      - "postgresql://user:pass@localhost:5432/wrp"
    """

    def __init__(self, dsn: str, *, min_size: int = 1, max_size: int = 10, statement_timeout_ms: int = 10000):
        self.dsn = dsn
        self.min_size = min_size
        self.max_size = max_size
        self.statement_timeout_ms = statement_timeout_ms
        self.pool: Optional[psycopg.Connection] = None
        self.paramstyle = "pyformat"  # we'll use %s placeholders

    def connect(self) -> None:
        if self.pool:
            return
        # simple pooled connection (psycopg "connection as pool" with .cursor())
        self.pool = psycopg.connect(self.dsn, row_factory=dict_row)
        with self.pool.cursor() as cur:
            cur.execute(f"SET statement_timeout = {self.statement_timeout_ms};")

    def close(self) -> None:
        if self.pool:
            self.pool.close()
            self.pool = None

    @contextmanager
    def transaction(self):
        assert self.pool is not None, "Engine not connected"
        with self.pool.transaction():
            yield

    def execute(self, sql: str, params: Iterable[Any] | Mapping[str, Any] | None = None) -> None:
        assert self.pool is not None, "Engine not connected"
        with self.pool.cursor() as cur:
            cur.execute(self.render(sql), [] if params is None else params)

    def query_one(self, sql: str, params: Iterable[Any] | Mapping[str, Any] | None = None) -> dict | None:
        assert self.pool is not None, "Engine not connected"
        with self.pool.cursor() as cur:
            cur.execute(self.render(sql), [] if params is None else params)
            row = cur.fetchone()
            return dict(row) if row else None

    def query_all(self, sql: str, params: Iterable[Any] | Mapping[str, Any] | None = None) -> list[dict]:
        assert self.pool is not None, "Engine not connected"
        with self.pool.cursor() as cur:
            cur.execute(self.render(sql), [] if params is None else params)
            return [dict(r) for r in cur.fetchall()]

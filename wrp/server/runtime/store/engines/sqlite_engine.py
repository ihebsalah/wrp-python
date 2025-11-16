# wrp/server/runtime/store/engines/sqlite_engine.py
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .engine import Engine


class SqliteEngine(Engine):
    """
    Simple SQLite engine with WAL, busy_timeout, and foreign keys ON.
    """

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._conn: sqlite3.Connection | None = None
        self.paramstyle = "qmark"

    def connect(self) -> None:
        if self._conn:
            return
        conn = sqlite3.connect(self.path, detect_types=sqlite3.PARSE_DECLTYPES)
        conn.row_factory = sqlite3.Row
        # Pragmas tuned for local app workloads
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("PRAGMA busy_timeout=5000;")
        self._conn = conn

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    @contextmanager
    def transaction(self):
        assert self._conn is not None, "Engine not connected"
        try:
            self._conn.execute("BEGIN;")
            yield
            self._conn.execute("COMMIT;")
        except Exception:
            self._conn.execute("ROLLBACK;")
            raise

    def execute(self, sql: str, params: Iterable[Any] | Mapping[str, Any] | None = None) -> None:
        assert self._conn is not None, "Engine not connected"
        if params is None:
            adapted_params: Sequence[Any] | Mapping[str, Any] = []
        elif isinstance(params, Mapping):
            adapted_params = params
        else:
            adapted_params = list(params)

        self._conn.execute(self.render(sql), adapted_params)

    def query_one(self, sql: str, params: Iterable[Any] | Mapping[str, Any] | None = None) -> dict | None:
        assert self._conn is not None, "Engine not connected"
        if params is None:
            adapted_params: Sequence[Any] | Mapping[str, Any] = []
        elif isinstance(params, Mapping):
            adapted_params = params
        else:
            adapted_params = list(params)

        cur = self._conn.execute(self.render(sql), adapted_params)
        row = cur.fetchone()
        return dict(row) if row else None

    def query_all(self, sql: str, params: Iterable[Any] | Mapping[str, Any] | None = None) -> list[dict]:
        assert self._conn is not None, "Engine not connected"
        if params is None:
            adapted_params: Sequence[Any] | Mapping[str, Any] = []
        elif isinstance(params, Mapping):
            adapted_params = params
        else:
            adapted_params = list(params)

        cur = self._conn.execute(self.render(sql), adapted_params)
        return [dict(r) for r in cur.fetchall()]
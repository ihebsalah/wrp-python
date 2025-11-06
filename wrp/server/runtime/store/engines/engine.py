# wrp/server/runtime/store/engines/engine.py
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterable, Mapping, Protocol


class Row(Mapping[str, Any], Protocol):  # structural typing
    ...


class Engine(Protocol):
    """
    Minimal DB engine protocol used by DAOs.
    Implementations can be sync; stores can optionally run them in a thread.
    """

    paramstyle: str  # "qmark" for SQLite, "pyformat" for Postgres

    def connect(self) -> None: ...
    def close(self) -> None: ...

    @contextmanager
    def transaction(self):
        """BEGIN...COMMIT/ROLLBACK context."""
        yield

    def execute(self, sql: str, params: Iterable[Any] | Mapping[str, Any] | None = None) -> None: ...
    def query_one(self, sql: str, params: Iterable[Any] | Mapping[str, Any] | None = None) -> dict | None: ...
    def query_all(self, sql: str, params: Iterable[Any] | Mapping[str, Any] | None = None) -> list[dict]: ...

    # Utilities -------------------------------------------------------------

    def render(self, sql: str) -> str:
        """
        DAOs write SQL using %s placeholders (Postgres style).
        - For Postgres, we pass through.
        - For SQLite, convert %s -> ?.
        """
        if self.paramstyle == "qmark":
            return sql.replace("%s", "?")
        return sql

    def placeholders(self, n: int) -> str:
        """
        Return '(?, ?, ...)' or '(%s, %s, ...)' depending on paramstyle.
        """
        token = "?" if self.paramstyle == "qmark" else "%s"
        return "(" + ", ".join([token] * n) + ")"

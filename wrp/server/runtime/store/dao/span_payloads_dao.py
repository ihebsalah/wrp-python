# wrp/server/runtime/store/dao/span_payloads_dao.py
from __future__ import annotations

from typing import Any

from ..engines.engine import Engine


class SpanPayloadsDAO:
    def __init__(self, engine: Engine):
        self.e = engine

    def get(self, run_id: str, span_id: str) -> dict | None:
        return self.e.query_one(
            "SELECT envelope_blob, updated_at FROM span_payloads WHERE run_id=%s AND span_id=%s;",
            (run_id, span_id),
        )

    def upsert(self, run_id: str, span_id: str, envelope_blob: bytes, updated_at: str) -> None:
        # Use standard UPSERT patterns for both engines
        sql = """
        INSERT INTO span_payloads(run_id, span_id, envelope_blob, updated_at)
        VALUES (%s,%s,%s,%s)
        ON CONFLICT(run_id, span_id) DO UPDATE SET envelope_blob=EXCLUDED.envelope_blob, updated_at=EXCLUDED.updated_at;
        """
        if self.e.paramstyle == "pyformat":
            # Postgres uses DO UPDATE syntax (already fine)
            self.e.execute(sql, (run_id, span_id, envelope_blob, updated_at))
        else:
            # SQLite has the same ON CONFLICT syntax (since 3.24+)
            self.e.execute(sql, (run_id, span_id, envelope_blob, updated_at))

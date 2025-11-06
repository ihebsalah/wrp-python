# wrp/server/runtime/store/dao/runs_dao.py
from __future__ import annotations

from typing import Any

from ..engines.engine import Engine


class RunsDAO:
    def __init__(self, engine: Engine):
        self.e = engine

    # schema_version managed by migrations

    def ensure_counters_row(self) -> None:
        self.e.execute(
            """
            INSERT INTO counters(key, value)
            SELECT 'run_id', 0
            WHERE NOT EXISTS (SELECT 1 FROM counters WHERE key='run_id');
            """
        )

    def alloc_run_id(self) -> int:
        """
        Global 001..999 counter (sessions later). Raises at >999.
        """
        self.ensure_counters_row()
        with self.e.transaction():
            row = self.e.query_one("SELECT value FROM counters WHERE key='run_id';")
            cur = int(row["value"]) if row else 0
            nxt = cur + 1
            if nxt > 999:
                raise ValueError("Run ID capacity reached (001..999). Please rotate or archive old runs.")
            self.e.execute("UPDATE counters SET value=%s WHERE key='run_id';", (nxt,))
        return nxt

    # CRUD -----------------------------------------------------------------

    def insert(self, meta: dict[str, Any]) -> None:
        self.e.execute(
            """
            INSERT INTO runs(run_id, workflow_name, thread_id, created_at, state,
                             message_count, channel_counts_json, outcome, error_text, run_output_blob, updated_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,NULL,NULL,NULL,%s);
            """,
            (
                meta["run_id"],
                meta["workflow_name"],
                meta["thread_id"],
                meta["created_at"],
                meta["state"],
                meta["message_count"],
                meta["channel_counts_json"],
                meta["updated_at"],
            ),
        )

    def update_conclude(self, run_id: str, outcome: str, error_text: str | None, run_output_blob: bytes | None, updated_at: str) -> None:
        self.e.execute(
            """
            UPDATE runs
               SET state='concluded', outcome=%s, error_text=%s, run_output_blob=%s, updated_at=%s
             WHERE run_id=%s;
            """,
            (outcome, error_text, run_output_blob, updated_at, run_id),
        )

    def get(self, run_id: str) -> dict | None:
        return self.e.query_one("SELECT * FROM runs WHERE run_id=%s;", (run_id,))

    def list_by_thread(self, workflow_name: str, thread_id: str) -> list[dict]:
        return self.e.query_all(
            """
            SELECT * FROM runs
             WHERE workflow_name=%s AND thread_id=%s
             ORDER BY created_at ASC;
            """,
            (workflow_name, thread_id),
        )

    # counters --------------------------------------------------------------

    def bump_counts(self, run_id: str, message_delta: int, channel_counts_json: str, updated_at: str) -> None:
        self.e.execute(
            """
            UPDATE runs
               SET message_count = message_count + %s,
                   channel_counts_json = %s,
                   updated_at=%s
             WHERE run_id=%s;
            """,
            (message_delta, channel_counts_json, updated_at, run_id),
        )

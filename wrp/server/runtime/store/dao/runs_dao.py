# wrp/server/runtime/store/dao/runs_dao.py
from __future__ import annotations

from typing import Any

from ..engines.engine import Engine


class RunsDAO:
    def __init__(self, engine: Engine):
        self.e = engine

    # schema_version managed by migrations

    def ensure_counters_row(self, system_session_id: str) -> None:
        self.e.execute(
            """
            INSERT INTO counters(system_session_id, key, value)
            SELECT %s, 'run_id', 0
            WHERE NOT EXISTS (
              SELECT 1 FROM counters WHERE system_session_id=%s AND key='run_id'
            );
            """,
            (system_session_id, system_session_id),
        )

    def alloc_run_id(self, system_session_id: str) -> int:
        """
        Per-session 001..999 counter. Raises at >999.
        """
        self.ensure_counters_row(system_session_id)
        with self.e.transaction():
            row = self.e.query_one(
                "SELECT value FROM counters WHERE system_session_id=%s AND key='run_id';",
                (system_session_id,),
            )
            cur = int(row["value"]) if row else 0
            nxt = cur + 1
            if nxt > 999:
                raise ValueError("Run ID capacity reached (001..999). Please rotate or archive old runs.")
            self.e.execute(
                "UPDATE counters SET value=%s WHERE system_session_id=%s AND key='run_id';",
                (nxt, system_session_id),
            )
        return nxt

    # CRUD -----------------------------------------------------------------

    def insert(self, meta: dict[str, Any]) -> None:
        self.e.execute(
            """
            INSERT INTO runs(system_session_id, run_id, workflow_name, thread_id, created_at, state,
                             outcome, error_text, run_output_blob, updated_at)
            VALUES (%s,%s,%s,%s,%s,%s,NULL,NULL,NULL,%s);
            """,
            (
                meta["system_session_id"],
                meta["run_id"],
                meta["workflow_name"],
                meta["thread_id"],
                meta["created_at"],
                meta["state"],
                meta["updated_at"],
            ),
        )

    def update_conclude(
        self,
        system_session_id: str,
        run_id: str,
        outcome: str,
        error_text: str | None,
        run_output_blob: bytes | None,
        updated_at: str,
    ) -> None:
        self.e.execute(
            """
            UPDATE runs
               SET state='concluded', outcome=%s, error_text=%s, run_output_blob=%s, updated_at=%s
             WHERE system_session_id=%s AND run_id=%s;
            """,
            (outcome, error_text, run_output_blob, updated_at, system_session_id, run_id),
        )

    def get(self, system_session_id: str, run_id: str) -> dict | None:
        return self.e.query_one(
            "SELECT * FROM runs WHERE system_session_id=%s AND run_id=%s;",
            (system_session_id, run_id),
        )

    def list_by_thread(self, system_session_id: str, workflow_name: str, thread_id: str) -> list[dict]:
        return self.e.query_all(
            """
            SELECT * FROM runs
             WHERE system_session_id=%s AND workflow_name=%s AND thread_id=%s
             ORDER BY created_at ASC;
            """,
            (system_session_id, workflow_name, thread_id),
        )

    def list_runs(
        self,
        system_session_id: str,
        *,
        workflow_name: str | None = None,
        thread_id: str | None = None,
        state: str | None = None,
        outcome: str | None = None,
    ) -> list[dict]:
        clauses = ["system_session_id=%s"]
        params: list[Any] = [system_session_id]
        if workflow_name is not None:
            clauses.append("workflow_name=%s")
            params.append(workflow_name)
        if thread_id is not None:
            clauses.append("thread_id=%s")
            params.append(thread_id)
        if state is not None:
            clauses.append("state=%s")
            params.append(state)
        if outcome is not None:
            clauses.append("outcome=%s")
            params.append(outcome)
        where = " AND ".join(clauses)
        query = f"SELECT * FROM runs WHERE {where} ORDER BY created_at ASC;"
        return self.e.query_all(query, tuple(params))

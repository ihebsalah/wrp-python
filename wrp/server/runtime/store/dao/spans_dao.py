# wrp/server/runtime/store/dao/spans_dao.py
from __future__ import annotations

from typing import Any, Iterable, Optional, Set

from ..engines.engine import Engine


class SpansDAO:
    def __init__(self, engine: Engine):
        self.e = engine

    def append_many(
        self, system_session_id: str, run_id: str, rows: Iterable[dict[str, Any]]
    ) -> None:
        rows = list(rows)
        if not rows:
            return
        placeholders = self.e.placeholders(5)
        sql = f"""
        INSERT INTO telemetry_events(system_session_id, run_id, ts, kind, payload_blob)
        VALUES {", ".join([placeholders] * len(rows))};
        """
        params: list[Any] = []
        for r in rows:
            params.extend(
                [system_session_id, run_id, r["ts"], r["kind"], r["payload_blob"]]
            )
        self.e.execute(sql, params)

    def load(
        self,
        system_session_id: str,
        run_id: str,
        kinds: Optional[Set[str]],
        limit: Optional[int],
    ) -> list[dict]:
        """Load telemetry events for a given run.

        Args:
            system_session_id: The ID of the system session.
            run_id: The ID of the run to load events for.
            kinds: An optional set of event kinds to filter by.
            limit: An optional number of events to return. If specified, the latest
                N events will be returned.

        Returns:
            A list of telemetry events, sorted by timestamp in ascending order.
        """
        if kinds:
            placeholders = self.e.placeholders(len(kinds))
            if limit:
                # latest N, return ASC
                rows = self.e.query_all(
                    f"""
                    SELECT ts, kind, payload_blob FROM telemetry_events
                     WHERE system_session_id=%s AND run_id=%s AND kind IN {placeholders}
                     ORDER BY ts DESC
                     LIMIT %s;
                    """,
                    (system_session_id, run_id, *list(kinds), limit),
                )
                return rows[::-1]
            return self.e.query_all(
                f"""
                SELECT ts, kind, payload_blob FROM telemetry_events
                 WHERE system_session_id=%s AND run_id=%s AND kind IN {placeholders}
                 ORDER BY ts ASC;
                """,
                (system_session_id, run_id, *list(kinds)),
            )

        if limit:
            # latest N, return ASC
            rows = self.e.query_all(
                """
                SELECT ts, kind, payload_blob FROM telemetry_events
                 WHERE system_session_id=%s AND run_id=%s
                 ORDER BY ts DESC
                 LIMIT %s;
                """,
                (system_session_id, run_id, limit),
            )
            return rows[::-1]

        return self.e.query_all(
            """
            SELECT ts, kind, payload_blob FROM telemetry_events
             WHERE system_session_id=%s AND run_id=%s
             ORDER BY ts ASC;
            """,
            (system_session_id, run_id),
        )
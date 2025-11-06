# wrp/server/runtime/store/dao/telemetry_dao.py
from __future__ import annotations

from typing import Any, Iterable, Optional, Set

from ..engines.engine import Engine


class TelemetryDAO:
    def __init__(self, engine: Engine):
        self.e = engine

    def append_many(self, run_id: str, rows: Iterable[dict[str, Any]]) -> None:
        rows = list(rows)
        if not rows:
            return
        placeholders = self.e.placeholders(4)
        sql = f"""
        INSERT INTO telemetry_events(run_id, ts, kind, payload_blob)
        VALUES {", ".join([placeholders] * len(rows))};
        """
        params: list[Any] = []
        for r in rows:
            params.extend([run_id, r["ts"], r["kind"], r["payload_blob"]])
        self.e.execute(sql, params)

    def load(self, run_id: str, kinds: Optional[Set[str]], limit: Optional[int]) -> list[dict]:
        """Load telemetry events for a given run.

        Args:
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
                     WHERE run_id=%s AND kind IN {placeholders}
                     ORDER BY ts DESC
                     LIMIT %s;
                    """,
                    (run_id, *list(kinds), limit),
                )
                return rows[::-1]
            return self.e.query_all(
                f"""
                SELECT ts, kind, payload_blob FROM telemetry_events
                 WHERE run_id=%s AND kind IN {placeholders}
                 ORDER BY ts ASC;
                """,
                (run_id, *list(kinds)),
            )

        if limit:
            # latest N, return ASC
            rows = self.e.query_all(
                """
                SELECT ts, kind, payload_blob FROM telemetry_events
                 WHERE run_id=%s
                 ORDER BY ts DESC
                 LIMIT %s;
                """,
                (run_id, limit),
            )
            return rows[::-1]

        return self.e.query_all(
            """
            SELECT ts, kind, payload_blob FROM telemetry_events
             WHERE run_id=%s
             ORDER BY ts ASC;
            """,
            (run_id,),
        )
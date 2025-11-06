# wrp/server/runtime/store/dao/conversations_dao.py
from __future__ import annotations

from typing import Any, Iterable

from ..engines.engine import Engine


class ConversationsDAO:
    def __init__(self, engine: Engine):
        self.e = engine

    def append_many(self, run_id: str, rows: Iterable[dict[str, Any]]) -> None:
        rows = list(rows)
        if not rows:
            return
        placeholders = self.e.placeholders(6)
        sql = f"""
        INSERT INTO conversation_items(run_id, idx, ts, channel, payload_blob, sort_ts)
        VALUES {", ".join([placeholders] * len(rows))};
        """
        params: list[Any] = []
        for r in rows:
            params.extend([run_id, r["idx"], r["ts"], r["channel"], r["payload_blob"], r["ts"]])
        self.e.execute(sql, params)

    def load_all(self, run_id: str) -> list[dict]:
        return self.e.query_all(
            """
            SELECT idx, ts, channel, payload_blob FROM conversation_items
            WHERE run_id=%s ORDER BY ts ASC;
            """,
            (run_id,),
        )

    def load_tail(self, run_id: str, limit: int, channels: set[str] | None) -> list[dict]:
        if channels:
            placeholders = self.e.placeholders(len(channels))
            return self.e.query_all(
                f"""
                SELECT idx, ts, channel, payload_blob
                  FROM conversation_items
                 WHERE run_id=%s AND channel IN {placeholders}
                 ORDER BY ts DESC
                 LIMIT %s;
                """,
                (run_id, *list(channels), limit),
            )[::-1]  # reverse to ASC
        return self.e.query_all(
            """
            SELECT idx, ts, channel, payload_blob
              FROM conversation_items
             WHERE run_id=%s
             ORDER BY ts DESC
             LIMIT %s;
            """,
            (run_id, limit),
        )[::-1]

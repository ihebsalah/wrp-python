# wrp/server/runtime/store/dao/conversations_dao.py
from __future__ import annotations

from typing import Any, Iterable

from ..engines.engine import Engine


class ConversationsDAO:
    def __init__(self, engine: Engine):
        self.e = engine

    def append_many(self, system_session_id: str, run_id: str, rows: Iterable[dict[str, Any]]) -> None:
        rows = list(rows)
        if not rows:
            return
        placeholders = self.e.placeholders(7)
        sql = f"""
        INSERT INTO conversation_items(system_session_id, run_id, idx, ts, channel, payload_blob, sort_ts)
        VALUES {", ".join([placeholders] * len(rows))};
        """
        params: list[Any] = []
        for r in rows:
            params.extend([system_session_id, run_id, r["idx"], r["ts"], r["channel"], r["payload_blob"], r["ts"]])
        self.e.execute(sql, params)

    def load_all(self, system_session_id: str, run_id: str) -> list[dict]:
        return self.e.query_all(
            """
            SELECT idx, ts, channel, payload_blob FROM conversation_items
            WHERE system_session_id=%s AND run_id=%s
            ORDER BY sort_ts ASC, idx ASC;
            """,
            (system_session_id, run_id),
        )

    def load_tail(self, system_session_id: str, run_id: str, limit: int, channels: set[str] | None) -> list[dict]:
        if channels:
            placeholders = self.e.placeholders(len(channels))
            return self.e.query_all(
                f"""
                SELECT idx, ts, channel, payload_blob
                  FROM conversation_items
                 WHERE system_session_id=%s AND run_id=%s AND channel IN {placeholders}
                 ORDER BY sort_ts DESC, idx DESC
                 LIMIT %s;
                """,
                (system_session_id, run_id, *list(channels), limit),
            )[::-1]  # reverse to ASC
        return self.e.query_all(
            """
            SELECT idx, ts, channel, payload_blob
              FROM conversation_items
             WHERE system_session_id=%s AND run_id=%s
             ORDER BY sort_ts DESC, idx DESC
             LIMIT %s;
            """,
            (system_session_id, run_id, limit),
        )[::-1]

    # ---- channel meta ----------------------------------------------------
    def upsert_channel_meta(
        self,
        system_session_id: str,
        run_id: str,
        channel: str,
        *,
        add_count: int,
        last_ts: str | None,
        name: str | None = None,
        description: str | None = None,
        item_type: str | None = None,
    ) -> None:
        # Use ON CONFLICT for engines that support it; Engine abstracts param styles.
        placeholders = self.e.placeholders(8)
        sql = f"""
        INSERT INTO conversation_channels(system_session_id, run_id, channel, name, description, items_count, last_ts, item_type)
        VALUES {placeholders}
        ON CONFLICT (system_session_id, run_id, channel)
        DO UPDATE SET
          items_count = conversation_channels.items_count + EXCLUDED.items_count,
          last_ts = CASE
                      WHEN conversation_channels.last_ts IS NULL THEN EXCLUDED.last_ts
                      WHEN EXCLUDED.last_ts IS NULL THEN conversation_channels.last_ts
                      ELSE (CASE WHEN conversation_channels.last_ts >= EXCLUDED.last_ts THEN conversation_channels.last_ts ELSE EXCLUDED.last_ts END)
                    END,
          item_type = COALESCE(conversation_channels.item_type, EXCLUDED.item_type)
          -- NOTE: name/description still not overwritten on bump (first writer wins).
        ;
        """
        self.e.execute(sql, (system_session_id, run_id, channel, name, description, add_count, last_ts, item_type))

    def list_channels_meta(self, system_session_id: str, run_id: str) -> list[dict]:
        return self.e.query_all(
            """
            SELECT channel, name, description, items_count, last_ts, item_type
              FROM conversation_channels
             WHERE system_session_id=%s AND run_id=%s
             ORDER BY channel ASC;
            """,
            (system_session_id, run_id),
        )
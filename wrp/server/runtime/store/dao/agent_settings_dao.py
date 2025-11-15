# wrp/server/runtime/store/dao/agent_settings_dao.py
from __future__ import annotations

from typing import Any

from ..engines.engine import Engine


class AgentSettingsDAO:
    def __init__(self, engine: Engine):
        self.e = engine

    def upsert(self, agent_name: str, values_json: str, overridden: bool, updated_at: str) -> None:
        """
        Insert or update agent settings row.

        values_json is plaintext JSON; agent settings are not considered secret.
        """
        self.e.execute(
            """
            INSERT INTO agent_settings(agent_name, values_json, overridden, updated_at)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT(agent_name) DO UPDATE
               SET values_json = EXCLUDED.values_json,
                   overridden = EXCLUDED.overridden,
                   updated_at = EXCLUDED.updated_at;
            """,
            (agent_name, values_json, int(bool(overridden)), updated_at),
        )

    def get(self, agent_name: str) -> dict | None:
        return self.e.query_one(
            "SELECT agent_name, values_json, overridden, updated_at FROM agent_settings WHERE agent_name=%s;",
            (agent_name,),
        )

    def list_all(self) -> list[dict[str, Any]]:
        return self.e.query_all(
            "SELECT agent_name, values_json, overridden, updated_at FROM agent_settings;",
            (),
        )

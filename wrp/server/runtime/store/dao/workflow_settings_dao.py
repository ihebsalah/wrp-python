# wrp/server/runtime/store/dao/workflow_settings_dao.py
from __future__ import annotations

from typing import Any

from ..engines.engine import Engine


class WorkflowSettingsDAO:
    def __init__(self, engine: Engine):
        self.e = engine

    def upsert(self, workflow_name: str, values_json: str, overridden: bool, updated_at: str) -> None:
        # SQLite uses INSERT...ON CONFLICT; Postgres too via same SQL shape in our engine
        self.e.execute(
            f"""
            INSERT INTO workflow_settings(workflow_name, values_json, overridden, updated_at)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT(workflow_name) DO UPDATE
               SET values_json = EXCLUDED.values_json,
                   overridden = EXCLUDED.overridden,
                   updated_at = EXCLUDED.updated_at;
            """,
            (workflow_name, values_json, int(bool(overridden)), updated_at),
        )

    def get(self, workflow_name: str) -> dict | None:
        return self.e.query_one(
            "SELECT workflow_name, values_json, overridden, updated_at FROM workflow_settings WHERE workflow_name=%s;",
            (workflow_name,),
        )

    def list_all(self) -> list[dict[str, Any]]:
        return self.e.query_all("SELECT workflow_name, values_json, overridden, updated_at FROM workflow_settings;", ())

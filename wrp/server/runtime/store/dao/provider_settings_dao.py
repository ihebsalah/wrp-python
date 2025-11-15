# wrp/server/runtime/store/dao/provider_settings_dao.py
from __future__ import annotations

from typing import Any

from ..engines.engine import Engine


class ProviderSettingsDAO:
    def __init__(self, engine: Engine):
        self.e = engine

    def upsert(self, provider_name: str, values_blob: bytes, overridden: bool, updated_at: str) -> None:
        """
        Insert or update provider settings row.

        values_blob is an encrypted envelope (see envelope_codec).
        """
        self.e.execute(
            """
            INSERT INTO provider_settings(provider_name, values_blob, overridden, updated_at)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT(provider_name) DO UPDATE
               SET values_blob = EXCLUDED.values_blob,
                   overridden = EXCLUDED.overridden,
                   updated_at = EXCLUDED.updated_at;
            """,
            (provider_name, values_blob, int(bool(overridden)), updated_at),
        )

    def get(self, provider_name: str) -> dict | None:
        return self.e.query_one(
            "SELECT provider_name, values_blob, overridden, updated_at FROM provider_settings WHERE provider_name=%s;",
            (provider_name,),
        )

    def list_all(self) -> list[dict[str, Any]]:
        return self.e.query_all(
            "SELECT provider_name, values_blob, overridden, updated_at FROM provider_settings;",
            (),
        )

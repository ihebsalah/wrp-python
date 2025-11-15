# wrp/server/runtime/store/dao/system_sessions_dao.py
from __future__ import annotations
from typing import Any
from ..engines.engine import Engine


class SystemSessionsDAO:
    """
    Data Access Object for managing system sessions in the database.
    """
    def __init__(self, engine: Engine):
        """
        Initializes the DAO with a database engine.

        Args:
            engine: The database engine instance to use for queries.
        """
        self.e = engine

    def upsert(self, system_session_id: str, name: str | None, created_at_iso: str) -> None:
        """
        Inserts a new system session or updates an existing one.

        If a session with the given system_session_id does not exist, it is created.
        If it already exists, its 'name' will be updated if the provided 'name'
        is not NULL. The creation timestamp is only set on the initial insert.

        Args:
            system_session_id: The unique identifier for the system session.
            name: The optional name or label for the session.
            created_at_iso: The ISO 8601 formatted timestamp of creation.
        """
        self.e.execute(
            """
            INSERT INTO system_sessions(system_session_id, name, created_at)
            VALUES (%s, %s, %s)
            ON CONFLICT(system_session_id) DO UPDATE
              SET name=COALESCE(EXCLUDED.name, system_sessions.name);
            """,
            (system_session_id, name, created_at_iso),
        )

    def get(self, system_session_id: str) -> dict | None:
        """
        Retrieves a single system session by its ID.

        Args:
            system_session_id: The ID of the session to retrieve.

        Returns:
            A dictionary representing the session, or None if not found.
        """
        return self.e.query_one(
            "SELECT system_session_id, name, created_at FROM system_sessions WHERE system_session_id=%s;",
            (system_session_id,),
        )

    def list_all(self) -> list[dict[str, Any]]:
        """
        Retrieves all system sessions, ordered by creation date.

        Returns:
            A list of dictionaries, where each dictionary represents a system session.
        """
        return self.e.query_all(
            "SELECT system_session_id, name, created_at FROM system_sessions ORDER BY created_at ASC;",
            (),
        )
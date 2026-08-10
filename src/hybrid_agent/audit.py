"""SQLite-backed, append-only action history."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AuditEvent:
    id: int
    created_at: str
    session_id: str
    event_type: str
    tool_name: str | None
    risk: int | None
    approved: bool | None
    status: str
    details: dict[str, Any]


class AuditStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RuntimeError(f"Could not initialize AuditStore directory: {exc}") from exc
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    tool_name TEXT,
                    risk INTEGER,
                    approved INTEGER,
                    status TEXT NOT NULL,
                    details_json TEXT NOT NULL
                )
                """
            )

    def record(
        self,
        *,
        session_id: str,
        event_type: str,
        status: str,
        details: dict[str, Any],
        tool_name: str | None = None,
        risk: int | None = None,
        approved: bool | None = None,
    ) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO audit_events (
                    created_at, session_id, event_type, tool_name,
                    risk, approved, status, details_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(UTC).isoformat(),
                    session_id,
                    event_type,
                    tool_name,
                    risk,
                    None if approved is None else int(approved),
                    status,
                    json.dumps(details, sort_keys=True, default=str),
                ),
            )
            return int(cursor.lastrowid)

    def recent(self, limit: int = 20) -> list[AuditEvent]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM audit_events ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [
            AuditEvent(
                id=row["id"],
                created_at=row["created_at"],
                session_id=row["session_id"],
                event_type=row["event_type"],
                tool_name=row["tool_name"],
                risk=row["risk"],
                approved=None if row["approved"] is None else bool(row["approved"]),
                status=row["status"],
                details=json.loads(row["details_json"]),
            )
            for row in rows
        ]

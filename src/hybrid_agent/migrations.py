"""Minimal, transactional SQLite schema migration engine."""

import sqlite3
from collections.abc import Callable
from pathlib import Path


def run_migrations(path: Path, migrations: list[Callable[[sqlite3.Connection], None]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.commit()
        current_version_row = connection.execute(
            "SELECT MAX(version) FROM schema_migrations"
        ).fetchone()
        current_version = (
            current_version_row[0]
            if current_version_row and current_version_row[0] is not None
            else -1
        )

        for version, migration in enumerate(migrations):
            if version <= current_version:
                continue
            with connection:
                migration(connection)
                connection.execute(
                    "INSERT INTO schema_migrations (version) VALUES (?)", (version,)
                )

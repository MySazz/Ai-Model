"""User-approved structured memory, separate from operational audit history."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class Memory:
    id: int
    created_at: str
    user_id: str
    workspace_id: str
    category: str
    content: str
    source: str


class MemoryStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    workspace_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS memories_scope ON memories(user_id, workspace_id)"
            )

    def add(
        self,
        content: str,
        *,
        user_id: str = "local",
        workspace_id: str = "default",
        category: str = "fact",
        source: str = "user",
    ) -> int:
        clean = content.strip()
        if not clean:
            raise ValueError("Memory content cannot be empty.")
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO memories (
                    created_at, user_id, workspace_id, category, content, source
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(UTC).isoformat(),
                    user_id,
                    workspace_id,
                    category,
                    clean,
                    source,
                ),
            )
            return int(cursor.lastrowid)

    def search(
        self,
        query: str = "",
        *,
        user_id: str = "local",
        workspace_id: str = "default",
        limit: int = 20,
    ) -> list[Memory]:
        escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        pattern = f"%{escaped}%"
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM memories
                WHERE user_id = ? AND workspace_id = ? AND content LIKE ? ESCAPE '\\'
                ORDER BY id DESC LIMIT ?
                """,
                (user_id, workspace_id, pattern, max(1, limit)),
            ).fetchall()
        return [Memory(**dict(row)) for row in rows]

    def relevant(
        self,
        query: str,
        *,
        user_id: str = "local",
        workspace_id: str = "default",
        limit: int = 5,
    ) -> list[Memory]:
        tokens = {
            token.lower()
            for token in re.findall(r"[A-Za-z0-9_-]{3,}", query)
            if token.lower()
            not in {"and", "are", "for", "from", "that", "the", "this", "with", "you"}
        }
        candidates = self.search(
            "",
            user_id=user_id,
            workspace_id=workspace_id,
            limit=200,
        )
        if not tokens:
            return candidates[:limit]

        scored: list[tuple[int, int, Memory]] = []
        for memory in candidates:
            content_tokens = set(re.findall(r"[A-Za-z0-9_-]{3,}", memory.content.lower()))
            score = len(tokens & content_tokens)
            if score:
                scored.append((score, memory.id, memory))
        scored.sort(reverse=True)
        return [item[2] for item in scored[:limit]]

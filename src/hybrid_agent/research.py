"""Durable research sources with immutable content hashes and line provenance."""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class ResearchSource:
    id: int
    created_at: str
    user_id: str
    workspace_id: str
    title: str
    uri: str
    media_type: str
    sha256: str
    content: str


@dataclass(frozen=True)
class SourceMatch:
    source_id: int
    title: str
    uri: str
    sha256: str
    line_number: int
    excerpt: str


class ResearchStore:
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
                CREATE TABLE IF NOT EXISTS research_sources (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    workspace_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    uri TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    content TEXT NOT NULL,
                    UNIQUE(user_id, workspace_id, sha256)
                )
                """
            )

    def add_text(
        self,
        *,
        title: str,
        uri: str,
        content: str,
        media_type: str = "text/plain",
        user_id: str = "local",
        workspace_id: str = "default",
    ) -> int:
        clean_title = title.strip()
        if not clean_title or not content:
            raise ValueError("Research sources require a title and non-empty content.")
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        with self._connect() as connection:
            existing = connection.execute(
                """
                SELECT id FROM research_sources
                WHERE user_id = ? AND workspace_id = ? AND sha256 = ?
                """,
                (user_id, workspace_id, digest),
            ).fetchone()
            if existing:
                return int(existing["id"])
            cursor = connection.execute(
                """
                INSERT INTO research_sources (
                    created_at, user_id, workspace_id, title, uri,
                    media_type, sha256, content
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(UTC).isoformat(),
                    user_id,
                    workspace_id,
                    clean_title,
                    uri,
                    media_type,
                    digest,
                    content,
                ),
            )
            return int(cursor.lastrowid)

    def add_workspace_file(
        self,
        path: Path,
        *,
        workspace: Path,
        title: str | None = None,
        user_id: str = "local",
        workspace_id: str = "default",
    ) -> int:
        root = workspace.resolve()
        resolved = path.resolve()
        if resolved != root and root not in resolved.parents:
            raise PermissionError("Research source escapes the configured workspace.")
        content = resolved.read_text(encoding="utf-8")
        return self.add_text(
            title=title or resolved.name,
            uri=str(resolved.relative_to(root)),
            content=content,
            user_id=user_id,
            workspace_id=workspace_id,
        )

    def list(
        self,
        *,
        user_id: str = "local",
        workspace_id: str = "default",
        limit: int = 50,
    ) -> list[ResearchSource]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM research_sources
                WHERE user_id = ? AND workspace_id = ?
                ORDER BY id DESC LIMIT ?
                """,
                (user_id, workspace_id, max(1, limit)),
            ).fetchall()
        return [ResearchSource(**dict(row)) for row in rows]

    def search(
        self,
        query: str,
        *,
        user_id: str = "local",
        workspace_id: str = "default",
        limit: int = 20,
    ) -> list[SourceMatch]:
        needle = query.strip().lower()
        if not needle:
            raise ValueError("Research query cannot be empty.")
        matches: list[SourceMatch] = []
        for source in self.list(
            user_id=user_id,
            workspace_id=workspace_id,
            limit=500,
        ):
            for line_number, line in enumerate(source.content.splitlines(), 1):
                if needle in line.lower():
                    matches.append(
                        SourceMatch(
                            source_id=source.id,
                            title=source.title,
                            uri=source.uri,
                            sha256=source.sha256,
                            line_number=line_number,
                            excerpt=line[:500],
                        )
                    )
                    if len(matches) >= limit:
                        return matches
        return matches

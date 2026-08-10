"""User-approved structured memory, separate from operational audit history."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import json
import math
import urllib.request

from .migrations import run_migrations


@dataclass(frozen=True)
class Memory:
    id: int
    created_at: str
    user_id: str
    workspace_id: str
    category: str
    content: str
    source: str
    embedding_json: str | None = None

def _get_embedding(text: str) -> list[float]:
    try:
        req = urllib.request.Request(
            "http://127.0.0.1:11434/api/embeddings",
            data=json.dumps({"model": "nomic-embed-text", "prompt": text}).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=2.0) as response:
            return json.loads(response.read().decode("utf-8"))["embedding"]
    except Exception:
        return []

def _cosine_similarity(v1: list[float], v2: list[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    return dot / (norm1 * norm2) if norm1 and norm2 else 0.0


class MemoryStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RuntimeError(f"Could not initialize MemoryStore directory: {exc}") from exc
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        def migration_0(conn: sqlite3.Connection) -> None:
            conn.execute(
                """
                CREATE TABLE memories (
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
            conn.execute("CREATE INDEX memories_scope ON memories(user_id, workspace_id)")
            
        def migration_1(conn: sqlite3.Connection) -> None:
            conn.execute("ALTER TABLE memories ADD COLUMN embedding_json TEXT")

        run_migrations(self.path, [migration_0, migration_1])

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
        embedding = _get_embedding(clean)
        embedding_json = json.dumps(embedding) if embedding else None
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO memories (
                    created_at, user_id, workspace_id, category, content, source, embedding_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(UTC).isoformat(),
                    user_id,
                    workspace_id,
                    category,
                    clean,
                    source,
                    embedding_json,
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

        scored: list[tuple[float, int, Memory]] = []
        query_embedding = _get_embedding(query)
        
        for memory in candidates:
            score = 0.0
            content_tokens = set(re.findall(r"[A-Za-z0-9_-]{3,}", memory.content.lower()))
            token_score = len(tokens & content_tokens)
            
            if query_embedding and memory.embedding_json:
                try:
                    mem_emb = json.loads(memory.embedding_json)
                    sem_score = _cosine_similarity(query_embedding, mem_emb)
                    # Require at least 0.4 cosine similarity to be considered a semantic match
                    score = (sem_score * 10) + token_score if sem_score > 0.4 else token_score
                except Exception:
                    score = token_score
            else:
                score = token_score
                
            if score > 0:
                scored.append((score, memory.id, memory))
                
        scored.sort(reverse=True)
        return [item[2] for item in scored[:limit]]

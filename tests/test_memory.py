import sqlite3
from pathlib import Path

from hybrid_agent.memory import MemoryStore


def test_memory_is_scoped_and_searchable(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.db")
    store.add("Jon prefers CLI-first development", workspace_id="agent")
    store.add("Unrelated workspace fact", workspace_id="other")

    results = store.search("CLI", workspace_id="agent")
    assert [memory.content for memory in results] == ["Jon prefers CLI-first development"]


def test_empty_memory_is_rejected(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.db")
    try:
        store.add("  ")
    except ValueError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("Expected empty memory to be rejected")


def test_relevant_memory_uses_token_overlap(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.db")
    store.add("The project uses a CLI-first development workflow")
    store.add("Jon likes blue")
    results = store.relevant("How should the CLI workflow behave?")
    assert [memory.content for memory in results] == [
        "The project uses a CLI-first development workflow"
    ]


def test_legacy_database_is_migrated_without_recreating_table(tmp_path: Path) -> None:
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as connection:
        connection.execute(
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
    store = MemoryStore(path)
    memory_id = store.add("Migrated successfully")
    assert memory_id == 1
    with sqlite3.connect(path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(memories)")}
        versions = connection.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
    assert "embedding_json" in columns
    assert versions == [(0,), (1,)]

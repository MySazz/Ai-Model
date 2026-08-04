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

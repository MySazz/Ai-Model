from pathlib import Path

import pytest

from hybrid_agent.research import ResearchStore


def test_source_search_has_line_provenance(tmp_path: Path) -> None:
    document = tmp_path / "report.md"
    document.write_text("First line\nImportant evidence\n", encoding="utf-8")
    store = ResearchStore(tmp_path / "state" / "research.db")
    source_id = store.add_workspace_file(document, workspace=tmp_path)

    matches = store.search("evidence")
    assert len(matches) == 1
    assert matches[0].source_id == source_id
    assert matches[0].line_number == 2
    assert matches[0].sha256


def test_duplicate_source_returns_existing_id(tmp_path: Path) -> None:
    store = ResearchStore(tmp_path / "research.db")
    first = store.add_text(title="One", uri="one", content="same content")
    second = store.add_text(title="Two", uri="two", content="same content")
    assert first == second


def test_source_path_is_workspace_scoped(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    store = ResearchStore(tmp_path / "research.db")
    with pytest.raises(PermissionError):
        store.add_workspace_file(outside, workspace=workspace)

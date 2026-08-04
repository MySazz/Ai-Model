from pathlib import Path

import pytest

from hybrid_agent.permissions import RiskLevel
from hybrid_agent.research import ResearchStore
from hybrid_agent.tools import default_registry


def test_read_file_is_workspace_scoped(tmp_path: Path) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("hello", encoding="utf-8")
    registry = default_registry(tmp_path)
    assert registry.get("read_file").handler({"path": "sample.txt"}) == "hello"


def test_path_escape_is_rejected(tmp_path: Path) -> None:
    registry = default_registry(tmp_path)
    with pytest.raises(PermissionError):
        registry.resolve_path("../secret.txt")


def test_search_text_returns_file_and_line(tmp_path: Path) -> None:
    (tmp_path / "module.py").write_text("first\nTarget value\n", encoding="utf-8")
    registry = default_registry(tmp_path)
    result = registry.get("search_text").handler({"query": "target"})
    assert "module.py:2:Target value" in result


def test_write_tool_refuses_overwrite(tmp_path: Path) -> None:
    (tmp_path / "existing.txt").write_text("keep", encoding="utf-8")
    registry = default_registry(tmp_path)
    with pytest.raises(FileExistsError):
        registry.get("write_new_file").handler(
            {"path": "existing.txt", "content": "replace"}
        )


def test_command_risk_is_dynamic(tmp_path: Path) -> None:
    tool = default_registry(tmp_path).get("run_command")
    assert tool.assess_risk({"argv": ["git", "status"]}) is RiskLevel.OBSERVE
    assert tool.assess_risk({"argv": ["python3", "-m", "pytest"]}) is RiskLevel.APPROVAL
    assert tool.assess_risk({"argv": ["rm", "-rf", "."]}) is RiskLevel.PROHIBITED


def test_command_rejects_non_allowlisted_git_argument(tmp_path: Path) -> None:
    tool = default_registry(tmp_path).get("run_command")
    with pytest.raises(PermissionError):
        tool.handler({"argv": ["git", "diff", "--no-index"]})


def test_command_returns_exit_code_and_bounded_output(tmp_path: Path) -> None:
    tool = default_registry(tmp_path).get("run_command")
    result = tool.handler({"argv": ["git", "status", "--short"]})
    assert result.startswith("exit_code=")
    assert len(result) <= 100_020


def test_patch_preview_and_backup(tmp_path: Path) -> None:
    target = tmp_path / "module.py"
    target.write_text("value = 1\n", encoding="utf-8")
    tool = default_registry(tmp_path).get("apply_text_patch")
    arguments = {"path": "module.py", "old_text": "value = 1", "new_text": "value = 2"}
    preview = tool.preview(arguments)
    assert preview and "-value = 1" in preview and "+value = 2" in preview

    result = tool.handler(arguments)
    assert target.read_text(encoding="utf-8") == "value = 2\n"
    assert "Backup:" in result
    backups = list((tmp_path / ".hybrid-agent" / "backups").glob("*.bak"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == "value = 1\n"


def test_patch_requires_one_exact_match(tmp_path: Path) -> None:
    target = tmp_path / "module.py"
    target.write_text("same\nsame\n", encoding="utf-8")
    tool = default_registry(tmp_path).get("apply_text_patch")
    with pytest.raises(ValueError, match="exactly once"):
        tool.preview({"path": "module.py", "old_text": "same", "new_text": "changed"})


def test_research_tool_returns_provenance(tmp_path: Path) -> None:
    store = ResearchStore(tmp_path / "research.db")
    store.add_text(title="Report", uri="report.md", content="Verified finding")
    tool = default_registry(tmp_path, store).get("search_research_sources")
    result = tool.handler({"query": "finding"})
    assert "source_id=" in result
    assert "report.md:1" in result
    assert "sha256=" in result

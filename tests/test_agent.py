from pathlib import Path

import pytest

from hybrid_agent.agent import AgentSession
from hybrid_agent.audit import AuditStore
from hybrid_agent.memory import MemoryStore
from hybrid_agent.models import ImageInput, Message, ModelTurn, ToolCall
from hybrid_agent.tools import default_registry


class ScriptedProvider:
    name = "scripted"
    is_cloud = False

    def __init__(self) -> None:
        self.turn = 0

    def complete(
        self, messages: list[Message], tools: list[dict[str, object]]
    ) -> ModelTurn:
        self.turn += 1
        if self.turn == 1:
            return ModelTurn(
                tool_calls=(
                    ToolCall(id="call-1", name="read_file", arguments={"path": "note.txt"}),
                )
            )
        assert any("evidence" in message.content for message in messages if message.role == "tool")
        return ModelTurn(content="Verified from the workspace.")


def test_agent_executes_tool_and_records_audit(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("evidence", encoding="utf-8")
    audit = AuditStore(tmp_path / "state" / "agent.db")
    session = AgentSession(
        provider=ScriptedProvider(),
        tools=default_registry(tmp_path),
        audit=audit,
    )

    assert session.run("Read the note") == "Verified from the workspace."
    events = audit.recent()
    assert any(event.event_type == "tool" and event.status == "completed" for event in events)


class WriteProvider:
    name = "scripted-write"
    is_cloud = False

    def __init__(self) -> None:
        self.turn = 0

    def complete(
        self, messages: list[Message], tools: list[dict[str, object]]
    ) -> ModelTurn:
        self.turn += 1
        if self.turn == 1:
            return ModelTurn(
                tool_calls=(
                    ToolCall(
                        id="write-1",
                        name="write_new_file",
                        arguments={"path": "created.txt", "content": "approved"},
                    ),
                )
            )
        return ModelTurn(content="Done.")


def test_approved_tool_executes(tmp_path: Path) -> None:
    session = AgentSession(
        provider=WriteProvider(),
        tools=default_registry(tmp_path),
        audit=AuditStore(tmp_path / "state" / "agent.db"),
        approval_handler=lambda tool, arguments, reason: True,
    )
    assert session.run("Create it") == "Done."
    assert (tmp_path / "created.txt").read_text(encoding="utf-8") == "approved"


def test_unapproved_tool_does_not_execute(tmp_path: Path) -> None:
    session = AgentSession(
        provider=WriteProvider(),
        tools=default_registry(tmp_path),
        audit=AuditStore(tmp_path / "state" / "agent.db"),
    )
    assert session.run("Create it") == "Done."
    assert not (tmp_path / "created.txt").exists()


class CloudProvider:
    name = "test-cloud"
    is_cloud = True

    def complete(
        self, messages: list[Message], tools: list[dict[str, object]]
    ) -> ModelTurn:
        return ModelTurn(content="cloud response")


def test_cloud_provider_blocks_secret_like_prompt(tmp_path: Path) -> None:
    session = AgentSession(
        provider=CloudProvider(),
        tools=default_registry(tmp_path),
        audit=AuditStore(tmp_path / "state" / "agent.db"),
    )
    with pytest.raises(PermissionError, match="local-only"):
        session.run("api_key=super-secret-value")


def test_cloud_provider_requires_approval_for_sensitive_prompt(tmp_path: Path) -> None:
    session = AgentSession(
        provider=CloudProvider(),
        tools=default_registry(tmp_path),
        audit=AuditStore(tmp_path / "state" / "agent.db"),
        cloud_approval_handler=lambda level, reasons: True,
    )
    assert session.run("Review this private configuration") == "cloud response"


class MemoryAwareProvider:
    name = "memory-aware"
    is_cloud = False

    def complete(
        self, messages: list[Message], tools: list[dict[str, object]]
    ) -> ModelTurn:
        memory_messages = [
            message.content
            for message in messages
            if message.role == "system" and "memory_id=" in message.content
        ]
        assert any("CLI-first" in content for content in memory_messages)
        return ModelTurn(content="I used the approved memory.")


def test_relevant_memory_is_injected_with_provenance(tmp_path: Path) -> None:
    memory = MemoryStore(tmp_path / "state" / "memory.db")
    memory.add("The project is CLI-first", workspace_id="agent")
    session = AgentSession(
        provider=MemoryAwareProvider(),
        tools=default_registry(tmp_path),
        audit=AuditStore(tmp_path / "state" / "agent.db"),
        memory=memory,
        workspace_id="agent",
    )
    assert session.run("What is our CLI project approach?") == "I used the approved memory."


def test_cloud_image_requires_explicit_disclosure_approval(tmp_path: Path) -> None:
    session = AgentSession(
        provider=CloudProvider(),
        tools=default_registry(tmp_path),
        audit=AuditStore(tmp_path / "state" / "agent.db"),
    )
    image = ImageInput(
        name="screen.png",
        media_type="image/png",
        data=b"image",
        sha256="digest",
    )
    with pytest.raises(PermissionError, match="image content"):
        session.run("Inspect this screenshot", images=(image,))


class PatchProvider:
    name = "scripted-patch"
    is_cloud = False

    def __init__(self) -> None:
        self.turn = 0

    def complete(
        self, messages: list[Message], tools: list[dict[str, object]]
    ) -> ModelTurn:
        self.turn += 1
        if self.turn == 1:
            return ModelTurn(
                tool_calls=(
                    ToolCall(
                        id="patch-1",
                        name="apply_text_patch",
                        arguments={
                            "path": "target.txt",
                            "old_text": "before",
                            "new_text": "after",
                        },
                    ),
                )
            )
        return ModelTurn(content="Patch turn complete.")


def test_patch_is_denied_if_file_changes_after_preview(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("before\noriginal context\n", encoding="utf-8")

    def change_during_approval(tool, arguments, reason) -> bool:
        target.write_text("before\nchanged context\n", encoding="utf-8")
        return True

    audit = AuditStore(tmp_path / "state" / "agent.db")
    session = AgentSession(
        provider=PatchProvider(),
        tools=default_registry(tmp_path),
        audit=audit,
        approval_handler=change_during_approval,
    )
    assert session.run("Patch target") == "Patch turn complete."
    assert target.read_text(encoding="utf-8") == "before\nchanged context\n"
    assert any(event.status == "approval_required" for event in audit.recent())

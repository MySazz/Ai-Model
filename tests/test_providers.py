import json
from pathlib import Path

import pytest

from hybrid_agent.agent import AgentSession
from hybrid_agent.audit import AuditStore
from hybrid_agent.models import Message
from hybrid_agent.providers import OllamaProvider
from hybrid_agent.tools import default_registry


class _Response:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        raw = json.dumps(self.payload).encode()
        return raw if size < 0 else raw[:size]


@pytest.mark.asyncio
async def test_ollama_tool_call_round_trip(monkeypatch, tmp_path: Path) -> None:
    requests = []
    responses = iter(
        [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "read_file", "arguments": {"path": "note.txt"}}}
                    ],
                }
            },
            {"message": {"role": "assistant", "content": "Verified."}},
        ]
    )

    def fake_urlopen(request, timeout):
        requests.append(request)
        return _Response(next(responses))

    monkeypatch.setattr("hybrid_agent.providers.urlopen", fake_urlopen)
    (tmp_path / "note.txt").write_text("evidence", encoding="utf-8")
    session = AgentSession(
        provider=OllamaProvider("tool-model"),
        tools=default_registry(tmp_path),
        audit=AuditStore(tmp_path / "state" / "agent.db"),
    )

    assert await session.run("Read note.txt") == "Verified."
    first = json.loads(requests[0].data)
    assert first["tools"][0]["type"] == "function"
    second = json.loads(requests[1].data)
    assistant = next(message for message in second["messages"] if message["role"] == "assistant")
    tool_result = next(message for message in second["messages"] if message["role"] == "tool")
    assert assistant["tool_calls"][0]["function"]["name"] == "read_file"
    assert tool_result == {"role": "tool", "content": "evidence", "tool_name": "read_file"}


def test_ollama_message_preserves_tool_name() -> None:
    assert OllamaProvider._ollama_message(
        Message(role="tool", content="ok", tool_name="read_file")
    ) == {"role": "tool", "content": "ok", "tool_name": "read_file"}


def test_ollama_rejects_non_http_url() -> None:
    with pytest.raises(ValueError, match="http or https"):
        OllamaProvider("model", base_url="file:///tmp/fake")

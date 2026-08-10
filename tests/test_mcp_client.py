import io
import json
from unittest.mock import patch

from hybrid_agent.mcp_client import MCPClient
from hybrid_agent.permissions import RiskLevel


class _Writable:
    def __init__(self) -> None:
        self.values: list[str] = []
        self.closed = False

    def write(self, value: str) -> int:
        self.values.append(value)
        return len(value)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


class _Process:
    def __init__(self) -> None:
        self.stdin = _Writable()
        self.stdout = io.StringIO(
            json.dumps({"jsonrpc": "2.0", "id": 1, "result": {}})
            + "\n"
            + json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "result": {
                        "tools": [
                            {
                                "name": "echo",
                                "description": "Echo text",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {"text": {"type": "string"}},
                                    "required": ["text"],
                                },
                            }
                        ]
                    },
                }
            )
            + "\n"
            + json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "result": {"content": [{"type": "text", "text": "echoed"}]},
                }
            )
            + "\n"
        )
        self.stderr = io.StringIO("")
        self.returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout):
        if self.stdin.closed:
            self.returncode = 0
            return 0
        return 0

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9


def test_mcp_handshake_notification_and_tool_mapping(monkeypatch) -> None:
    process = _Process()
    captured = {}

    def fake_popen(command, **kwargs):
        captured.update(kwargs)
        return process

    monkeypatch.setenv("MCP_TEST_TOKEN", "selected-secret")
    with patch("hybrid_agent.mcp_client.subprocess.Popen", side_effect=fake_popen):
        client = MCPClient(["server"], pass_environment=["MCP_TEST_TOKEN"])
        tools = client.list_tools()
        result = tools[0].handler({"text": "hello"})
        client.close()

    messages = [json.loads(value) for value in process.stdin.values]
    initialized = next(message for message in messages if message["method"] == "notifications/initialized")
    assert "id" not in initialized
    assert tools[0].name == "mcp_echo"
    assert tools[0].assess_risk({"text": "hi"}) is RiskLevel.APPROVAL
    assert result == "echoed"
    assert captured["env"]["MCP_TEST_TOKEN"] == "selected-secret"
    assert "HOME" not in captured["env"]
    assert process.stdin.closed

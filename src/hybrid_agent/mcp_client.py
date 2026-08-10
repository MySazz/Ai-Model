"""Zero-dependency, synchronous Model Context Protocol (MCP) Stdio Client."""

from __future__ import annotations

import json
import subprocess
import threading
import time
from typing import Any

from .permissions import RiskLevel
from .tools import Tool


class MCPClient:
    """Minimal, synchronous MCP client communicating over stdio."""

    def __init__(self, command: list[str]) -> None:
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,  # Line-buffered
        )
        self.msg_id = 1
        self.responses: dict[int, dict[str, Any]] = {}
        self.lock = threading.Lock()
        self.is_running = True

        self.reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self.reader_thread.start()
        
        # Perform MCP initialization handshake
        init_res = self.call(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "hybrid-agent", "version": "1.0"},
            },
        )
        if "error" in init_res:
            raise RuntimeError(f"MCP Initialization failed: {init_res['error']}")
        self.call("notifications/initialized")

    def _read_loop(self) -> None:
        if not self.process.stdout:
            return
        for line in self.process.stdout:
            if not line.strip():
                continue
            try:
                msg = json.loads(line)
                if "id" in msg:
                    with self.lock:
                        self.responses[msg["id"]] = msg
            except json.JSONDecodeError:
                pass

    def call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Send a JSON-RPC request and block until a response is received."""
        with self.lock:
            msg_id = self.msg_id
            self.msg_id += 1

        req = {"jsonrpc": "2.0", "id": msg_id, "method": method}
        if params is not None:
            req["params"] = params

        if not self.process.stdin:
            raise RuntimeError("MCP process stdin is closed.")

        self.process.stdin.write(json.dumps(req) + "\n")
        self.process.stdin.flush()

        # Synchronous blocking wait with timeout
        start_time = time.time()
        while time.time() - start_time < 30:  # 30 second timeout
            with self.lock:
                if msg_id in self.responses:
                    return self.responses.pop(msg_id)
            time.sleep(0.05)
            
        raise TimeoutError(f"MCP server timed out responding to '{method}'")

    def list_tools(self) -> list[Tool]:
        """Fetch tools from the MCP server and convert them into hybrid-agent Tool objects."""
        response = self.call("tools/list")
        if "error" in response:
            raise RuntimeError(f"Failed to list MCP tools: {response['error']}")
            
        mcp_tools = response.get("result", {}).get("tools", [])
        agent_tools: list[Tool] = []
        
        for t in mcp_tools:
            name = t["name"]
            description = t.get("description", "")
            schema = t.get("inputSchema", {"type": "object", "properties": {}})
            
            # Create a closure to capture the tool name for execution
            def make_handler(tool_name: str):
                def handler(arguments: dict[str, Any]) -> str:
                    res = self.call("tools/call", {"name": tool_name, "arguments": arguments})
                    if "error" in res:
                        return f"MCP Error: {res['error']}"
                    content = res.get("result", {}).get("content", [])
                    return "\n".join(
                        c.get("text", str(c)) for c in content if c.get("type") == "text"
                    )
                return handler

            # Map all MCP tools to APPROVAL risk by default as they are external and unknown
            agent_tools.append(
                Tool(
                    name=f"mcp_{name}",
                    description=f"[MCP] {description}",
                    parameters=schema,
                    assess_risk=lambda _args: RiskLevel.APPROVAL,
                    handler=make_handler(name),
                )
            )
        return agent_tools

    def close(self) -> None:
        self.is_running = False
        self.process.terminate()

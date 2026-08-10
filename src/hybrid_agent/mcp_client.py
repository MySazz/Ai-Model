"""Small synchronous Model Context Protocol (MCP) client for stdio servers."""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from typing import Any

from .permissions import RiskLevel
from .tools import Tool

PROTOCOL_VERSION = "2024-11-05"
DEFAULT_TIMEOUT = 30.0
MAX_MESSAGE_CHARS = 2 * 1024 * 1024
MAX_BUFFERED_RESPONSES = 64
_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class MCPClient:
    """Synchronous JSON-RPC client for a user-selected MCP stdio process."""

    def __init__(
        self,
        command: list[str],
        *,
        pass_environment: Iterable[str] = (),
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        if not command or not all(isinstance(part, str) and part for part in command):
            raise ValueError("MCP command must be a non-empty list of strings.")
        if timeout <= 0:
            raise ValueError("MCP timeout must be positive.")
        self.timeout = timeout
        self.msg_id = 1
        self.responses: dict[int, dict[str, Any]] = {}
        self.pending_ids: set[int] = set()
        self.condition = threading.Condition()
        self.reader_error: str | None = None
        self.stderr_lines: deque[str] = deque(maxlen=20)
        self.closed = False

        environment = {
            "PATH": os.environ.get("PATH") or os.defpath,
            "LANG": os.environ.get("LANG", "C.UTF-8"),
        }
        for name in pass_environment:
            if not _ENVIRONMENT_NAME.fullmatch(name):
                raise ValueError(f"Invalid environment variable name: {name!r}")
            if name not in os.environ:
                raise ValueError(f"Requested MCP environment variable is not set: {name}")
            environment[name] = os.environ[name]

        self.process = subprocess.Popen(  # noqa: S603 - explicit user-selected stdio server
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=environment,
        )
        self.reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self.stderr_thread = threading.Thread(target=self._stderr_loop, daemon=True)
        self.reader_thread.start()
        self.stderr_thread.start()

        try:
            response = self.call(
                "initialize",
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "hybrid-agent", "version": "0.1.0"},
                },
            )
            self._raise_rpc_error(response, "MCP initialization")
            self.notify("notifications/initialized")
        except BaseException:
            self.close()
            raise

    def _read_loop(self) -> None:
        if not self.process.stdout:
            return
        try:
            while True:
                line = self.process.stdout.readline(MAX_MESSAGE_CHARS + 1)
                if not line:
                    break
                if len(line) > MAX_MESSAGE_CHARS:
                    while line and not line.endswith("\n"):
                        line = self.process.stdout.readline(MAX_MESSAGE_CHARS + 1)
                    self._set_reader_error(
                        f"MCP server response exceeds {MAX_MESSAGE_CHARS} characters."
                    )
                    return
                if not line.strip():
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError as exc:
                    self._set_reader_error(
                        f"MCP server wrote invalid JSON to stdout: {exc.msg}"
                    )
                    return
                message_id = message.get("id") if isinstance(message, dict) else None
                if isinstance(message_id, int):
                    with self.condition:
                        if (
                            message_id in self.pending_ids
                            or len(self.responses) < MAX_BUFFERED_RESPONSES
                        ):
                            self.responses[message_id] = message
                        else:
                            self.reader_error = (
                                "MCP server exceeded the buffered response limit."
                            )
                        self.condition.notify_all()
                        if self.reader_error:
                            return
        except (OSError, ValueError) as exc:
            self._set_reader_error(f"MCP stdout failed: {exc}")
        finally:
            with self.condition:
                self.condition.notify_all()

    def _stderr_loop(self) -> None:
        if not self.process.stderr:
            return
        try:
            while line := self.process.stderr.readline(2_001):
                self.stderr_lines.append(line.rstrip()[:2_000])
        except (OSError, ValueError):
            return

    def _set_reader_error(self, message: str) -> None:
        with self.condition:
            self.reader_error = message
            self.condition.notify_all()

    def _send(self, payload: dict[str, Any]) -> None:
        if self.closed or not self.process.stdin or self.process.stdin.closed:
            raise RuntimeError("MCP process stdin is closed.")
        try:
            self.process.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            raise RuntimeError(f"Could not write to MCP server: {exc}") from exc

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        """Send a JSON-RPC notification, which intentionally has no request ID."""
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        self._send(payload)

    def call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Send a JSON-RPC request and block until its response arrives."""
        with self.condition:
            message_id = self.msg_id
            self.msg_id += 1
            self.pending_ids.add(message_id)
        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": message_id,
            "method": method,
        }
        if params is not None:
            payload["params"] = params
        try:
            self._send(payload)
        except Exception:
            with self.condition:
                self.pending_ids.discard(message_id)
            raise

        deadline = time.monotonic() + self.timeout
        with self.condition:
            while True:
                if message_id in self.responses:
                    self.pending_ids.discard(message_id)
                    return self.responses.pop(message_id)
                if self.reader_error:
                    self.pending_ids.discard(message_id)
                    raise RuntimeError(self.reader_error)
                return_code = self.process.poll()
                if return_code is not None:
                    self.pending_ids.discard(message_id)
                    detail = "\n".join(self.stderr_lines)
                    suffix = f"\nMCP stderr:\n{detail}" if detail else ""
                    raise RuntimeError(
                        f"MCP server exited with code {return_code} while handling "
                        f"{method}.{suffix}"
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.pending_ids.discard(message_id)
                    raise TimeoutError(f"MCP server timed out responding to {method!r}.")
                self.condition.wait(timeout=min(remaining, 0.25))

    @staticmethod
    def _raise_rpc_error(response: dict[str, Any], context: str) -> None:
        if "error" in response:
            raise RuntimeError(f"{context} failed: {response['error']}")

    def list_tools(self) -> list[Tool]:
        """Fetch MCP tools and map them to approval-required local tools."""
        response = self.call("tools/list")
        self._raise_rpc_error(response, "MCP tool discovery")
        result_block = response.get("result")
        if not isinstance(result_block, dict):
            raise RuntimeError("MCP tools/list response is missing its result object.")
        raw_tools = result_block.get("tools", [])
        if not isinstance(raw_tools, list):
            raise RuntimeError("MCP tools/list response does not contain a tool array.")

        tools: list[Tool] = []
        for raw_tool in raw_tools:
            if not isinstance(raw_tool, dict) or not isinstance(raw_tool.get("name"), str):
                raise RuntimeError("MCP server returned a malformed tool definition.")
            name = raw_tool["name"]
            description = raw_tool.get("description", "")
            schema = raw_tool.get("inputSchema", {"type": "object", "properties": {}})
            if not isinstance(schema, dict):
                raise RuntimeError(f"MCP tool {name!r} has an invalid input schema.")

            def make_handler(
                tool_name: str,
            ) -> Callable[[dict[str, Any]], str]:
                def handler(arguments: dict[str, Any]) -> str:
                    result = self.call(
                        "tools/call", {"name": tool_name, "arguments": arguments}
                    )
                    self._raise_rpc_error(result, f"MCP tool {tool_name}")
                    result_block = result.get("result")
                    if not isinstance(result_block, dict):
                        raise RuntimeError(
                            f"MCP tool {tool_name!r} response is missing its result object."
                        )
                    content = result_block.get("content", [])
                    if not isinstance(content, list):
                        raise RuntimeError(f"MCP tool {tool_name!r} returned invalid content.")
                    rendered: list[str] = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            rendered.append(str(item.get("text", "")))
                        else:
                            rendered.append(json.dumps(item, sort_keys=True, default=str))
                    return "\n".join(rendered)

                return handler

            tools.append(
                Tool(
                    name=f"mcp_{name}",
                    description=f"[MCP] {description}",
                    risk=RiskLevel.APPROVAL,
                    parameters=schema,
                    handler=make_handler(name),
                )
            )
        return tools

    def close(self) -> None:
        """Close stdin, then escalate from a graceful wait to TERM and KILL."""
        if self.closed:
            return
        self.closed = True
        if self.process.stdin and not self.process.stdin.closed:
            try:
                self.process.stdin.close()
            except OSError:
                pass
        try:
            self.process.wait(timeout=2)
            return
        except subprocess.TimeoutExpired:
            self.process.terminate()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=2)

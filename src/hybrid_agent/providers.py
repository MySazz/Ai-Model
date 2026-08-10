"""Initial model providers."""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from uuid import uuid4

from .models import Message, ModelTurn, ToolCall

MAX_RESPONSE_BYTES = 10 * 1024 * 1024


class OfflineProvider:
    """A credential-free provider used for setup checks and architecture testing."""

    name = "offline"
    is_cloud = False

    async def complete(self, messages: list[Message], tools: list[dict[str, object]]) -> ModelTurn:
        prompt = next(
            (message.content for message in reversed(messages) if message.role == "user"),
            "",
        )
        return ModelTurn(
            content=(
                "The agent core is running in offline setup mode. "
                f"I received: {prompt!r}. "
                f"Image inputs: {sum(len(message.images) for message in messages)}. "
                "Configure an Ollama model to generate real responses."
            )
        )


class OllamaProvider:
    """Adapter for Ollama's local chat API."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 120.0,
    ) -> None:
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Ollama URL must use http or https and include a host.")
        if timeout <= 0:
            raise ValueError("Ollama timeout must be positive.")
        self.model = model
        self.endpoint = f"{base_url.rstrip('/')}/api/chat"
        self.timeout = timeout
        self.name = f"ollama:{model}"
        self.is_cloud = False

    async def complete(self, messages: list[Message], tools: list[dict[str, object]]) -> ModelTurn:
        payload: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "messages": [self._ollama_message(message) for message in messages],
        }
        if tools:
            payload["tools"] = tools

        request = Request(  # noqa: S310 - constructor restricts endpoint scheme
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            def fetch() -> Any:
                with urlopen(request, timeout=self.timeout) as response:  # noqa: S310
                    raw = response.read(MAX_RESPONSE_BYTES + 1)
                    if len(raw) > MAX_RESPONSE_BYTES:
                        raise RuntimeError(
                            f"Ollama response exceeds {MAX_RESPONSE_BYTES} bytes."
                        )
                    return json.loads(raw.decode("utf-8"))
            body = await asyncio.to_thread(fetch)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Ollama returned invalid JSON: {exc.msg}") from exc
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.endpoint}. Is `ollama serve` running?"
            ) from exc

        if not isinstance(body, dict):
            raise RuntimeError("Ollama response must be a JSON object.")
        message = body.get("message")
        if not isinstance(message, dict):
            raise RuntimeError("Ollama response is missing a message object.")
        content = message.get("content", "")
        if not isinstance(content, str):
            raise RuntimeError("Ollama response message content must be text.")
        calls: list[ToolCall] = []
        try:
            for call in message.get("tool_calls", []):
                function = call["function"]
                arguments = function.get("arguments", {})
                if not isinstance(arguments, dict):
                    raise TypeError("tool arguments must be an object")
                name = function["name"]
                if not isinstance(name, str) or not name:
                    raise TypeError("tool name must be non-empty text")
                calls.append(
                    ToolCall(
                        id=str(call.get("id") or uuid4()),
                        name=name,
                        arguments=arguments,
                    )
                )
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"Ollama returned malformed tool calls: {exc}") from exc
        return ModelTurn(content=content, tool_calls=tuple(calls))

    @staticmethod
    def _ollama_message(message: Message) -> dict[str, object]:
        result: dict[str, object] = {"role": message.role, "content": message.content}
        if message.images:
            result["images"] = [
                base64.b64encode(image.data).decode("ascii") for image in message.images
            ]
        if message.tool_calls:
            result["tool_calls"] = [
                {
                    "function": {
                        "name": call.name,
                        "arguments": call.arguments,
                    }
                }
                for call in message.tool_calls
            ]
        if message.tool_name:
            result["tool_name"] = message.tool_name
        return result

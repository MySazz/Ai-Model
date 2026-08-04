"""Initial model providers."""

from __future__ import annotations

import base64
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

from .models import Message, ModelTurn, ToolCall


class OfflineProvider:
    """A credential-free provider used for setup checks and architecture testing."""

    name = "offline"
    is_cloud = False

    def complete(self, messages: list[Message], tools: list[dict[str, object]]) -> ModelTurn:
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
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.name = f"ollama:{model}"
        self.is_cloud = False

    def complete(self, messages: list[Message], tools: list[dict[str, object]]) -> ModelTurn:
        ollama_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"],
                },
            }
            for tool in tools
        ]
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [self._ollama_message(message) for message in messages],
            "tools": ollama_tools,
        }
        request = Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.base_url}. Is `ollama serve` running?"
            ) from exc

        message = body.get("message", {})
        calls = tuple(
            ToolCall(
                id=str(uuid4()),
                name=call["function"]["name"],
                arguments=call["function"].get("arguments", {}),
            )
            for call in message.get("tool_calls", [])
        )
        return ModelTurn(content=message.get("content", ""), tool_calls=calls)

    @staticmethod
    def _ollama_message(message: Message) -> dict[str, object]:
        result: dict[str, object] = {"role": message.role, "content": message.content}
        if message.images:
            result["images"] = [
                base64.b64encode(image.data).decode("ascii") for image in message.images
            ]
        return result

"""Provider-neutral model contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ImageInput:
    name: str
    media_type: str
    data: bytes
    sha256: str


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Message:
    role: str
    content: str
    images: tuple[ImageInput, ...] = field(default_factory=tuple)
    tool_calls: tuple[ToolCall, ...] = field(default_factory=tuple)
    tool_name: str | None = None
    tool_call_id: str | None = None


@dataclass(frozen=True)
class ModelTurn:
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = field(default_factory=tuple)


class ModelProvider(Protocol):
    name: str
    is_cloud: bool

    async def complete(
        self, messages: list[Message], tools: list[dict[str, object]]
    ) -> ModelTurn:
        """Return the model's next response or requested tool calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    role: str  # user, assistant, tool, system
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Session:
    """Maintain multi-turn conversation state."""

    messages: list[Message] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add(self, role: str, content: str, **kwargs: Any) -> None:
        self.messages.append(Message(role=role, content=content, metadata=kwargs))

    def to_llm_history(self) -> list[dict[str, Any]]:
        """Convert to LLM-compatible message format."""
        return [
            {"role": msg.role, "content": msg.content}
            for msg in self.messages
        ]

    def last_user_message(self) -> str:
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg.content
        return ""

    def clear(self) -> None:
        self.messages.clear()
        self.metadata.clear()

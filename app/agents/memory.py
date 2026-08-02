from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ShortTermMemory:
    """Bounded scratchpad shared across agent roles in one run."""

    max_items: int = 40
    items: list[dict[str, Any]] = field(default_factory=list)

    def add(self, role: str, content: str, **extra: Any) -> None:
        self.items.append({"role": role, "content": content, **extra})
        if len(self.items) > self.max_items:
            self.items = self.items[-self.max_items :]

    def as_text(self) -> str:
        lines = []
        for it in self.items:
            lines.append(f"[{it['role']}] {it['content']}")
        return "\n".join(lines)

    def by_role(self, role: str) -> list[dict[str, Any]]:
        return [i for i in self.items if i["role"] == role]

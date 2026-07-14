from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


# Rough USD per 1M tokens (illustrative portfolio estimates, not billing)
COST_PER_1M = {
    "mock-workbench": (0.0, 0.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.0),
    "claude-3-5-haiku": (0.80, 4.0),
    "gemini-1.5-flash": (0.075, 0.30),
    "llama-3.1-8b": (0.05, 0.08),
    "groq-llama": (0.05, 0.08),
}


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    inp, out = COST_PER_1M.get(model, (0.5, 1.5))
    return round((prompt_tokens * inp + completion_tokens * out) / 1_000_000, 6)


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    id: str = ""


@dataclass
class CompletionResult:
    text: str
    model: str
    provider: str
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    tool_calls: list[ToolCall] = field(default_factory=list)
    structured: dict[str, Any] | None = None
    raw: dict[str, Any] | None = None

    @property
    def tokens_estimate(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@runtime_checkable
class LLMProvider(Protocol):
    name: str

    def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.2,
        json_mode: bool = False,
        tools: list[dict[str, Any]] | None = None,
        mock_builder=None,
    ) -> CompletionResult: ...

    def models(self) -> list[dict[str, str]]: ...

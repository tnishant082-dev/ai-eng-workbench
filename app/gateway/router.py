from __future__ import annotations

from typing import Any

from app.core.config import Settings
from app.llm.base import CompletionResult, estimate_cost, estimate_tokens
from app.llm.factory import get_provider


class GatewayRouter:
    """Routes completions through the configured LLM provider; tracks token/cost estimates."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.provider = get_provider(settings)
        self._log: list[dict[str, Any]] = []

    @property
    def mode(self) -> str:
        return self.provider.name

    def models(self) -> list[dict[str, str]]:
        models = self.provider.models()
        # always advertise mock availability
        if not any(m.get("provider") == "mock" for m in models):
            models.insert(
                0,
                {
                    "id": "mock-workbench",
                    "provider": "mock",
                    "description": "Deterministic offline composer (default)",
                },
            )
        return models

    def complete(self, *, system: str, user: str, purpose: str = "chat", mock_builder=None) -> CompletionResult:
        result = self.provider.complete(system=system, user=user, mock_builder=mock_builder)
        self._log.append(
            {
                "purpose": purpose,
                "model": result.model,
                "provider": result.provider,
                "tokens": result.tokens_estimate,
                "cost_usd": result.cost_usd,
                "latency_ms": result.latency_ms,
            }
        )
        return result

    def compare_estimate(self, prompt: str, completion_chars: int = 400) -> list[dict[str, Any]]:
        """Side-by-side token/cost estimate across known model price cards."""
        pt = estimate_tokens(prompt)
        ct = max(1, completion_chars // 4)
        catalog = [
            "mock-workbench",
            "gpt-4o-mini",
            "gpt-4o",
            "claude-3-5-haiku",
            "gemini-1.5-flash",
            "llama-3.1-8b",
        ]
        return [
            {
                "model": m,
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "est_cost_usd": estimate_cost(m, pt, ct),
            }
            for m in catalog
        ]

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        return list(reversed(self._log[-limit:]))

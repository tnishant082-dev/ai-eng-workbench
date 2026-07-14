from __future__ import annotations

from typing import Any

from app.llm.base import CompletionResult
from app.llm.mock import MockProvider
from app.llm.openai_compat import OpenAICompatibleProvider


class _KeyGatedProvider:
    """Provider stub that errors clearly without a key, or delegates when present."""

    name = "stub"
    key_env: str = "API_KEY"
    model_id: str = "stub-model"

    def __init__(self, api_key: str | None, **kwargs: Any):
        self.api_key = api_key
        self.kwargs = kwargs
        self._mock = MockProvider()

    def models(self) -> list[dict[str, str]]:
        status = "ready" if self.api_key else f"needs {self.key_env}"
        return [
            {
                "id": self.model_id,
                "provider": self.name,
                "description": f"{self.name} ({status})",
            }
        ]

    def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.2,
        json_mode: bool = False,
        tools: list[dict[str, Any]] | None = None,
        mock_builder=None,
    ) -> CompletionResult:
        if not self.api_key:
            raise RuntimeError(
                f"{self.name} provider requires {self.key_env}. "
                f"Set the env var or use llm_provider=mock for offline demos."
            )
        # When key present, try OpenAI-compatible path if base_url given, else clear error
        base = self.kwargs.get("base_url")
        model = self.kwargs.get("model", self.model_id)
        if base:
            return OpenAICompatibleProvider(self.api_key, base, model).complete(
                system=system,
                user=user,
                temperature=temperature,
                json_mode=json_mode,
                tools=tools,
                mock_builder=mock_builder,
            )
        raise RuntimeError(
            f"{self.name} key is set but no compatible HTTP adapter is configured in this portfolio build. "
            "Use openai provider with an OpenAI-compatible base URL, or mock."
        )


class AnthropicProvider(_KeyGatedProvider):
    name = "anthropic"
    key_env = "ANTHROPIC_API_KEY"
    model_id = "claude-3-5-haiku"


class GeminiProvider(_KeyGatedProvider):
    name = "gemini"
    key_env = "GEMINI_API_KEY"
    model_id = "gemini-1.5-flash"


class GroqProvider(_KeyGatedProvider):
    name = "groq"
    key_env = "GROQ_API_KEY"
    model_id = "llama-3.1-8b-instant"

    def __init__(self, api_key: str | None, **kwargs: Any):
        kwargs.setdefault("base_url", "https://api.groq.com/openai/v1")
        kwargs.setdefault("model", self.model_id)
        super().__init__(api_key, **kwargs)


class OllamaProvider(_KeyGatedProvider):
    name = "ollama"
    key_env = "OLLAMA_BASE_URL"
    model_id = "llama3.1"

    def __init__(self, base_url: str = "http://127.0.0.1:11434", **kwargs: Any):
        # Ollama needs no key; treat reachable base as "key"
        kwargs.setdefault("base_url", base_url.rstrip("/") + "/v1")
        kwargs.setdefault("model", self.model_id)
        super().__init__(api_key="ollama-local", **kwargs)
        self.key_env = "OLLAMA_BASE_URL (local server)"

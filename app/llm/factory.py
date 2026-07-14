from __future__ import annotations

from app.core.config import Settings
from app.llm.base import LLMProvider
from app.llm.mock import MockProvider
from app.llm.openai_compat import OpenAICompatibleProvider
from app.llm.stubs import AnthropicProvider, GeminiProvider, GroqProvider, OllamaProvider


def get_provider(settings: Settings) -> LLMProvider:
    name = (settings.llm_provider or "mock").lower()
    if name in ("openai",) and settings.openai_api_key:
        return OpenAICompatibleProvider(
            settings.openai_api_key, settings.openai_base_url, settings.openai_model
        )
    if name == "anthropic":
        return AnthropicProvider(settings.anthropic_api_key)
    if name == "gemini":
        return GeminiProvider(settings.gemini_api_key)
    if name == "groq":
        return GroqProvider(settings.groq_api_key)
    if name == "ollama":
        return OllamaProvider(settings.ollama_base_url)
    # openai without key → mock
    return MockProvider()

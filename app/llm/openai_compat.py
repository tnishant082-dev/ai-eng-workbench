from __future__ import annotations

import json
import time
from typing import Any

import httpx

from app.llm.base import CompletionResult, ToolCall, estimate_cost, estimate_tokens
from app.llm.mock import MockProvider


class OpenAICompatibleProvider:
    name = "openai"

    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._fallback = MockProvider()

    def models(self) -> list[dict[str, str]]:
        return [
            {
                "id": self.model,
                "provider": "openai",
                "description": f"OpenAI-compatible via {self.base_url}",
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
            raise RuntimeError("OPENAI_API_KEY not set — use mock provider or set the key")
        t0 = time.perf_counter()
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if tools:
            payload["tools"] = tools
        try:
            with httpx.Client(timeout=60.0) as client:
                r = client.post(url, headers=headers, json=payload)
                r.raise_for_status()
                data = r.json()
        except Exception as exc:  # noqa: BLE001
            if mock_builder is None:
                raise
            fb = self._fallback.complete(
                system=system, user=user, json_mode=json_mode, tools=tools, mock_builder=mock_builder
            )
            fb.text = fb.text + f"\n\n_Fell back to mock after OpenAI error: {type(exc).__name__}_"
            fb.raw = {"fallback_error": str(exc)}
            return fb

        msg = data["choices"][0]["message"]
        text = msg.get("content") or ""
        tool_calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            args = fn.get("arguments") or "{}"
            try:
                parsed = json.loads(args) if isinstance(args, str) else args
            except json.JSONDecodeError:
                parsed = {"raw": args}
            tool_calls.append(ToolCall(name=fn.get("name", ""), arguments=parsed, id=tc.get("id", "")))

        usage = data.get("usage") or {}
        pt = int(usage.get("prompt_tokens") or estimate_tokens(system + user))
        ct = int(usage.get("completion_tokens") or estimate_tokens(text))
        structured = None
        if json_mode:
            try:
                structured = json.loads(text)
            except json.JSONDecodeError:
                structured = None
        return CompletionResult(
            text=text,
            model=self.model,
            provider="openai",
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            prompt_tokens=pt,
            completion_tokens=ct,
            cost_usd=estimate_cost(self.model, pt, ct),
            tool_calls=tool_calls,
            structured=structured,
            raw={"usage": usage},
        )

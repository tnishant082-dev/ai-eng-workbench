from __future__ import annotations

import json
import re
import time
from typing import Any

from app.llm.base import CompletionResult, ToolCall, estimate_cost, estimate_tokens


class MockProvider:
    name = "mock"

    def models(self) -> list[dict[str, str]]:
        return [
            {
                "id": "mock-workbench",
                "provider": "mock",
                "description": "Deterministic offline composer — no API key required",
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
        t0 = time.perf_counter()
        tool_calls: list[ToolCall] = []
        structured = None

        if mock_builder:
            text = mock_builder()
        elif json_mode:
            structured = {"status": "ok", "summary": user[:200], "provider": "mock"}
            text = json.dumps(structured)
        else:
            text = f"[mock] {user[:400]}"

        # Simple tool-call heuristic for agent loops
        if tools and "TOOL:" in user.upper():
            m = re.search(r"TOOL:\s*(\w+)\((.*)\)", user, re.IGNORECASE | re.DOTALL)
            if m:
                name = m.group(1)
                tool_calls.append(ToolCall(name=name, arguments={"raw": m.group(2)}, id="mock-1"))

        pt = estimate_tokens(system + user)
        ct = estimate_tokens(text)
        latency = (time.perf_counter() - t0) * 1000
        return CompletionResult(
            text=text,
            model="mock-workbench",
            provider="mock",
            latency_ms=round(latency, 2),
            prompt_tokens=pt,
            completion_tokens=ct,
            cost_usd=estimate_cost("mock-workbench", pt, ct),
            tool_calls=tool_calls,
            structured=structured,
            raw={"purpose": "mock"},
        )

from __future__ import annotations

import re

from app.rag.index import Hit


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{3,}", text.lower()))


def faithfulness_heuristic(answer: str, hits: list[Hit]) -> float:
    """Fraction of answer content tokens supported by retrieved context."""
    ans = _tokens(answer)
    if not ans:
        return 0.0
    ctx = set()
    for h in hits:
        ctx |= _tokens(h.chunk.text)
        ctx |= _tokens(h.chunk.title)
    # ignore boilerplate mock phrases
    noise = {"based", "local", "corpus", "mock", "gateway", "grounded", "via", "retrieval"}
    ans -= noise
    if not ans:
        return 1.0
    return round(len(ans & ctx) / len(ans), 4)


def confidence_score(hits: list[Hit], faithfulness: float) -> float:
    if not hits:
        return 0.0
    top = hits[0].score
    spread = hits[0].score - hits[-1].score if len(hits) > 1 else hits[0].score
    raw = 0.45 * min(top, 1.0) + 0.35 * faithfulness + 0.20 * min(spread, 1.0)
    return round(max(0.0, min(1.0, raw)), 4)


def hallucination_flag(faithfulness: float, confidence: float) -> bool:
    return faithfulness < 0.35 or confidence < 0.3

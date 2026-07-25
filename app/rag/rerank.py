from __future__ import annotations

import re
from typing import Any

from app.rag.index import Hit


def _overlap(query: str, text: str) -> float:
    q = set(re.findall(r"[a-z0-9]+", query.lower()))
    t = set(re.findall(r"[a-z0-9]+", text.lower()))
    if not q:
        return 0.0
    return len(q & t) / len(q)


def lightweight_lexical_rerank(query: str, hits: list[Hit], top_k: int = 5) -> tuple[list[Hit], str]:
    """Labeled lightweight scorer — no cross-encoder dependency."""
    scored: list[Hit] = []
    for h in hits:
        lex = _overlap(query, h.chunk.title + " " + h.chunk.text)
        title_boost = 0.15 if _overlap(query, h.chunk.title) > 0.3 else 0.0
        length_pen = 0.05 if len(h.chunk.text) < 80 else 0.0
        parent_boost = 0.05 if h.chunk.parent_text else 0.0
        rr = 0.50 * h.score + 0.30 * lex + 0.10 * h.dense + title_boost + parent_boost - length_pen
        scored.append(Hit(chunk=h.chunk, score=h.score, bm25=h.bm25, tfidf=h.tfidf, dense=h.dense, rrf=h.rrf, rerank=rr))
    scored.sort(key=lambda x: x.rerank, reverse=True)
    out = []
    for h in scored[:top_k]:
        out.append(Hit(chunk=h.chunk, score=round(h.rerank, 4), bm25=h.bm25, tfidf=h.tfidf, dense=h.dense, rrf=h.rrf, rerank=h.rerank))
    return out, "lightweight_lexical"


def cross_encoder_rerank(query: str, hits: list[Hit], top_k: int = 5):
    try:
        from sentence_transformers import CrossEncoder
    except Exception:
        return None
    try:
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs = [(query, h.chunk.title + " " + h.chunk.text) for h in hits]
        scores = model.predict(pairs)
        ranked = sorted(zip(hits, scores), key=lambda x: float(x[1]), reverse=True)
        out = []
        for h, sc in ranked[:top_k]:
            out.append(Hit(chunk=h.chunk, score=round(float(sc), 4), bm25=h.bm25, tfidf=h.tfidf, dense=h.dense, rrf=h.rrf, rerank=float(sc)))
        return out, "cross_encoder"
    except Exception:
        return None


def rerank(query: str, hits: list[Hit], top_k: int = 5, prefer_cross_encoder: bool = True) -> list[Hit]:
    if prefer_cross_encoder:
        ce = cross_encoder_rerank(query, hits, top_k=top_k)
        if ce is not None:
            return ce[0]
    out, _ = lightweight_lexical_rerank(query, hits, top_k=top_k)
    return out


def rerank_with_meta(query: str, hits: list[Hit], top_k: int = 5) -> tuple[list[Hit], dict[str, Any]]:
    ce = cross_encoder_rerank(query, hits, top_k=top_k)
    if ce is not None:
        return ce[0], {"reranker": ce[1]}
    out, name = lightweight_lexical_rerank(query, hits, top_k=top_k)
    return out, {"reranker": name}

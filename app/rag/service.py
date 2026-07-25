from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from app.llm.base import LLMProvider, estimate_tokens
from app.llm.prompts import load_prompt
from app.rag.confidence import confidence_score, faithfulness_heuristic, hallucination_flag
from app.rag.expand import multi_queries
from app.rag.index import CorpusIndex, Hit
from app.rag.rerank import rerank_with_meta


def _short(text: str, n: int = 220) -> str:
    flat = " ".join(text.split())
    if len(flat) <= n:
        return flat
    return flat[: n - 1].rstrip() + "…"


class RagService:
    def __init__(self, index: CorpusIndex, provider: LLMProvider, prompts_dir: Path):
        self.index = index
        self.provider = provider
        self.prompts_dir = prompts_dir

    def query(self, question: str, top_k: int = 5, metadata_filter: dict | None = None, use_multi_query: bool = True, use_rrf: bool = True) -> dict[str, Any]:
        t0 = time.perf_counter()
        events: list[dict[str, Any]] = []
        queries = multi_queries(question) if use_multi_query else [question]
        events.append({"type": "query_expand", "queries": queries})
        info = self.index.info()
        events.append({"type": "index_info", **info})
        pooled: dict[str, Hit] = {}
        for q in queries:
            events.append({"type": "retrieve_start", "query": q, "top_k": top_k, "use_rrf": use_rrf})
            for h in self.index.search(q, top_k=top_k, metadata_filter=metadata_filter, use_rrf=use_rrf):
                prev = pooled.get(h.chunk.doc_id)
                if prev is None or h.score > prev.score:
                    pooled[h.chunk.doc_id] = h
            events.append({"type": "retrieve_end", "pool_size": len(pooled)})
        candidates = sorted(pooled.values(), key=lambda h: h.score, reverse=True)
        hits, rr_meta = rerank_with_meta(question, candidates, top_k=top_k)
        events.append({"type": "rerank", "hit_count": len(hits), "doc_ids": [h.chunk.doc_id for h in hits], **rr_meta})
        system = load_prompt(self.prompts_dir, "rag_answer_v1.txt") or "Answer using only the context. Cite [doc_id]. If unsure, say so."
        context = self._format_context(hits)
        user = f"Question: {question}\n\nContext:\n{context}"

        def mock_builder() -> str:
            return self._mock_answer(question, hits, rr_meta.get("reranker", "lightweight_lexical"))

        events.append({"type": "llm_call_start", "provider": self.provider.name})
        result = self.provider.complete(system=system, user=user, mock_builder=mock_builder)
        events.append({"type": "llm_call_end", "model": result.model, "provider": result.provider, "tokens": result.tokens_estimate, "cost_usd": result.cost_usd})
        faith = faithfulness_heuristic(result.text, hits)
        conf = confidence_score(hits, faith)
        hallu = hallucination_flag(faith, conf)
        events.append({"type": "confidence", "faithfulness": faith, "confidence": conf, "hallucination_risk": hallu})
        citations = [self._citation(h) for h in hits]
        latency = (time.perf_counter() - t0) * 1000
        return {
            "answer": result.text,
            "citations": citations,
            "model": result.model,
            "provider": result.provider,
            "gateway": result.provider,
            "latency_ms": round(latency, 2),
            "tokens_estimate": result.tokens_estimate or estimate_tokens(system + user + result.text),
            "cost_usd": result.cost_usd,
            "faithfulness": faith,
            "confidence": conf,
            "hallucination_risk": hallu,
            "events": events,
            "hits": hits,
            "retrieval": {
                "dense_backend": info.get("dense_backend"),
                "vector_backend": info.get("vector_backend"),
                "reranker": rr_meta.get("reranker"),
                "fusion": "rrf" if use_rrf else "weighted",
            },
        }

    def _format_context(self, hits: list[Hit]) -> str:
        if not hits:
            return "(no retrieved chunks)"
        blocks = []
        for h in hits:
            body = h.chunk.parent_text or h.chunk.text
            blocks.append(f"[{h.chunk.doc_id}] {h.chunk.title}\n{body}")
        return "\n\n---\n\n".join(blocks)

    def _mock_answer(self, question: str, hits: list[Hit], reranker: str) -> str:
        if not hits:
            return "I could not find matching documents in the local corpus. Try rephrasing or ingesting docs under data/corpus/."
        info = self.index.info()
        lines = [f"Based on the local corpus for: _{question.strip()}_", ""]
        for h in hits[:3]:
            snippet = h.chunk.parent_text or h.chunk.text
            lines.append(f"- [{h.chunk.doc_id}] {_short(snippet, 180)}")
        lines.append("")
        lines.append(
            f"_Grounded via hybrid BM25+dense ({info.get('dense_backend')}) with RRF; "
            f"reranker={reranker}; vector_store={info.get('vector_backend')}; mock provider (offline)._"
        )
        return "\n".join(lines)

    @staticmethod
    def _citation(h: Hit) -> dict[str, Any]:
        text = h.chunk.parent_text or h.chunk.text
        return {
            "doc_id": h.chunk.doc_id,
            "title": h.chunk.title,
            "path": h.chunk.path,
            "score": round(h.score, 4),
            "bm25": round(h.bm25, 4),
            "dense": round(h.dense, 4),
            "rrf": round(h.rrf, 4),
            "snippet": _short(text, 200),
            "metadata": h.chunk.metadata,
            "parent_id": h.chunk.parent_id,
        }

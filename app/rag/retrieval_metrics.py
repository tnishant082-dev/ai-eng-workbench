"""Retrieval ranking metrics on a golden set: Recall@K, Precision@K, MRR, NDCG."""
from __future__ import annotations

import math
from typing import Any


def recall_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    if not relevant:
        return 1.0
    top = set(retrieved[:k])
    return len(relevant & top) / len(relevant)


def precision_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    top = retrieved[:k]
    if not top:
        return 0.0
    return len(relevant & set(top)) / len(top)


def mrr(relevant: set[str], retrieved: list[str]) -> float:
    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def dcg_at_k(relevances: list[float], k: int) -> float:
    s = 0.0
    for i, rel in enumerate(relevances[:k], start=1):
        s += (2**rel - 1) / math.log2(i + 1)
    return s


def ndcg_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    rels = [1.0 if d in relevant else 0.0 for d in retrieved[:k]]
    ideal = sorted(rels, reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(rels, k) / idcg


def evaluate_retrieval(
    cases: list[dict[str, Any]],
    retrieve_fn,
    k: int = 5,
) -> dict[str, Any]:
    """
    cases: [{id, query, relevant_doc_ids: [...] }]
    retrieve_fn(query) -> list[str] doc/chunk ids (stems ok)
    """
    rows = []
    for case in cases:
        q = case.get("query") or case.get("question") or ""
        relevant_raw = case.get("relevant_doc_ids") or case.get("expected_doc_ids") or case.get("must_cite") or []
        if isinstance(relevant_raw, str):
            relevant_raw = [relevant_raw]
        relevant = {str(x).split("#")[0] for x in relevant_raw}
        retrieved = retrieve_fn(q)
        retrieved_stems = [str(x).split("#")[0] for x in retrieved]
        row = {
            "id": case.get("id", ""),
            "recall_at_k": round(recall_at_k(relevant, retrieved_stems, k), 4),
            "precision_at_k": round(precision_at_k(relevant, retrieved_stems, k), 4),
            "mrr": round(mrr(relevant, retrieved_stems), 4),
            "ndcg_at_k": round(ndcg_at_k(relevant, retrieved_stems, k), 4),
        }
        rows.append(row)
    n = len(rows) or 1
    summary = {
        "k": k,
        "n_cases": len(rows),
        "avg_recall_at_k": round(sum(r["recall_at_k"] for r in rows) / n, 4) if rows else 0.0,
        "avg_precision_at_k": round(sum(r["precision_at_k"] for r in rows) / n, 4) if rows else 0.0,
        "avg_mrr": round(sum(r["mrr"] for r in rows) / n, 4) if rows else 0.0,
        "avg_ndcg_at_k": round(sum(r["ndcg_at_k"] for r in rows) / n, 4) if rows else 0.0,
    }
    return {"summary": summary, "cases": rows}

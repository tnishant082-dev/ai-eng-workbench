from __future__ import annotations

import re
from typing import Any


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{3,}", (text or "").lower()))


def faithfulness(answer: str, contexts: list[str]) -> float:
    ans = _tokens(answer)
    noise = {"based", "local", "corpus", "mock", "gateway", "grounded", "via", "retrieval", "provider", "hybrid", "dense", "reranker", "offline"}
    ans -= noise
    if not ans:
        return 1.0
    ctx = set()
    for c in contexts:
        ctx |= _tokens(c)
    return round(len(ans & ctx) / len(ans), 4)


def answer_relevance(question: str, answer: str) -> float:
    q, a = _tokens(question), _tokens(answer)
    if not q:
        return 0.0
    return round(len(q & a) / len(q), 4)


relevance = answer_relevance


def context_precision(relevant_ctx: list[str], retrieved_ctx: list[str]) -> float:
    if not retrieved_ctx:
        return 0.0
    rel_tok = set()
    for c in relevant_ctx:
        rel_tok |= _tokens(c)
    if not rel_tok:
        return 1.0
    hits = sum(1 for c in retrieved_ctx if _tokens(c) & rel_tok)
    return round(hits / len(retrieved_ctx), 4)


def context_recall(relevant_ctx: list[str], retrieved_ctx: list[str]) -> float:
    if not relevant_ctx:
        return 1.0
    ret_tok = set()
    for c in retrieved_ctx:
        ret_tok |= _tokens(c)
    hits = sum(1 for c in relevant_ctx if _tokens(c) & ret_tok)
    return round(hits / len(relevant_ctx), 4)


def citation_precision_recall(expected_docs: list[str], cited_docs: list[str]) -> tuple[float, float]:
    exp, cit = set(expected_docs or []), set(cited_docs or [])
    if not exp and not cit:
        return 1.0, 1.0
    if not cit:
        return 0.0, 0.0 if exp else 1.0
    if not exp:
        return 1.0, 1.0
    tp = len(exp & cit)
    return round(tp / len(cit), 4), round(tp / len(exp), 4)


def llm_judge_stub(question: str, answer: str, contexts: list[str]) -> dict[str, Any]:
    faith = faithfulness(answer, contexts)
    rel = answer_relevance(question, answer)
    score = round(0.6 * faith + 0.4 * rel, 4)
    return {"judge": "heuristic_stub", "score": score, "rationale": f"LLM-judge stub. faithfulness={faith} relevance={rel}."}


def score_case(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    contexts = [c.get("snippet") or "" for c in result.get("citations") or []]
    faith = faithfulness(result.get("answer", ""), contexts)
    rel = answer_relevance(case.get("question", ""), result.get("answer", ""))
    cited = [c.get("doc_id", "").split("#")[0] for c in result.get("citations") or []]
    raw_exp = case.get("expected_doc_ids") or case.get("must_cite") or case.get("expected_docs") or []
    if isinstance(raw_exp, bool):
        expected = []
    elif isinstance(raw_exp, str):
        expected = [raw_exp]
    else:
        expected = list(raw_exp)
    expected_stems = [str(e).split("#")[0] for e in expected]
    relevant_ctx = case.get("relevant_contexts") or []
    if not relevant_ctx and expected_stems:
        relevant_ctx = [c.get("snippet") or "" for c in result.get("citations") or [] if (c.get("doc_id") or "").split("#")[0] in expected_stems]
    ctx_prec = context_precision(relevant_ctx, contexts) if contexts else 0.0
    ctx_rec = context_recall(relevant_ctx or contexts, contexts) if contexts else 0.0
    prec, rec = citation_precision_recall(expected_stems, cited)
    judge = llm_judge_stub(case.get("question", ""), result.get("answer", ""), contexts)
    passed = faith >= 0.25 and rel >= 0.15 and (not expected_stems or rec >= 0.3)
    if case.get("expect_nonempty", True) and not (result.get("answer") or "").strip():
        passed = False
    return {
        "id": case.get("id", ""),
        "passed": passed,
        "faithfulness": faith,
        "relevance": rel,
        "context_precision": ctx_prec,
        "context_recall": ctx_rec,
        "citation_precision": prec,
        "citation_recall": rec,
        "judge_score": judge["score"],
        "judge": judge["judge"],
        "latency_ms": float(result.get("latency_ms") or 0),
        "cost_usd": float(result.get("cost_usd") or 0),
        "answer_preview": (result.get("answer") or "")[:240],
    }


def score_agent_run(task: str, result: dict[str, Any], expected_tools: list[str] | None = None) -> dict[str, Any]:
    tools = result.get("tools_used") or []
    expected_tools = expected_tools or []
    tool_acc = (len(set(tools) & set(expected_tools)) / len(expected_tools)) if expected_tools else (1.0 if tools else 0.0)
    status = result.get("status", "")
    success = status in ("ok", "ok_with_recovery") and bool(result.get("answer"))
    return {
        "task_success": bool(success),
        "tool_selection_accuracy": round(tool_acc, 4),
        "latency_ms": float(result.get("latency_ms") or 0),
        "tools_used": tools,
        "status": status,
    }

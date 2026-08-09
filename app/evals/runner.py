from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from app.evals.metrics import score_agent_run, score_case
from app.evals.report import write_eval_report
from app.rag.service import RagService


def load_golden(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if isinstance(data, list):
        return data
    return data.get("cases") or data.get("items") or []


def run_evals(rag: RagService, golden_path: Path, limit: int | None = None, reports_dir: Path | None = None) -> dict[str, Any]:
    cases = load_golden(golden_path)
    if limit:
        cases = cases[:limit]
    t0 = time.perf_counter()
    results = []
    for case in cases:
        q = case.get("question") or case.get("query") or ""
        out = rag.query(q, top_k=case.get("top_k", 4), use_multi_query=case.get("multi_query", True))
        scored = score_case(case, out)
        results.append(scored)

    wall = (time.perf_counter() - t0) * 1000
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    summary = {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "avg_faithfulness": round(sum(r["faithfulness"] for r in results) / total, 4) if total else 0.0,
        "avg_relevance": round(sum(r["relevance"] for r in results) / total, 4) if total else 0.0,
        "avg_citation_precision": round(sum(r["citation_precision"] for r in results) / total, 4) if total else 0.0,
        "avg_citation_recall": round(sum(r["citation_recall"] for r in results) / total, 4) if total else 0.0,
        "avg_context_precision": round(sum(r.get("context_precision", 0) for r in results) / total, 4) if total else 0.0,
        "avg_context_recall": round(sum(r.get("context_recall", 0) for r in results) / total, 4) if total else 0.0,
        "avg_judge_score": round(sum(r.get("judge_score", 0) for r in results) / total, 4) if total else 0.0,
        "avg_latency_ms": round(sum(r["latency_ms"] for r in results) / total, 2) if total else 0.0,
        "total_cost_usd": round(sum(r["cost_usd"] for r in results), 6),
        "wall_ms": round(wall, 2),
    }
    payload = {"summary": summary, "cases": results}
    if reports_dir is not None:
        payload["report_paths"] = write_eval_report(payload, reports_dir)
    return payload


def run_agent_evals(workflow, cases: list[dict[str, Any]]) -> dict[str, Any]:
    results = []
    for case in cases:
        task = case.get("task") or case.get("question") or ""
        out = workflow.run(task)
        scored = score_agent_run(task, out, expected_tools=case.get("expected_tools"))
        scored["id"] = case.get("id", "")
        results.append(scored)
    n = len(results) or 1
    summary = {
        "total": len(results),
        "task_success_rate": round(sum(1 for r in results if r["task_success"]) / n, 4) if results else 0.0,
        "avg_tool_selection_accuracy": round(sum(r["tool_selection_accuracy"] for r in results) / n, 4) if results else 0.0,
        "avg_latency_ms": round(sum(r["latency_ms"] for r in results) / n, 2) if results else 0.0,
    }
    return {"summary": summary, "cases": results}

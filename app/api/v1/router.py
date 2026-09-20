from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Body, HTTPException, Query, Request

from app.api.schemas import (
    AgentRunRequest,
    AgentRunResponse,
    CompareRequest,
    EvalCaseResult,
    EvalRunRequest,
    EvalRunResponse,
    GatewayModelInfo,
    IngestRequest,
    MLPredictRequest,
    MLTrainRequest,
    RagQueryRequest,
    RagQueryResponse,
    TraceDetail,
    TraceSummary,
)
from app.core.jobs import JobQueue
from app.dataeng.etl import run_etl
from app.dataeng.quality import corpus_quality, tabular_quality
from app.dataeng.validate import validate_corpus_frames
from app.evals.runner import run_agent_evals, run_evals
from app.ml.pipeline import FEATURE_COLS, TARGET, check_drift, list_registry, predict, train_model
from app.observability.analytics import trace_analytics
from app.observability.otel import configure_otel_console_exporter
from app.rag.retrieval_metrics import evaluate_retrieval

api_router = APIRouter(prefix="/api/v1")


def _state(request: Request) -> dict[str, Any]:
    return request.app.state.wb


@api_router.post("/rag/query", response_model=RagQueryResponse)
def rag_query(body: RagQueryRequest, request: Request) -> RagQueryResponse:
    st = _state(request)
    settings = st["settings"]
    top_k = body.top_k or settings.rag_top_k
    cache = st["cache"]
    ck = cache.key("rag", body.question, top_k, body.metadata_filter, body.use_multi_query, body.use_rrf)
    cached = cache.get(ck)
    if cached:
        out = cached
    else:
        out = st["rag"].query(
            body.question,
            top_k=top_k,
            metadata_filter=body.metadata_filter,
            use_multi_query=body.use_multi_query,
            use_rrf=body.use_rrf,
        )
        cache.set(ck, out)
    trace_id = st["store"].add(
        kind="rag",
        model=out["model"],
        gateway=out["provider"],
        latency_ms=out["latency_ms"],
        tokens_estimate=out["tokens_estimate"],
        cost_usd=out.get("cost_usd", 0),
        status="ok",
        request={"question": body.question, "top_k": top_k},
        response={
            "answer": out["answer"],
            "citations": out["citations"],
            "confidence": out.get("confidence"),
            "faithfulness": out.get("faithfulness"),
        },
        events=out["events"],
    )
    return RagQueryResponse(
        answer=out["answer"],
        citations=out["citations"],
        model=out["model"],
        provider=out["provider"],
        gateway=out["gateway"],
        latency_ms=out["latency_ms"],
        tokens_estimate=out["tokens_estimate"],
        cost_usd=out.get("cost_usd", 0),
        faithfulness=out.get("faithfulness", 0),
        confidence=out.get("confidence", 0),
        hallucination_risk=out.get("hallucination_risk", False),
        retrieval=out.get("retrieval"),
        trace_id=trace_id,
    )


@api_router.post("/rag/ingest")
def rag_ingest(body: IngestRequest, request: Request) -> dict[str, Any]:
    st = _state(request)
    n = st["index"].ingest_text(body.name, body.text, body.metadata)
    st["cache"].clear()
    return {"chunks": n, "docs": st["index"].doc_count, "status": "rebuilt"}


@api_router.post("/agents/run", response_model=AgentRunResponse)
def agents_run(body: AgentRunRequest, request: Request) -> AgentRunResponse:
    st = _state(request)
    settings = st["settings"]
    max_steps = body.max_steps or settings.agent_max_steps
    out = st["agents"].run(body.task, max_steps=max_steps, approve=body.approve)
    status = out.get("status", "ok")
    trace_id = st["store"].add(
        kind="agent",
        model=out.get("model", "n/a"),
        gateway=out.get("provider", ""),
        latency_ms=out["latency_ms"],
        tokens_estimate=out.get("tokens_estimate", 0),
        cost_usd=out.get("cost_usd", 0),
        status=status,
        request={"task": body.task, "max_steps": max_steps, "approve": body.approve},
        response={"answer": out["answer"], "tools_used": out.get("tools_used"), "plan": out.get("plan")},
        events=out.get("events") or [],
    )
    return AgentRunResponse(
        status=status,
        answer=out["answer"],
        plan=out.get("plan"),
        tools_used=out.get("tools_used") or [],
        events=out.get("events") or [],
        citations=out.get("citations") or [],
        model=out.get("model", "n/a"),
        provider=out.get("provider", ""),
        gateway=out.get("gateway", ""),
        latency_ms=out["latency_ms"],
        tokens_estimate=out.get("tokens_estimate", 0),
        cost_usd=out.get("cost_usd", 0),
        memory=out.get("memory"),
        failures=out.get("failures"),
        trace_id=trace_id,
    )


# Back-compat alias
@api_router.post("/agent/run", response_model=AgentRunResponse)
def agent_run_alias(body: AgentRunRequest, request: Request) -> AgentRunResponse:
    return agents_run(body, request)


@api_router.post("/evals/run", response_model=EvalRunResponse)
def evals_run(body: EvalRunRequest | None, request: Request) -> EvalRunResponse:
    body = body or EvalRunRequest()
    st = _state(request)
    settings = st["settings"]
    reports = settings.reports_dir if getattr(body, "write_report", True) else None
    out = run_evals(st["rag"], settings.golden_path, limit=body.limit, reports_dir=reports)
    trace_id = st["store"].add(
        kind="eval",
        model=st["gateway"].mode,
        gateway=st["gateway"].mode,
        latency_ms=out["summary"]["wall_ms"],
        tokens_estimate=0,
        cost_usd=out["summary"].get("total_cost_usd", 0),
        status="ok" if out["summary"]["pass_rate"] >= 0.7 else "degraded",
        request={"limit": body.limit},
        response={"summary": out["summary"]},
        events=[{"type": "eval_case", **c} for c in out["cases"]],
    )
    cases = [EvalCaseResult(**c) for c in out["cases"]]
    return EvalRunResponse(summary=out["summary"], cases=cases, report_paths=out.get("report_paths"), trace_id=trace_id)


@api_router.get("/traces", response_model=list[TraceSummary])
def list_traces(request: Request, limit: int = Query(default=50, ge=1, le=200)) -> list[TraceSummary]:
    rows = _state(request)["store"].list(limit=limit)
    return [TraceSummary(**r) for r in rows]


@api_router.get("/traces/analytics", tags=["traces"])
def traces_analytics(request: Request, limit: int = 200) -> dict[str, Any]:
    return trace_analytics(_state(request)["store"], limit=limit)

@api_router.get("/traces/{trace_id}", response_model=TraceDetail)
def get_trace(trace_id: int, request: Request) -> TraceDetail:
    row = _state(request)["store"].get(trace_id)
    if not row:
        raise HTTPException(status_code=404, detail="trace not found")
    return TraceDetail(**row)


@api_router.get("/traces/compare/{a_id}/{b_id}")
def compare_traces(a_id: int, b_id: int, request: Request) -> dict[str, Any]:
    out = _state(request)["store"].compare(a_id, b_id)
    if not out:
        raise HTTPException(status_code=404, detail="one or both traces not found")
    return out


@api_router.get("/gateway/models", response_model=list[GatewayModelInfo])
def gateway_models(request: Request) -> list[GatewayModelInfo]:
    models = []
    for m in _state(request)["gateway"].models():
        models.append(
            GatewayModelInfo(
                id=m["id"],
                provider=m.get("provider") or m.get("gateway"),
                gateway=m.get("gateway") or m.get("provider"),
                description=m.get("description", ""),
            )
        )
    return models


@api_router.post("/gateway/compare")
def gateway_compare(body: CompareRequest, request: Request) -> dict[str, Any]:
    rows = _state(request)["gateway"].compare_estimate(body.prompt, body.completion_chars)
    return {"estimates": rows}


@api_router.post("/ml/train", tags=["ml"])
def ml_train(
    request: Request, body: Annotated[MLTrainRequest | None, Body()] = None
) -> dict[str, Any]:
    body = body or MLTrainRequest()
    st = _state(request)
    settings = st["settings"]
    queue: JobQueue = st["jobs"]
    job = queue.submit_sync(
        "ml_train",
        lambda: train_model(settings.dataset_path, settings.experiments_dir, settings.registry_dir, tune=body.tune, version=body.version),
    )
    if job.status.value == "failed":
        raise HTTPException(status_code=500, detail=job.error)
    return {"job_id": job.id, "status": job.status.value, "result": job.result}


@api_router.post("/ml/predict", tags=["ml"])
def ml_predict(body: MLPredictRequest, request: Request) -> dict[str, Any]:
    settings = _state(request)["settings"]
    try:
        return predict(settings.registry_dir, body.records, mode=getattr(body, "mode", "batch"))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@api_router.get("/ml/registry")
def ml_registry(request: Request) -> dict[str, Any]:
    settings = _state(request)["settings"]
    return {"models": list_registry(settings.registry_dir)}


@api_router.post("/ml/drift")
def ml_drift(
    request: Request, body: Annotated[MLPredictRequest | None, Body()] = None
) -> dict[str, Any]:
    settings = _state(request)["settings"]
    sample = body.records if body and body.records else None
    return check_drift(settings.dataset_path, sample)


@api_router.post("/dataeng/etl")
def dataeng_etl(request: Request) -> dict[str, Any]:
    settings = _state(request)["settings"]
    return run_etl(
        settings.corpus_dir,
        settings.chunk_size,
        settings.chunk_overlap,
        settings.catalog_path,
        settings.lineage_path,
        dataset_path=settings.dataset_path,
    )


@api_router.get("/dataeng/validate")
def dataeng_validate(request: Request) -> dict[str, Any]:
    settings = _state(request)["settings"]
    return validate_corpus_frames(settings.corpus_dir)


@api_router.get("/jobs")
def list_jobs(request: Request) -> list[dict[str, Any]]:
    jobs = _state(request)["jobs"].list()
    return [
        {
            "id": j.id,
            "name": j.name,
            "status": j.status.value,
            "error": j.error,
            "created_at": j.created_at,
            "finished_at": j.finished_at,
        }
        for j in jobs
    ]


@api_router.get("/prompts")
def list_prompts_api(request: Request) -> list[dict[str, str]]:
    from app.llm.prompts import list_prompts

    return list_prompts(_state(request)["settings"].prompts_dir)


@api_router.get("/rag/index", tags=["rag"])
def rag_index_info(request: Request) -> dict[str, Any]:
    return _state(request)["index"].info()


@api_router.post("/rag/retrieval-metrics", tags=["rag"])
def rag_retrieval_metrics(request: Request, k: int = 5, limit: int = 20) -> dict[str, Any]:
    st = _state(request)
    from app.evals.runner import load_golden
    cases = load_golden(st["settings"].golden_path)[:limit]
    def retrieve_fn(q: str):
        return [h.chunk.doc_id for h in st["index"].search(q, top_k=k)]
    return evaluate_retrieval(cases, retrieve_fn, k=k)


@api_router.post("/evals/agents", tags=["evals"])
def evals_agents(request: Request) -> dict[str, Any]:
    st = _state(request)
    cases = [
        {"id": "a1", "task": "Search the corpus for faithfulness evals", "expected_tools": ["corpus_search"]},
        {"id": "a2", "task": "Calculate 12 * (3 + 4)", "expected_tools": ["calculator"]},
    ]
    return run_agent_evals(st["agents"], cases)



@api_router.get("/dataeng/quality", tags=["dataeng"])
def dataeng_quality(request: Request) -> dict[str, Any]:
    settings = _state(request)["settings"]
    return {"corpus": corpus_quality(settings.corpus_dir), "tabular": tabular_quality(settings.dataset_path, FEATURE_COLS + [TARGET])}


@api_router.get("/observability/status", tags=["observability"])
def observability_status(request: Request) -> dict[str, Any]:
    st = _state(request)
    return {"langsmith": st["langsmith"].info(), "otel": configure_otel_console_exporter(), "analytics": trace_analytics(st["store"], limit=50)}

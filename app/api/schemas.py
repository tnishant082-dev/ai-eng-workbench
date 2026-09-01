from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    version: str
    corpus_docs: int
    chunks: int
    gateway: str
    provider: str
    index_ready: bool
    dense_backend: str | None = None
    vector_backend: str | None = None


class RagQueryRequest(BaseModel):
    question: str
    top_k: int | None = None
    metadata_filter: dict[str, str] | None = None
    use_multi_query: bool = True
    use_rrf: bool = True


class RagQueryResponse(BaseModel):
    answer: str
    citations: list[dict[str, Any]]
    model: str
    provider: str
    gateway: str
    latency_ms: float
    tokens_estimate: int
    cost_usd: float = 0.0
    faithfulness: float = 0.0
    confidence: float = 0.0
    hallucination_risk: bool = False
    retrieval: dict[str, Any] | None = None
    trace_id: int | None = None


class AgentRunRequest(BaseModel):
    task: str
    max_steps: int | None = None
    approve: bool = False


class AgentRunResponse(BaseModel):
    status: str = "ok"
    answer: str
    plan: dict[str, Any] | None = None
    tools_used: list[str]
    events: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    model: str
    provider: str
    gateway: str
    latency_ms: float
    tokens_estimate: int
    cost_usd: float = 0.0
    memory: list[dict[str, Any]] | None = None
    failures: list[str] | None = None
    trace_id: int | None = None


class EvalRunRequest(BaseModel):
    limit: int | None = None
    write_report: bool = True


class EvalCaseResult(BaseModel):
    id: str
    passed: bool
    faithfulness: float
    relevance: float
    citation_precision: float
    citation_recall: float
    context_precision: float = 0.0
    context_recall: float = 0.0
    judge_score: float = 0.0
    judge: str = "heuristic_stub"
    latency_ms: float
    cost_usd: float = 0.0
    answer_preview: str = ""


class EvalRunResponse(BaseModel):
    summary: dict[str, Any]
    cases: list[EvalCaseResult]
    report_paths: dict[str, str] | None = None
    trace_id: int | None = None


class TraceSummary(BaseModel):
    id: int
    kind: str
    created_at: str
    model: str | None = None
    gateway: str | None = None
    latency_ms: float | None = None
    tokens_estimate: int | None = None
    cost_usd: float | None = 0
    status: str | None = None


class TraceDetail(TraceSummary):
    request: dict[str, Any]
    response: dict[str, Any]
    events: list[dict[str, Any]]


class GatewayModelInfo(BaseModel):
    id: str
    provider: str | None = None
    gateway: str | None = None
    description: str = ""


class IngestRequest(BaseModel):
    name: str
    text: str
    metadata: dict[str, str] | None = None


class MLPredictRequest(BaseModel):
    records: list[dict[str, Any]] = Field(default_factory=list)
    mode: str = "batch"


class MLTrainRequest(BaseModel):
    tune: bool = True
    version: str | None = None


class CompareRequest(BaseModel):
    prompt: str = "Explain hybrid RAG briefly."
    completion_chars: int = 400

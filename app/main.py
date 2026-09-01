from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.agents.workflow import MultiAgentWorkflow
from app.api.middleware import APIKeyAuthMiddleware, RateLimitMiddleware, StructuredLoggingMiddleware
from app.api.schemas import HealthResponse
from app.api.v1.router import api_router
from app.core.cache import SimpleCache
from app.core.config import Settings, get_settings
from app.core.jobs import JobQueue
from app.gateway.router import GatewayRouter
from app.observability.langsmith_client import LangSmithClient
from app.observability.store import TraceStore
from app.rag.index import CorpusIndex, build_index
from app.rag.service import RagService


def _boot(app: FastAPI, settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    index = build_index(
        settings.corpus_dir,
        settings.chunk_size,
        settings.chunk_overlap,
        parent_child=settings.parent_child_chunking,
        vector_backend=settings.vector_backend,
    )
    gateway = GatewayRouter(settings)
    store = TraceStore(settings.db_path)
    rag = RagService(index, gateway.provider, settings.prompts_dir)
    agents = MultiAgentWorkflow(
        index,
        gateway.provider,
        settings.prompts_dir,
        require_human_approval=settings.require_human_approval,
    )
    app.state.wb = {
        "settings": settings,
        "index": index,
        "gateway": gateway,
        "store": store,
        "rag": rag,
        "agents": agents,
        "cache": SimpleCache(),
        "jobs": JobQueue(),
        "langsmith": LangSmithClient(api_key=settings.langchain_api_key),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    _boot(app)
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(
        title="AI Engineering Workbench",
        description=(
            "Local-first portfolio platform resembling what an AI platform team maintains: "
            "RAG, multi-agent workflows, evals, observability, ML eng, and data eng — "
            "offline with mock providers by default."
        ),
        version=__version__,
        lifespan=lifespan,
        openapi_tags=[
            {"name": "health", "description": "Liveness and index readiness"},
            {"name": "rag", "description": "Hybrid retrieval + grounded answers"},
            {"name": "agents", "description": "Planner / research / executor / critic"},
            {"name": "evals", "description": "Golden-set and agent evaluation"},
            {"name": "traces", "description": "SQLite traces, compare, analytics"},
            {"name": "gateway", "description": "LLM provider routing"},
            {"name": "ml", "description": "Demo-scale train / predict / drift / registry"},
            {"name": "dataeng", "description": "ETL, validation, quality, catalog"},
            {"name": "jobs", "description": "Background job status"},
            {"name": "observability", "description": "LangSmith stub / OTel status"},
        ],
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.add_middleware(StructuredLoggingMiddleware)
    application.add_middleware(RateLimitMiddleware, limit_per_minute=settings.rate_limit_per_minute)
    # Auth can be disabled for pytest via AIWB_AUTH_DISABLED=1
    import os
    auth_enabled = os.getenv("AIWB_AUTH_DISABLED", "").lower() not in ("1", "true", "yes")
    application.add_middleware(
        APIKeyAuthMiddleware,
        api_key=settings.api_key,
        enabled=auth_enabled,
        roles_json=settings.api_roles_json,
    )
    application.include_router(api_router)

    @application.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        st: dict[str, Any] = application.state.wb
        idx: CorpusIndex = st["index"]
        gw: GatewayRouter = st["gateway"]
        info = idx.info()
        return HealthResponse(
            status="ok",
            version=__version__,
            corpus_docs=idx.doc_count,
            chunks=len(idx.chunks),
            gateway=gw.mode,
            provider=gw.mode,
            index_ready=idx.ready,
            dense_backend=info.get("dense_backend"),
            vector_backend=info.get("vector_backend"),
        )

    @application.get("/")
    def root() -> dict[str, str]:
        return {
            "name": "AI Engineering Workbench",
            "version": __version__,
            "docs": "/docs",
            "health": "/health",
            "api": "/api/v1",
        }

    return application


app = create_app()


def create_app_for_tests(db_path=None, corpus_dir=None) -> FastAPI:
    get_settings.cache_clear()
    settings = get_settings()
    if db_path:
        settings.db_path = db_path
    if corpus_dir:
        settings.corpus_dir = corpus_dir
    _boot(app, settings)
    return app

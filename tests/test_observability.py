from app.observability.store import TraceStore
from app.observability.analytics import trace_analytics
from app.observability.langsmith_client import LangSmithClient
from app.observability.otel import start_span


def test_trace_compare_and_analytics(tmp_path):
    store = TraceStore(tmp_path / "t.db")
    a = store.add(kind="rag", model="mock", gateway="mock", latency_ms=10, tokens_estimate=100, status="ok", request={}, response={}, cost_usd=0.01)
    b = store.add(kind="rag", model="mock", gateway="mock", latency_ms=20, tokens_estimate=150, status="ok", request={}, response={}, cost_usd=0.02)
    assert store.compare(a, b)["delta"]["latency_ms"] == 10
    assert trace_analytics(store)["total_tokens"] == 250


def test_langsmith_local_and_otel(tmp_path):
    client = LangSmithClient(api_key=None, local_root=tmp_path / "ls")
    rid = client.create_run("test", {"q": 1})
    client.end_run(rid, {"ok": True})
    assert client.list_runs()
    with start_span("unit") as span:
        span.set_attribute("k", 1)

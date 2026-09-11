# Scalability notes

This portfolio build is **single-node / demo-scale**. A production evolution path:

1. **Retrieval** — swap CorpusIndex for a vector DB (Qdrant/pgvector) + managed BM25; keep the RagService interface.
2. **LLM gateway** — move GatewayRouter to a sidecar with retries, budgets, and per-tenant keys.
3. **Agents** — durable orchestration (Temporal/Ray) for long-running tools; keep role contracts.
4. **Evals** — schedule regression in CI against golden sets; add LLM-judge optionally with cost caps.
5. **Observability** — export traces to OpenTelemetry / ClickHouse; retain SQLite for local.
6. **ML** — feature store + model registry (MLflow/Feast); keep train/predict API shapes.
7. **API** — horizontal pods behind a gateway; replace in-process rate limit with Redis token buckets.

None of the above are implemented here — documenting the seam is the point.

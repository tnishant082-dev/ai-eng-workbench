# FastAPI for AI Services

Expose stable versioned routes like `/v1/rag/query` and `/v1/agent/run`.
Use Pydantic models for request/response contracts and OpenAPI docs for free.
Health endpoints should report index readiness and gateway mode.
Keep business logic out of route handlers — services own RAG, agent, and evals.

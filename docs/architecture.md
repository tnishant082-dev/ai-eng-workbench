# Architecture

Streamlit Ops Console → FastAPI `/api/v1` (authz roles, rate limit, cache, jobs) → RAG (BM25+dense+RRF) / Agents / Evals / ML / Data Eng → Observability (SQLite traces, OTel hooks, LangSmith stub).

Local-first: mock LLM, hashing dense proxy, in-memory vectors by default. Optional extras for ST/FAISS/Chroma/MLflow/OTel.

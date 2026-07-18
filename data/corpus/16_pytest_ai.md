# Testing AI Pipelines

Unit-test chunking, retrieval ranking, and tool argument parsing without an LLM.
Integration tests hit FastAPI TestClient with the mock gateway.
Golden-set evals assert citation presence and non-empty answers on fixtures.
Avoid flaky network tests; optional OpenAI path stays behind a skip marker.

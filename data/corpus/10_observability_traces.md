# Observability and Traces

Store each run with request, response, latency_ms, model, and token estimate.
Trace events are ordered spans: retrieve, prompt_render, llm_call, tool_call.
SQLite is enough for local-first workbenches; export JSON for LangSmith-style tools later.
List and detail APIs let Streamlit and recruiters inspect failures without reading logs.

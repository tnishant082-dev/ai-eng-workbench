from __future__ import annotations

import streamlit as st

from ui.api_client import api

st.title("RAG Studio")
st.caption("Hybrid BM25 + dense · Reciprocal Rank Fusion · multi-query · rerank · metadata filter")

q = st.text_area("Question", "What is hybrid RAG and why use Reciprocal Rank Fusion?")
c1, c2, c3 = st.columns(3)
top_k = c1.slider("top_k", 1, 10, 5)
multi = c2.checkbox("Multi-query expand", True)
use_rrf = c3.checkbox("RRF fusion", True)
topic = st.text_input("Metadata filter topic (optional)", "")

if st.button("Run retrieval", type="primary"):
    body = {"question": q, "top_k": top_k, "use_multi_query": multi, "use_rrf": use_rrf}
    if topic.strip():
        body["metadata_filter"] = {"topic": topic.strip()}
    out = api("POST", "/api/v1/rag/query", json_body=body)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Latency ms", out["latency_ms"])
    m2.metric("Faithfulness", out.get("faithfulness", 0))
    m3.metric("Confidence", out.get("confidence", 0))
    m4.metric("Tokens", out.get("tokens_estimate", 0))
    if out.get("retrieval"):
        st.json(out["retrieval"])
    st.subheader("Answer")
    st.markdown(out["answer"])
    st.subheader("Citations / retrieval inspection")
    st.dataframe(out["citations"], use_container_width=True)
    st.caption(f"trace_id={out.get('trace_id')}")

with st.expander("Index info"):
    st.json(api("GET", "/api/v1/rag/index"))

if st.button("Run retrieval metrics @K"):
    st.json(api("POST", "/api/v1/rag/retrieval-metrics?k=5&limit=12"))

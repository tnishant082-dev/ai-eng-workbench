from __future__ import annotations

import streamlit as st

from ui.api_client import api

st.title("Evaluation Center")
st.caption("Context precision/recall · faithfulness · relevance · citation · LLM-judge stub · HTML/MD reports")

limit = st.slider("Golden cases", 3, 30, 10)
write_report = st.checkbox("Write reports/ artifacts", True)

if st.button("Run RAG evals", type="primary"):
    out = api("POST", "/api/v1/evals/run", json_body={"limit": limit, "write_report": write_report})
    s = out["summary"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pass rate", s.get("pass_rate"))
    c2.metric("Faithfulness", s.get("avg_faithfulness"))
    c3.metric("Context P", s.get("avg_context_precision"))
    c4.metric("Context R", s.get("avg_context_recall"))
    st.dataframe(out["cases"], use_container_width=True)
    if out.get("report_paths"):
        st.success(out["report_paths"])

if st.button("Run agent evals"):
    st.json(api("POST", "/api/v1/evals/agents"))

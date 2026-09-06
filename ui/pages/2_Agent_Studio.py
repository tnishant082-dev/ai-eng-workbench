from __future__ import annotations

import streamlit as st

from ui.api_client import api

st.title("Agent Studio")
st.caption("Planner · Research · Executor · Critic · Reflection · retry / recovery · approval gate")

task = st.text_area(
    "Task",
    "Search the corpus for faithfulness evals, then calculate 12 * (3 + 4).",
)
approve = st.checkbox("Approve plan (when human-approval mode is on)", True)
max_steps = st.slider("max_steps", 2, 10, 6)

st.markdown(
    """
```mermaid
flowchart LR
  P[Planner] --> R[Research]
  R --> E[Executor]
  E --> C[Critic]
  C --> F[Reflect]
  F -->|improve| R
  F -->|pass| S[Synthesize]
```
"""
)

if st.button("Run agents", type="primary"):
    out = api("POST", "/api/v1/agents/run", json_body={"task": task, "max_steps": max_steps, "approve": approve})
    st.metric("Status", out.get("status"))
    cols = st.columns(4)
    cols[0].metric("Latency ms", out["latency_ms"])
    cols[1].metric("Tools", len(out.get("tools_used") or []))
    cols[2].metric("Events", len(out.get("events") or []))
    cols[3].metric("Failures recovered", len(out.get("failures") or []))
    st.subheader("Answer")
    st.markdown(out["answer"])
    st.subheader("Plan")
    st.json(out.get("plan"))
    st.subheader("Event timeline")
    for e in out.get("events") or []:
        st.write(f"**{e.get('type')}** / {e.get('role', '')} — {str(e.get('content', e))[:240]}")
    if out.get("failures"):
        st.warning(out["failures"])

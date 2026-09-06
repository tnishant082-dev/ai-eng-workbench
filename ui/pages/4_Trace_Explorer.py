from __future__ import annotations

import streamlit as st

from ui.api_client import api

st.title("Trace Explorer")
st.caption("SQLite traces · run compare · token/cost/error analytics")

tabs = st.tabs(["Recent", "Detail", "Compare", "Analytics"])

with tabs[0]:
    rows = api("GET", "/api/v1/traces?limit=50")
    st.dataframe(rows, use_container_width=True)

with tabs[1]:
    tid = st.number_input("Trace id", min_value=1, value=1)
    if st.button("Load trace"):
        st.json(api("GET", f"/api/v1/traces/{int(tid)}"))

with tabs[2]:
    a = st.number_input("A", min_value=1, value=1, key="ca")
    b = st.number_input("B", min_value=1, value=2, key="cb")
    if st.button("Compare"):
        st.json(api("GET", f"/api/v1/traces/compare/{int(a)}/{int(b)}"))

with tabs[3]:
    if st.button("Refresh analytics"):
        st.json(api("GET", "/api/v1/traces/analytics"))

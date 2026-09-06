from __future__ import annotations

import streamlit as st

from ui.api_client import api

st.title("System Monitoring")
st.caption("Health · jobs · gateway · LangSmith stub · OTel status · analytics")

h = api("GET", "/health")
st.subheader("Health")
st.json(h)

st.subheader("Observability")
st.json(api("GET", "/api/v1/observability/status"))

st.subheader("Gateway models")
st.json(api("GET", "/api/v1/gateway/models"))

st.subheader("Jobs")
st.json(api("GET", "/api/v1/jobs"))

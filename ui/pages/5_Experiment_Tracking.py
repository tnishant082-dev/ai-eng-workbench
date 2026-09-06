from __future__ import annotations

import streamlit as st

from ui.api_client import api

st.title("Experiment Tracking")
st.caption("Demo-scale train · hyperparam grid · filesystem/MLflow tracking backend")

tune = st.checkbox("Hyperparam grid search", True)
version = st.text_input("Dataset version (optional, e.g. v1)", "v1")

if st.button("Train", type="primary"):
    body = {"tune": tune, "version": version or None}
    out = api("POST", "/api/v1/ml/train", json_body=body)
    st.json(out)

st.subheader("Registry snapshot")
if st.button("Refresh registry"):
    st.json(api("GET", "/api/v1/ml/registry"))

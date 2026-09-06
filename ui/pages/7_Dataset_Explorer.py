from __future__ import annotations

import streamlit as st

from ui.api_client import api

st.title("Dataset Explorer")
st.caption("Catalog · lineage · quality · ETL · versioning folder convention")

c1, c2, c3 = st.columns(3)
if c1.button("Run ETL", type="primary"):
    st.json(api("POST", "/api/v1/dataeng/etl"))
if c2.button("Validate"):
    st.json(api("GET", "/api/v1/dataeng/validate"))
if c3.button("Quality"):
    st.json(api("GET", "/api/v1/dataeng/quality"))

st.info("Versions live under `data/datasets/versions/<ver>/` with manifest.json — not DVC/LakeFS.")

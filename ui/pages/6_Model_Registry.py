from __future__ import annotations

import json

import streamlit as st

from ui.api_client import api

st.title("Model Registry")
st.caption("Registered demo models · batch/online predict · PSI drift")

reg = api("GET", "/api/v1/ml/registry")
st.json(reg)

st.subheader("Predict")
sample = st.text_area(
    "JSON records",
    json.dumps(
        [
            {
                "tenure_months": 12,
                "monthly_charges": 70.0,
                "total_charges": 800.0,
                "support_tickets": 2,
                "contract": "month-to-month",
            }
        ],
        indent=2,
    ),
)
mode = st.selectbox("Mode", ["online", "batch"])
if st.button("Predict"):
    records = json.loads(sample)
    st.json(api("POST", "/api/v1/ml/predict", json_body={"records": records, "mode": mode}))

if st.button("Drift check"):
    st.json(api("POST", "/api/v1/ml/drift", json_body={"records": []}))

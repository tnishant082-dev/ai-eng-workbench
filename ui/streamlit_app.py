from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
from ui.api_client import API_URL, api

st.set_page_config(page_title="AI Eng Workbench — Flagship Console", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
.block-container { padding-top: 1.1rem; max-width: 1280px; }
div[data-testid="stMetric"] { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); color: #e2e8f0; border: 1px solid #334155; border-radius: 12px; padding: 12px 14px; }
div[data-testid="stMetric"] label { color: #94a3b8 !important; }
.aiwb-chip { display:inline-block; padding:4px 10px; margin:2px; border-radius:999px; background:#1e293b; border:1px solid #475569; color:#cbd5e1; font-size:0.8rem; }
</style>""", unsafe_allow_html=True)
st.title("AI Engineering Workbench")
st.caption("Flagship local-first ops console · RAG Studio · Agents · Evals · Traces · Experiments · Registry · Monitoring")
try:
    health = api("GET", "/health")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Corpus docs", health["corpus_docs"])
    c2.metric("Chunks", health["chunks"])
    c3.metric("Provider", health.get("provider") or health["gateway"])
    c4.metric("Dense", health.get("dense_backend") or "—")
    c5.metric("Version", health["version"])
    st.sidebar.success(f"API ok @ {API_URL}")
    st.markdown(f'<span class="aiwb-chip">vector={health.get("vector_backend")}</span><span class="aiwb-chip">dense={health.get("dense_backend")}</span><span class="aiwb-chip">mock-first</span>', unsafe_allow_html=True)
except Exception as exc:
    st.sidebar.error(f"API unreachable at {API_URL}: {exc}")
    st.warning("Start the API: `uvicorn app.main:app --port 8000`")
    st.stop()
st.markdown("""
### Console map
| Page | Purpose |
|---|---|
| **RAG Studio** | Hybrid BM25+dense+RRF, multi-query, rerank, citation inspector |
| **Agent Studio** | Planner → research → critic → executor + reflection graph |
| **Evaluation Center** | Golden metrics, context P/R, reports |
| **Trace Explorer** | SQLite timelines, compare, token/cost analytics |
| **Experiment Tracking** | Train runs, hyperparam grid, metrics |
| **Model Registry** | Registered demo models + predict/drift |
| **Dataset Explorer** | Catalog, lineage, quality, versions |
| **System Monitoring** | Health, jobs, observability status |
""")

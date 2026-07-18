# Architecture Overview

AI Eng Workbench is one product: FastAPI services + Streamlit UI + SQLite traces.
Modules: ingest, RAG, agent/tools, prompts, evals, observability, gateway.
Default path is fully offline; OpenAI-compatible routing is optional.
Docker Compose and pytest ship with the repo for one-command demos and CI.

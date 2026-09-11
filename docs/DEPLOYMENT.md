# Deployment

## Local (recommended)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

uvicorn app.main:app --host 0.0.0.0 --port 8000
# other terminal
AIWB_API_KEY=dev-workbench-key API_URL=http://127.0.0.1:8000 streamlit run ui/streamlit_app.py
```

## Docker Compose

```bash
docker compose up --build
# API :8000  UI :8501
```

Compose sets `AIWB_LLM_PROVIDER=mock` so the stack stays offline.

## Optional cloud LLM

```bash
export AIWB_LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-...
# or groq / ollama base URLs — see .env.example
```

## CI

GitHub Actions runs `pytest` on push/PR (`.github/workflows/ci.yml`).

## Honest limits

This is a local-first portfolio platform. There is no hosted production environment, SLA, or multi-tenant control plane in this repository.

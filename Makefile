.PHONY: install test api ui eval docker

install:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

test:
	AIWB_AUTH_DISABLED=1 AIWB_LLM_PROVIDER=mock pytest -q

api:
	uvicorn app.main:app --reload --port 8000

ui:
	AIWB_API_KEY=dev-workbench-key API_URL=http://127.0.0.1:8000 streamlit run ui/streamlit_app.py

eval:
	python -m app.evals

docker:
	docker compose up --build

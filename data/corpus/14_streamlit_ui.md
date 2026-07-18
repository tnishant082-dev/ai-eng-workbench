# Streamlit Workbench UI

Tabs for RAG, Agent, Evals, and Traces match how engineers actually debug.
Show citations as expandable snippets with scores, not just raw JSON.
Call the FastAPI backend over HTTP so UI and API stay independently testable.
Keep the UI thin: no second copy of retrieval or agent logic in Streamlit.

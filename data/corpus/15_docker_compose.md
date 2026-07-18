# Docker Compose Local Stack

Compose should start API and UI with shared volumes for corpus and SQLite.
Default env uses mock LLM so `docker compose up` works without secrets.
Pin base images lightly and document ports 8000 (API) and 8501 (UI).
Compose is for demos; production would add reverse proxy and managed storage.

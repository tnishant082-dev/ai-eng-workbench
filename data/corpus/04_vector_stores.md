# Vector Stores

FAISS, Chroma, and Qdrant are common local vector stores.
SQLite with stored embeddings works for tiny demos and simplifies Docker Compose.
Never treat a vector DB as a source of truth — keep the raw corpus versioned in git.
Index rebuilds should be idempotent and logged with document counts and latency.

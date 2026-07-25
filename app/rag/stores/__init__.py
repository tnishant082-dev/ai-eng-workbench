"""Vector store adapters: in-memory (default), FAISS/Chroma optional, pgvector stub."""
from __future__ import annotations

from app.rag.stores.base import VectorRecord, VectorStore
from app.rag.stores.memory import InMemoryVectorStore
from app.rag.stores.factory import get_vector_store

__all__ = ["VectorRecord", "VectorStore", "InMemoryVectorStore", "get_vector_store"]

from __future__ import annotations

from app.rag.stores.memory import InMemoryVectorStore
from app.rag.stores.pgvector_stub import PgVectorStoreStub


def get_vector_store(backend: str = "memory", dim: int = 384, **kwargs):
    """
    backend: memory | faiss | chroma | pgvector
    FAISS/Chroma require optional extras; fall back to memory with a note.
    """
    backend = (backend or "memory").lower()
    if backend == "memory":
        return InMemoryVectorStore(), "memory"
    if backend == "faiss":
        try:
            from app.rag.stores.faiss_store import FaissVectorStore

            return FaissVectorStore(dim=dim), "faiss"
        except Exception as exc:  # noqa: BLE001
            store = InMemoryVectorStore()
            store._fallback_note = f"faiss unavailable ({exc}); using in-memory"  # type: ignore[attr-defined]
            return store, "memory_fallback"
    if backend == "chroma":
        try:
            from app.rag.stores.chroma_store import ChromaVectorStore

            return ChromaVectorStore(
                collection_name=kwargs.get("collection_name", "aiwb"),
                persist_directory=kwargs.get("persist_directory"),
            ), "chroma"
        except Exception as exc:  # noqa: BLE001
            store = InMemoryVectorStore()
            store._fallback_note = f"chromadb unavailable ({exc}); using in-memory"  # type: ignore[attr-defined]
            return store, "memory_fallback"
    if backend == "pgvector":
        return PgVectorStoreStub(dsn=kwargs.get("dsn")), "pgvector_stub"
    return InMemoryVectorStore(), "memory"

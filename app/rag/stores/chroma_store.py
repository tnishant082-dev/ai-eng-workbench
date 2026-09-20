"""Optional Chroma adapter — graceful if chromadb missing."""
from __future__ import annotations

import numpy as np

from app.rag.stores.base import VectorRecord


class ChromaVectorStore:
    name = "chroma"
    backend = "chromadb"

    def __init__(self, collection_name: str = "aiwb", persist_directory: str | None = None):
        import chromadb  # type: ignore

        if persist_directory:
            client = chromadb.PersistentClient(path=persist_directory)
        else:
            client = chromadb.Client()
        self._col = client.get_or_create_collection(collection_name, metadata={"hnsw:space": "cosine"})
        self._count = 0

    def upsert(self, records: list[VectorRecord]) -> int:
        ids, docs, metas, embs = [], [], [], []
        for r in records:
            if r.vector is None:
                continue
            ids.append(r.id)
            docs.append(r.text)
            metas.append({k: str(v) for k, v in (r.metadata or {}).items()})
            embs.append(np.asarray(r.vector, dtype=np.float32).ravel().tolist())
        if not ids:
            return 0
        self._col.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
        self._count += len(ids)
        return len(ids)

    def search(self, query_vector: np.ndarray, top_k: int = 5, metadata_filter: dict | None = None) -> list[tuple[str, float]]:
        where = {k: str(v) for k, v in metadata_filter.items()} if metadata_filter else None
        kwargs = {
            "query_embeddings": [np.asarray(query_vector, dtype=np.float32).ravel().tolist()],
            "n_results": top_k,
        }
        if where:
            kwargs["where"] = where
        res = self._col.query(**kwargs)
        ids = (res.get("ids") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        # chroma cosine distance → similarity-ish
        out = []
        for i, d in zip(ids, dists):
            sim = 1.0 - float(d) if d is not None else 0.0
            out.append((i, sim))
        return out

    def count(self) -> int:
        try:
            return int(self._col.count())
        except (TypeError, ValueError, AttributeError, RuntimeError, OSError):
            return self._count

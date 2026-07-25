"""Optional FAISS adapter — import fails gracefully if faiss-cpu missing."""
from __future__ import annotations

import numpy as np

from app.rag.stores.base import VectorRecord


class FaissVectorStore:
    name = "faiss"
    backend = "faiss"

    def __init__(self, dim: int):
        import faiss  # type: ignore

        self._faiss = faiss
        self.dim = dim
        self._index = faiss.IndexFlatIP(dim)
        self._ids: list[str] = []
        self._meta: list[dict] = []

    def upsert(self, records: list[VectorRecord]) -> int:
        vecs = []
        for r in records:
            if r.vector is None:
                continue
            v = np.asarray(r.vector, dtype=np.float32).ravel()
            v = v / (np.linalg.norm(v) + 1e-9)
            if v.shape[0] != self.dim:
                raise ValueError(f"dim mismatch: expected {self.dim}, got {v.shape[0]}")
            self._ids.append(r.id)
            self._meta.append(r.metadata or {})
            vecs.append(v)
        if not vecs:
            return 0
        self._index.add(np.vstack(vecs))
        return len(vecs)

    def search(self, query_vector: np.ndarray, top_k: int = 5, metadata_filter: dict | None = None) -> list[tuple[str, float]]:
        if not self._ids:
            return []
        q = np.asarray(query_vector, dtype=np.float32).ravel()
        q = q / (np.linalg.norm(q) + 1e-9)
        # overfetch if filtering
        fetch = top_k * 5 if metadata_filter else top_k
        fetch = min(fetch, len(self._ids))
        scores, idxs = self._index.search(q.reshape(1, -1), fetch)
        out: list[tuple[str, float]] = []
        for score, i in zip(scores[0], idxs[0]):
            if i < 0:
                continue
            if metadata_filter:
                meta = self._meta[i]
                if not all(str(meta.get(k, "")) == str(v) for k, v in metadata_filter.items()):
                    continue
            out.append((self._ids[i], float(score)))
            if len(out) >= top_k:
                break
        return out

    def count(self) -> int:
        return len(self._ids)

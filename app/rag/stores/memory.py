from __future__ import annotations

import numpy as np

from app.rag.stores.base import VectorRecord


class InMemoryVectorStore:
    """Always-available dense store using cosine similarity over numpy arrays."""

    name = "in_memory"
    backend = "numpy"

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._meta: list[dict] = []
        self._matrix: np.ndarray | None = None

    def upsert(self, records: list[VectorRecord]) -> int:
        vecs = []
        for r in records:
            if r.vector is None:
                continue
            self._ids.append(r.id)
            self._meta.append(r.metadata or {})
            v = np.asarray(r.vector, dtype=np.float32).ravel()
            n = np.linalg.norm(v) + 1e-9
            vecs.append(v / n)
        if not vecs:
            return 0
        block = np.vstack(vecs)
        self._matrix = block if self._matrix is None else np.vstack([self._matrix, block])
        return len(vecs)

    def search(self, query_vector: np.ndarray, top_k: int = 5, metadata_filter: dict | None = None) -> list[tuple[str, float]]:
        if self._matrix is None or not self._ids:
            return []
        q = np.asarray(query_vector, dtype=np.float32).ravel()
        q = q / (np.linalg.norm(q) + 1e-9)
        scores = self._matrix @ q
        order = np.argsort(-scores)
        out: list[tuple[str, float]] = []
        for i in order:
            if metadata_filter:
                meta = self._meta[i]
                if not all(str(meta.get(k, "")) == str(v) for k, v in metadata_filter.items()):
                    continue
            out.append((self._ids[i], float(scores[i])))
            if len(out) >= top_k:
                break
        return out

    def count(self) -> int:
        return len(self._ids)

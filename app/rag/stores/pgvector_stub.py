"""pgvector interface stub — no real Postgres required.

Documented adapter surface for when a Postgres+pgvector instance is available.
Methods raise NotImplementedError with guidance unless a DSN is provided and
psycopg is installed (still planned for local demos — skip real DB by default).
"""
from __future__ import annotations

import numpy as np

from app.rag.stores.base import VectorRecord


class PgVectorStoreStub:
    """Interface-compatible stub. Status: planned unless AIWB_PG_DSN is set."""

    name = "pgvector"
    backend = "pgvector_stub"
    status = "planned"

    def __init__(self, dsn: str | None = None):
        self.dsn = dsn
        if dsn:
            # Real wire-up is optional / planned — keep stub honest.
            self.status = "planned_with_dsn"
        self._note = (
            "pgvector adapter is an interface stub. Provide Postgres+pgvector and "
            "implement SQL upsert/search, or use InMemory/FAISS/Chroma locally."
        )

    def upsert(self, records: list[VectorRecord]) -> int:
        raise NotImplementedError(self._note)

    def search(self, query_vector: np.ndarray, top_k: int = 5, metadata_filter: dict | None = None):
        raise NotImplementedError(self._note)

    def count(self) -> int:
        return 0

    def info(self) -> dict:
        return {"name": self.name, "backend": self.backend, "status": self.status, "note": self._note}

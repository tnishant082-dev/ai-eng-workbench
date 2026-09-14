import numpy as np

from app.rag.stores.factory import get_vector_store
from app.rag.stores.base import VectorRecord
from app.rag.stores.pgvector_stub import PgVectorStoreStub


def test_memory_store_and_pg_stub():
    store, name = get_vector_store("memory", dim=8)
    assert name == "memory"
    vecs = [np.random.randn(8).astype(np.float32) for _ in range(3)]
    records = [VectorRecord(id=f"d{i}", text="t", vector=v, metadata={"topic": "x"}) for i, v in enumerate(vecs)]
    store.upsert(records)
    hits = store.search(vecs[0], top_k=2)
    assert hits and hits[0][0] == "d0"
    stub = PgVectorStoreStub()
    info = stub.info()
    assert info["status"] == "planned"

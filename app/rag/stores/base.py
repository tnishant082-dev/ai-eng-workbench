from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class VectorRecord:
    id: str
    text: str
    vector: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)


class VectorStore(Protocol):
    name: str
    backend: str

    def upsert(self, records: list[VectorRecord]) -> int: ...
    def search(self, query_vector: np.ndarray, top_k: int = 5, metadata_filter: dict | None = None) -> list[tuple[str, float]]: ...
    def count(self) -> int: ...

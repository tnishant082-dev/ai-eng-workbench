from __future__ import annotations

import hashlib
import json
import time
from typing import Any


class SimpleCache:
    """In-process TTL cache for RAG/gateway responses (demo-scale)."""

    def __init__(self, ttl_seconds: float = 120.0, max_items: int = 256):
        self.ttl = ttl_seconds
        self.max_items = max_items
        self._store: dict[str, tuple[float, Any]] = {}

    @staticmethod
    def key(*parts: Any) -> str:
        raw = json.dumps(parts, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    def get(self, key: str) -> Any | None:
        item = self._store.get(key)
        if not item:
            return None
        expires, value = item
        if time.time() > expires:
            self._store.pop(key, None)
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        if len(self._store) >= self.max_items:
            # drop oldest
            oldest = min(self._store.items(), key=lambda kv: kv[1][0])[0]
            self._store.pop(oldest, None)
        self._store[key] = (time.time() + self.ttl, value)

    def clear(self) -> None:
        self._store.clear()

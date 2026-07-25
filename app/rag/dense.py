"""Dense retrieval backends.

Default: HashingVectorizer dense proxy (no heavy deps) — clearly labeled.
Optional: sentence-transformers if installed (extras: dense).
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class DenseEncoder(Protocol):
    name: str
    backend: str

    def encode(self, texts: list[str]) -> np.ndarray: ...


class HashingDenseEncoder:
    """Hashing / FeatureHasher-style dense proxy via sklearn HashingVectorizer.

    Not a semantic embedding model — deterministic sparse-hashed dense vectors
    for offline demos when sentence-transformers is unavailable.
    """

    name = "hashing-dense-proxy"
    backend = "hashing"

    def __init__(self, n_features: int = 384):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.n_features = n_features
        self._vec = HashingVectorizer(
            n_features=n_features,
            alternate_sign=False,
            norm="l2",
            stop_words="english",
            ngram_range=(1, 2),
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        mat = self._vec.transform(texts)
        return np.asarray(mat.todense(), dtype=np.float32)


class TfidfDenseProxy:
    """TF-IDF truncated to fixed dim via TruncatedSVD — labeled dense proxy."""

    name = "tfidf-svd-dense-proxy"
    backend = "tfidf_svd"

    def __init__(self, n_components: int = 128):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.n_components = n_components
        self._tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=8000)
        self._svd = TruncatedSVD(n_components=n_components, random_state=42)
        self._fitted = False

    def fit(self, texts: list[str]) -> "TfidfDenseProxy":
        X = self._tfidf.fit_transform(texts)
        n = min(self.n_components, max(X.shape[1] - 1, 1), max(X.shape[0] - 1, 1))
        if n < self.n_components:
            self._svd = type(self._svd)(n_components=max(n, 1), random_state=42)
        self._svd.fit(X)
        self._fitted = True
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfDenseProxy must be fit() on corpus first")
        X = self._tfidf.transform(texts)
        return np.asarray(self._svd.transform(X), dtype=np.float32)


class SentenceTransformerEncoder:
    """Optional real dense embeddings behind extras."""

    name = "sentence-transformers"
    backend = "sentence_transformers"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.model_name = model_name

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._model.encode(texts, show_progress_bar=False), dtype=np.float32)


def get_dense_encoder(prefer: str = "auto", corpus_texts: list[str] | None = None) -> DenseEncoder:
    """Resolve dense backend: sentence-transformers → hashing proxy."""
    if prefer in ("sentence_transformers", "auto"):
        try:
            enc = SentenceTransformerEncoder()
            return enc
        except Exception:
            if prefer == "sentence_transformers":
                raise
    if prefer == "tfidf_svd" and corpus_texts is not None:
        return TfidfDenseProxy().fit(corpus_texts)
    return HashingDenseEncoder()

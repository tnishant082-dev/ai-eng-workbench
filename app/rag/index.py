from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.rag.chunking import parent_child_chunk, strip_title
from app.rag.dense import HashingDenseEncoder, get_dense_encoder
from app.rag.fusion import reciprocal_rank_fusion
from app.rag.stores.base import VectorRecord
from app.rag.stores.factory import get_vector_store


@dataclass
class Chunk:
    doc_id: str
    title: str
    text: str
    path: str
    source_name: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None
    parent_text: str | None = None


@dataclass
class Hit:
    chunk: Chunk
    score: float
    bm25: float = 0.0
    tfidf: float = 0.0
    dense: float = 0.0
    rrf: float = 0.0
    rerank: float = 0.0


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _chunk_text(text: str, size: int, overlap: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paras:
        paras = [text]
    chunks: list[str] = []
    buf = ""
    for p in paras:
        if not buf:
            buf = p
        elif len(buf) + 1 + len(p) <= size:
            buf = f"{buf}\n\n{p}"
        else:
            chunks.append(buf)
            if overlap > 0 and len(buf) > overlap:
                buf = buf[-overlap:] + "\n\n" + p
            else:
                buf = p
    if buf:
        chunks.append(buf)
    final: list[str] = []
    for c in chunks:
        if len(c) <= size * 2:
            final.append(c)
        else:
            step = max(size - overlap, 1)
            for i in range(0, len(c), step):
                final.append(c[i : i + size])
    return [c for c in final if len(c.strip()) >= 40]


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs = docs
        self.N = len(docs)
        self.df: dict[str, int] = defaultdict(int)
        self.doc_len = [len(d) for d in docs]
        self.avgdl = sum(self.doc_len) / max(self.N, 1)
        for d in docs:
            for t in set(d):
                self.df[t] += 1

    def score(self, query: list[str], idx: int) -> float:
        doc = self.docs[idx]
        freqs = Counter(doc)
        dl = self.doc_len[idx]
        s = 0.0
        for t in query:
            if t not in self.df:
                continue
            n = self.df[t]
            idf = math.log(1 + (self.N - n + 0.5) / (n + 0.5))
            f = freqs.get(t, 0)
            denom = f + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-9))
            s += idf * (f * (self.k1 + 1)) / max(denom, 1e-9)
        return s


class CorpusIndex:
    """Hybrid BM25 + dense with RRF fusion; optional parent-child chunking."""

    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.doc_count = 0
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None
        self._bm25: BM25 | None = None
        self._tokens: list[list[str]] = []
        self.ready = False
        self._corpus_dir: Path | None = None
        self._chunk_size = 500
        self._chunk_overlap = 80
        self._use_parent_child = False
        self._dense_encoder = None
        self._dense_backend = "hashing"
        self._vector_store = None
        self._vector_backend = "memory"
        self._id_to_idx: dict[str, int] = {}
        self.fusion_mode = "rrf"

    def build(
        self,
        corpus_dir: Path,
        chunk_size: int = 500,
        chunk_overlap: int = 80,
        *,
        parent_child: bool = False,
        vector_backend: str = "memory",
        dense_prefer: str = "auto",
    ) -> int:
        corpus_dir = Path(corpus_dir)
        self._corpus_dir = corpus_dir
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._use_parent_child = parent_child
        if not corpus_dir.exists():
            raise FileNotFoundError(f"corpus not found: {corpus_dir}")

        chunks: list[Chunk] = []
        files = sorted(corpus_dir.rglob("*.md"))
        for path in files:
            raw = path.read_text(encoding="utf-8")
            title, body = strip_title(raw)
            if not title or title == "Untitled":
                title = path.stem.replace("_", " ").title()
            topic = path.stem.split("_", 1)[-1] if "_" in path.stem else path.stem
            meta_base = {"topic": topic, "source": path.name}
            rel = str(path.relative_to(corpus_dir))

            if parent_child:
                pcs = parent_child_chunk(
                    doc_id=path.stem,
                    title=title,
                    body=body or raw,
                    path=rel,
                    source_name=path.name,
                    metadata=meta_base,
                )
                for pc in pcs:
                    if pc.is_parent:
                        continue
                    chunks.append(
                        Chunk(
                            doc_id=pc.chunk_id,
                            title=pc.title,
                            text=pc.child_text,
                            path=pc.path,
                            source_name=pc.source_name,
                            metadata=pc.metadata,
                            parent_id=pc.parent_id,
                            parent_text=pc.parent_text,
                        )
                    )
            else:
                parts = _chunk_text(body or raw, chunk_size, chunk_overlap)
                if not parts:
                    parts = [raw.strip()]
                for i, part in enumerate(parts):
                    chunks.append(
                        Chunk(
                            doc_id=f"{path.stem}#{i}",
                            title=title,
                            text=part.strip(),
                            path=rel,
                            source_name=path.name,
                            metadata=meta_base,
                        )
                    )

        if not chunks:
            raise RuntimeError(f"no chunks under {corpus_dir}")

        self._fit(chunks, len(files), vector_backend=vector_backend, dense_prefer=dense_prefer)
        return len(chunks)

    def _fit(self, chunks, doc_count, *, vector_backend="memory", dense_prefer="auto"):
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=12000)
        texts = [f"{c.title}\n{c.text}" for c in chunks]
        matrix = vectorizer.fit_transform(texts)
        tokens = [_tokenize(t) for t in texts]
        try:
            encoder = get_dense_encoder(prefer=dense_prefer, corpus_texts=texts)
        except Exception:
            encoder = HashingDenseEncoder()
        dense_vecs = encoder.encode(texts)
        dim = int(dense_vecs.shape[1])
        store, vb = get_vector_store(vector_backend, dim=dim)
        records = [
            VectorRecord(id=c.doc_id, text=c.text, vector=dense_vecs[i], metadata=c.metadata)
            for i, c in enumerate(chunks)
        ]
        try:
            store.upsert(records)
        except NotImplementedError:
            store, vb = get_vector_store("memory", dim=dim)
            store.upsert(records)
        self.chunks = chunks
        self.doc_count = doc_count
        self._vectorizer = vectorizer
        self._matrix = matrix
        self._tokens = tokens
        self._bm25 = BM25(tokens)
        self._dense_encoder = encoder
        self._dense_backend = getattr(encoder, "backend", "unknown")
        self._vector_store = store
        self._vector_backend = vb
        self._id_to_idx = {c.doc_id: i for i, c in enumerate(chunks)}
        self.ready = True

    def rebuild(self) -> int:
        if not self._corpus_dir:
            raise RuntimeError("no corpus_dir — call build first")
        return self.build(
            self._corpus_dir,
            self._chunk_size,
            self._chunk_overlap,
            parent_child=self._use_parent_child,
            vector_backend=self._vector_backend if self._vector_backend != "memory_fallback" else "memory",
        )

    def ingest_text(self, name: str, text: str, metadata: dict | None = None) -> int:
        if not self._corpus_dir:
            raise RuntimeError("index not built")
        path = self._corpus_dir / name
        if not name.endswith(".md"):
            path = self._corpus_dir / f"{name}.md"
        path.write_text(text if text.lstrip().startswith("#") else f"# {name}\n\n{text}", encoding="utf-8")
        return self.rebuild()

    def info(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "chunks": len(self.chunks),
            "docs": self.doc_count,
            "dense_backend": self._dense_backend,
            "vector_backend": self._vector_backend,
            "fusion_mode": self.fusion_mode,
            "parent_child": self._use_parent_child,
        }

    def search(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
        alpha: float = 0.55,
        use_rrf: bool = True,
        overfetch: int = 40,
    ) -> list[Hit]:
        if not self.ready or self._vectorizer is None or self._matrix is None or self._bm25 is None:
            raise RuntimeError("index not ready")
        q_tokens = _tokenize(query)
        doc_ids = [c.doc_id for c in self.chunks]
        bm25_raw = np.array([self._bm25.score(q_tokens, i) for i in range(len(self.chunks))])
        bm25_norm = bm25_raw / max(float(bm25_raw.max()), 1e-9)
        bm25_order = np.argsort(-bm25_raw)
        bm25_ids = [doc_ids[i] for i in bm25_order if bm25_raw[i] > 0][:overfetch]
        q_vec = self._vectorizer.transform([query])
        tfidf_scores = cosine_similarity(q_vec, self._matrix)[0]
        tfidf_norm = tfidf_scores / max(float(tfidf_scores.max()), 1e-9)
        dense_ids: list[str] = []
        dense_score_map: dict[str, float] = {}
        if self._dense_encoder is not None and self._vector_store is not None:
            q_dense = self._dense_encoder.encode([query])[0]
            dense_hits = self._vector_store.search(q_dense, top_k=overfetch, metadata_filter=metadata_filter)
            for did, sc in dense_hits:
                dense_ids.append(did)
                dense_score_map[did] = sc
        if use_rrf and dense_ids:
            fused = reciprocal_rank_fusion([bm25_ids, dense_ids], k=60)
            fusion_label = "rrf"
        else:
            hybrid = alpha * bm25_norm + (1 - alpha) * tfidf_norm
            order = np.argsort(-hybrid)
            fused = [(doc_ids[i], float(hybrid[i])) for i in order if hybrid[i] > 0][:overfetch]
            fusion_label = "weighted"
        hits: list[Hit] = []
        for did, fuse_score in fused:
            idx = self._id_to_idx.get(did)
            if idx is None:
                continue
            chunk = self.chunks[idx]
            if metadata_filter:
                ok = all(str(chunk.metadata.get(k, "")) == str(v) for k, v in metadata_filter.items())
                if not ok:
                    continue
            hits.append(
                Hit(
                    chunk=chunk,
                    score=float(fuse_score),
                    bm25=float(bm25_norm[idx]),
                    tfidf=float(tfidf_norm[idx]),
                    dense=float(dense_score_map.get(did, 0.0)),
                    rrf=float(fuse_score) if fusion_label == "rrf" else 0.0,
                )
            )
            if len(hits) >= top_k * 3:
                break
        return hits


def build_index(corpus_dir: Path, chunk_size: int = 500, chunk_overlap: int = 80, **kwargs) -> CorpusIndex:
    idx = CorpusIndex()
    idx.build(corpus_dir, chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
    return idx

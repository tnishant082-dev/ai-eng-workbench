from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.dataeng.quality import corpus_quality, tabular_quality
from app.rag.index import build_index


def run_etl(
    corpus_dir: Path,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
    catalog_path: Path | None = None,
    lineage_path: Path | None = None,
    dataset_path: Path | None = None,
) -> dict[str, Any]:
    """Corpus → chunks ETL with lineage + catalog update."""
    index = build_index(corpus_dir, chunk_size, chunk_overlap)
    chunks = [
        {
            "doc_id": c.doc_id,
            "title": c.title,
            "path": c.path,
            "chars": len(c.text),
            "metadata": c.metadata,
        }
        for c in index.chunks
    ]
    ts = datetime.now(timezone.utc).isoformat()
    cq = corpus_quality(corpus_dir)
    tq = None
    if dataset_path and Path(dataset_path).exists():
        from app.ml.pipeline import FEATURE_COLS, TARGET
        tq = tabular_quality(Path(dataset_path), FEATURE_COLS + [TARGET])
    catalog = {
        "updated_at": ts,
        "datasets": [
            {
                "id": "corpus_markdown",
                "path": str(corpus_dir),
                "format": "markdown",
                "docs": index.doc_count,
                "chunks": len(chunks),
                "quality": cq,
                "description": "Synthetic AI-engineering knowledge base for offline RAG demos",
            },
            {
                "id": "churn_demo",
                "path": "data/datasets/churn_demo.csv",
                "format": "csv",
                "rows": 200,
                "quality": tq,
                "versions": "data/datasets/versions/",
                "description": "Synthetic tabular churn set for demo-scale ML pipeline",
            },
        ],
    }
    lineage = {
        "updated_at": ts,
        "edges": [
            {
                "from": "data/corpus/*.md",
                "to": "CorpusIndex.chunks",
                "transform": "chunk_text + hybrid BM25/TFIDF fit",
                "params": {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap},
            },
            {
                "from": "CorpusIndex.chunks",
                "to": "RAG citations",
                "transform": "hybrid retrieve → expand → rerank",
            },
            {
                "from": "data/datasets/churn_demo.csv",
                "to": "experiments/*/model.joblib",
                "transform": "sklearn Pipeline train",
            },
        ],
    }
    if catalog_path:
        Path(catalog_path).parent.mkdir(parents=True, exist_ok=True)
        Path(catalog_path).write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    if lineage_path:
        Path(lineage_path).parent.mkdir(parents=True, exist_ok=True)
        Path(lineage_path).write_text(json.dumps(lineage, indent=2), encoding="utf-8")
    return {
        "docs": index.doc_count,
        "chunks": len(chunks),
        "catalog": catalog,
        "lineage": lineage,
        "quality": {"corpus": cq, "tabular": tq},
        "sample_chunks": chunks[:5],
        "status": "ok",
    }

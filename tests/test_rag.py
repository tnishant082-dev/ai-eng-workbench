from app.core.config import get_settings
from app.rag.index import build_index
from app.rag.expand import multi_queries
from app.rag.rerank import rerank_with_meta
from app.rag.fusion import reciprocal_rank_fusion
from app.rag.retrieval_metrics import evaluate_retrieval, recall_at_k, mrr, ndcg_at_k
from app.rag.dense import HashingDenseEncoder
from app.llm.mock import MockProvider
from app.rag.service import RagService


def test_hybrid_search_rrf_and_rerank():
    settings = get_settings()
    idx = build_index(settings.corpus_dir)
    hits = idx.search("hybrid BM25 retrieval", top_k=5, use_rrf=True)
    assert hits
    assert idx.info()["dense_backend"] in ("hashing", "sentence_transformers", "tfidf_svd")
    rr, meta = rerank_with_meta("hybrid BM25 retrieval", hits, top_k=3)
    assert len(rr) <= 3
    assert meta["reranker"] in ("lightweight_lexical", "cross_encoder")


def test_multi_query_rag():
    settings = get_settings()
    idx = build_index(settings.corpus_dir)
    rag = RagService(idx, MockProvider(), settings.prompts_dir)
    out = rag.query("What is RAG and why use citations?", top_k=4)
    assert out["citations"]
    assert out["confidence"] >= 0
    assert out["retrieval"]["fusion"] == "rrf"
    assert any(e["type"] == "query_expand" for e in out["events"])


def test_multi_queries_helper():
    assert len(multi_queries("What is faithfulness evaluation?")) >= 2


def test_rrf_fusion():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]], k=60)
    assert fused[0][0] in ("a", "b")


def test_dense_hashing_encoder():
    enc = HashingDenseEncoder(n_features=64)
    assert enc.encode(["hello world", "retrieval"]).shape == (2, 64)


def test_retrieval_metrics_helpers():
    rel = {"doc1", "doc2"}
    retrieved = ["doc2", "x", "doc1"]
    assert recall_at_k(rel, retrieved, 3) == 1.0
    assert mrr(rel, retrieved) == 1.0
    assert ndcg_at_k(rel, retrieved, 3) > 0


def test_parent_child_build():
    settings = get_settings()
    idx = build_index(settings.corpus_dir, parent_child=True)
    assert idx.ready and len(idx.chunks) > 0


def test_evaluate_retrieval_on_golden():
    settings = get_settings()
    idx = build_index(settings.corpus_dir)

    def retrieve_fn(q):
        return [h.chunk.doc_id for h in idx.search(q, top_k=5)]

    cases = [
        {"id": "t1", "query": "hybrid search BM25", "relevant_doc_ids": ["23_hybrid_search"]},
        {"id": "t2", "query": "faithfulness evaluation", "relevant_doc_ids": ["08_eval_faithfulness"]},
    ]
    out = evaluate_retrieval(cases, retrieve_fn, k=5)
    assert out["summary"]["n_cases"] == 2

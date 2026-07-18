# Lightweight Embeddings and TF-IDF

Dense embeddings (sentence-transformers) capture paraphrase similarity but add install weight.
TF-IDF with cosine similarity is offline-friendly, deterministic, and enough for small corpora.
Hybrid retrieval (BM25 + dense) often beats either alone on mixed keyword/semantic queries.
For portfolio demos, TF-IDF keeps `pip install` clean and CI green without GPU.

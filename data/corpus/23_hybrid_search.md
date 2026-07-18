# Hybrid Search Notes

Combine lexical scores (TF-IDF/BM25) with dense similarity via weighted sum or RRF.
Normalize scores before fusion; raw cosine and BM25 live on different scales.
For small corpora, TF-IDF alone often matches hybrid quality.
Log both score components in traces when debugging bad retrievals.

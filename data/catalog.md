# Data catalog

| ID | Path | Format | Description |
|---|---|---|---|
| corpus_markdown | `data/corpus/*.md` | markdown | Synthetic AI-engineering knowledge base (~30 docs) |
| golden_rag | `data/golden/rag_cases.jsonl` | jsonl | Offline RAG eval cases |
| churn_demo | `data/datasets/churn_demo.csv` | csv | Synthetic 200-row churn table for demo ML |

Run `POST /api/v1/dataeng/etl` to refresh `data/catalog.json` and `data/lineage.json`.

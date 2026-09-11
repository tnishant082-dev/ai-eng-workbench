# Tradeoffs

| Choice | Benefit | Cost |
|---|---|---|
| TF-IDF/BM25 vs dense vectors | Offline, fast, no model download | Weaker semantic recall on paraphrases |
| Mock LLM default | Reliable demos | Answers are template-grounded, not generative prose |
| SQLite | Simple deploy | Not multi-writer / multi-region |
| In-process cache/queue | No infra | Lost on process restart; single node |
| Heuristic evals | Free, deterministic | Correlates loosely with human judgment |
| Synthetic churn CSV | Clear ML loop | Not a real business dataset |
| Streamlit | Ship UI quickly | Limited product polish vs custom SPA |

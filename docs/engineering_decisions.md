# Engineering decisions

| Decision | Choice | Why |
|---|---|---|
| Default dense | HashingVectorizer proxy | Zero heavy deps; clearly labeled |
| Fusion | RRF | Robust hybrid without score calibration |
| Vector default | In-memory numpy | Always works offline |
| pgvector | Interface stub | Documented; skip real Postgres |
| Rerank | Lexical / CE if present | Passes pytest without ST |
| Tracking | Filesystem ≈ MLflow API | Same surface |
| Authz | API key → role map | Simple local RBAC |
| Evals | Heuristics + judge stub | Honest without paid judge |
| LangSmith | Local recorder if no key | No pretend cloud |

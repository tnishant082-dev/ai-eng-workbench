# Citation Presence Evals

Citation presence is a cheap binary metric: answer includes at least one valid doc_id.
Stricter checks verify that cited chunks actually appear in the retrieval set for that query.
Missing citations are a product bug even when the answer text looks fluent.
Report citation rate alongside latency and faithfulness in the eval summary.

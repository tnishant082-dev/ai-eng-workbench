# Corpus Search Tool Spec

`corpus_search(query, top_k)` wraps the same index as RAG retrieve.
Agents use it when the user asks to look something up mid-task.
Return doc_id, title, score, and a short snippet for the model and traces.
Keep top_k small (3–5) to control context growth.

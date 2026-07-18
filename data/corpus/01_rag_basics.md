# RAG Basics

Retrieval-Augmented Generation (RAG) grounds LLM answers in external documents.
A typical pipeline: ingest → chunk → embed → retrieve top-k → prompt with context → generate.

Citations should map answer claims back to retrieved chunk IDs so reviewers can audit grounding.
Chunk size and overlap trade off recall vs. precision; 300–800 tokens with 10–20% overlap is a common starting point.

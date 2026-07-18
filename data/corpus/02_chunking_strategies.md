# Chunking Strategies

Fixed-size chunking is simple and works with TF-IDF or dense embeddings.
Semantic chunking splits on headings or topic boundaries and often improves citation quality.
Recursive character splitters (paragraph → sentence → token) are popular in LangChain-style stacks.
Always store source path, title, and chunk index for observability and evals.

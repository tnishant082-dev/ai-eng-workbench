# Context Window Management

Stuffing too many chunks overflows the context window and dilutes attention.
Rank by relevance, then pack until a token budget (e.g. 2k tokens) is hit.
Drop lowest-scoring chunks first; keep citation metadata even if text is truncated.
Mock LLMs should still respect a max context char budget for realism.

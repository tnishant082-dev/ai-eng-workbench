# LLM Gateway Routing

A tiny gateway routes `mock` vs `openai-compatible` backends behind one interface.
Log model name, estimated tokens, and latency on every call for cost awareness.
Fall back to mock when no API key is set so CI and offline demos never break.
Gateways are also the place for rate limits, retries, and redaction hooks.

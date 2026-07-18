# Latency Budgets

Track p50/p95 latency for retrieve, LLM, and end-to-end query paths.
Mock LLM latency stubs keep evals comparable across machines.
A local TF-IDF retrieve under 50ms is typical for tens of documents.
Gateways should attach latency_ms to every response for the traces UI.
